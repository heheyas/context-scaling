#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Rewrite structured JSON prompts from T2I parquets into natural language captions.

Reads structured prompts from parquet files, sends to LLM for rewriting,
outputs JSONL with checkpoint/resume support.

Usage:
    # Gemini backend
    python -m rewrite_prompts.parquet_rewrite \
        --dataset_name <DATASET_NAME> \
        --num_parquets 10 --rows_per_parquet 100 \
        --backend gemini --api_config /path/to/keys.json \
        --target_tokens 500 \
        --output rewritten/parquet_nl_500tok.jsonl \
        --workers 16

    # Seed PSM backend
    python -m rewrite_prompts.parquet_rewrite \
        --dataset_name <DATASET_NAME> \
        --num_parquets 0 \
        --backend seed --psm "<RAY_PSM>" \
        --target_tokens 500 \
        --output rewritten/parquet_nl_500tok.jsonl

    # Generic OpenAI API backend
    python -m rewrite_prompts.parquet_rewrite \
        --dataset_name <DATASET_NAME> \
        --num_parquets 5 --rows_per_parquet 50 \
        --backend openai --api_url http://localhost:8000/v1 --api_key test \
        --target_tokens 500 \
        --output rewritten/parquet_nl_500tok.jsonl
"""

import os
import sys
import io
import base64
import json
import random
import logging
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.parquet as pq
import pyarrow.fs as pf
from PIL import Image
from tqdm import tqdm

from rewrite_prompts.base import KeyPool, load_api_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Default system prompt ────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT_TEXT = """You are a professional image caption writer. Given a structured JSON description of an image, rewrite it as a natural language description. The description should capture all the key information (objects, attributes, spatial relationships, style, lighting, atmosphere) from the structured prompt, but expressed as flowing natural language paragraphs rather than JSON fields.

CRITICAL LENGTH REQUIREMENT: Your response MUST be approximately {target_tokens} tokens (roughly {target_words} words). This is a hard requirement. Write detailed, rich, flowing paragraphs. Describe every object's appearance, material, color, texture, position. Describe the lighting direction, quality, shadows. Describe the atmosphere, mood, background details. Keep writing until you reach approximately {target_words} words. DO NOT stop early.

Output ONLY the natural language description, no JSON, no markdown, no explanations."""

DEFAULT_SYSTEM_PROMPT_VISION = """You are a professional image caption writer. You are given a structured JSON description AND the actual image. Using BOTH the structured JSON and the visual content of the image, write a detailed natural language description.

The description should capture all visual details you can see in the image, guided by the structure in the JSON (objects, attributes, spatial relationships, style, lighting, atmosphere), expressed as flowing natural language paragraphs.

CRITICAL LENGTH REQUIREMENT: Your response MUST be approximately {target_tokens} tokens (roughly {target_words} words). This is a hard requirement. Describe every visual detail: objects, colors, textures, materials, lighting, shadows, spatial arrangement, background, atmosphere, composition. Keep writing until you reach approximately {target_words} words. DO NOT stop early.

Output ONLY the natural language description, no JSON, no markdown, no explanations."""


# ── HDFS / Parquet reading ───────────────────────────────────────

def init_arrow_hdfs_fs(path):
    """Initialize an arrow filesystem for the given path."""
    if path.startswith("hdfs://"):
        return pf.HadoopFileSystem.from_uri(path)
    return pf.LocalFileSystem()


def resolve_parquet_paths_from_datacard(datacard_name, num_parquets=0):
    """Resolve parquet paths from a DataCard name."""
    from bytedmerlin.datacard import DataCard
    files = DataCard(datacard_name).get_data_file_paths()
    all_paths = [f for f in files if f.endswith(".parquet")]
    log.info("DataCard '%s': %d total parquet files", datacard_name, len(all_paths))

    if num_parquets > 0 and num_parquets < len(all_paths):
        random.seed(42)
        all_paths = random.sample(all_paths, num_parquets)
        log.info("Sampled %d parquets", num_parquets)
    return all_paths


def resolve_parquet_paths_from_dataset_info(dataset_name, num_parquets=0):
    """Resolve parquet paths via context-scaling's dataset_info (legacy)."""
    # Add context-scaling to path if available
    CONTEXT_SCALING_ROOT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "..", "context-scaling"
    )
    if os.path.isdir(CONTEXT_SCALING_ROOT):
        sys.path.insert(0, CONTEXT_SCALING_ROOT)

    from data.dataset_info import DATASET_INFO
    from data.parquet_utils import get_parquet_data_paths

    meta = None
    for group_name, group_datasets in DATASET_INFO.items():
        if dataset_name in group_datasets:
            meta = group_datasets[dataset_name]
            break
    if meta is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in DATASET_INFO")

    data_dir = meta['data_dir']
    total_files = meta['num_files']

    all_paths = get_parquet_data_paths([data_dir], [total_files])
    log.info("Dataset '%s': %d total parquet files", dataset_name, len(all_paths))

    if num_parquets > 0 and num_parquets < len(all_paths):
        random.seed(42)
        all_paths = random.sample(all_paths, num_parquets)
        log.info("Sampled %d parquets", num_parquets)
    return all_paths


def resolve_parquet_paths(args):
    """Resolve parquet paths from --datacard or --dataset_name."""
    if args.datacard:
        return resolve_parquet_paths_from_datacard(args.datacard, args.num_parquets)
    elif args.dataset_name:
        return resolve_parquet_paths_from_dataset_info(args.dataset_name, args.num_parquets)
    else:
        raise ValueError("Specify --datacard or --dataset_name")


def resize_image_max_side(img, max_side):
    """Resize PIL image so the long side <= max_side, keeping aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def image_to_base64(img):
    """Convert PIL image to base64 JPEG string."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def iter_prompts_from_parquets(parquet_paths, rows_per_parquet=0, done_ids=None,
                               with_image=False, max_image_size=512):
    """Lazily yield items from parquets, skipping already-done IDs.

    Yields dicts: {id, parquet_file, structured_prompt, image_base64 (optional)}
    """
    if done_ids is None:
        done_ids = set()

    for pq_idx, pf_path in enumerate(parquet_paths):
        pf_basename = os.path.basename(pf_path)
        count = 0
        try:
            fs = init_arrow_hdfs_fs(pf_path)
            with fs.open_input_file(pf_path) as f:
                fr = pq.ParquetFile(f)
                for rg_idx in range(fr.num_row_groups):
                    df = fr.read_row_group(rg_idx).to_pandas()
                    n_rows = len(df) if rows_per_parquet <= 0 else min(rows_per_parquet, len(df))
                    for row_idx in range(n_rows):
                        item_id = f"pq{pq_idx:04d}_rg{rg_idx}_row{row_idx}"
                        if item_id in done_ids:
                            continue
                        row = df.iloc[row_idx]
                        try:
                            raw_json = json.loads(row['inputs'])[-5]["text"]
                            json.loads(raw_json)  # validate

                            item = {
                                "id": item_id,
                                "parquet_file": pf_basename,
                                "structured_prompt": raw_json,
                            }

                            if with_image:
                                try:
                                    img_bytes = row['images'][0]
                                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                    img = resize_image_max_side(img, max_image_size)
                                    item["image_base64"] = image_to_base64(img)
                                except Exception:
                                    item["image_base64"] = None

                            yield item
                            count += 1
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue
                    if rows_per_parquet > 0:
                        break
            log.info("[%d/%d] %s: %d new items", pq_idx + 1, len(parquet_paths), pf_basename, count)
        except Exception as e:
            log.error("[%d/%d] Failed: %s: %s", pq_idx + 1, len(parquet_paths), pf_path, e)


# ── Client builder ───────────────────────────────────────────────

def build_client(args):
    """Build the appropriate RewriteClient based on --backend."""
    if args.backend == "gemini":
        from rewrite_prompts.gemini3pro import GeminiRewriteClient
        configs = load_api_config(args.api_config)
        key_pool = KeyPool(configs, api_version="2024-03-01-preview")
        log.info("Gemini KeyPool: %d key(s)", key_pool.size)
        return GeminiRewriteClient(
            key_pool=key_pool,
            model=args.model or "gemini-3-pro-preview-new",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    elif args.backend == "gpt":
        from rewrite_prompts.gpt import GPTRewriteClient
        configs = load_api_config(args.api_config)
        key_pool = KeyPool(configs, api_version="2024-02-01")
        log.info("GPT KeyPool: %d key(s)", key_pool.size)
        return GPTRewriteClient(
            key_pool=key_pool,
            model=args.model or "gpt-5-chat-2025-08-07",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    elif args.backend == "seed":
        from rewrite_prompts.seed import SeedRewriteClient
        return SeedRewriteClient(
            psm=args.psm,
            thinking=args.thinking,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    elif args.backend == "openai":
        from rewrite_prompts.openai_generic import OpenAIGenericRewriteClient
        return OpenAIGenericRewriteClient(
            base_url=args.api_url,
            api_key=args.api_key or "test",
            model=args.model or "default",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    elif args.backend == "vllm_qwen":
        from rewrite_prompts.vllm_qwen import VllmQwenRewriteClient
        return VllmQwenRewriteClient(
            url=args.api_url or "http://localhost:8000",
            model=args.model,
            max_tokens=args.max_tokens,
            min_tokens=getattr(args, 'min_tokens', 0),
            temperature=args.temperature,
            thinking=args.thinking,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")


# ── Checkpoint ───────────────────────────────────────────────────

def load_checkpoint(output_path):
    """Load already-completed IDs from the output JSONL."""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                done.add(item["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# ── Rewrite worker ───────────────────────────────────────────────

def rewrite_one(client, item, system_prompt, input_template, target_tokens, tokenizer=None):
    """Rewrite a single structured prompt to natural language."""
    try:
        result = client.rewrite(
            item["structured_prompt"],
            system_prompt,
            input_template=input_template,
            image_base64=item.get("image_base64"),
        )
        rewritten = result["rewritten_prompt"]

        # Count actual tokens if tokenizer available
        actual_tokens = None
        if tokenizer is not None:
            actual_tokens = len(tokenizer.encode(rewritten))

        return {
            "id": item["id"],
            "parquet_file": item["parquet_file"],
            "original_structured_prompt": item["structured_prompt"],
            "rewritten_prompt": rewritten,
            "target_tokens": target_tokens,
            "actual_tokens": actual_tokens,
            "rewrite_model": None,  # filled by caller
            "reasoning": result.get("reasoning"),
            "usage": result.get("usage", {}),
        }
    except Exception as e:
        log.error("Failed to rewrite id=%s: %s", item["id"], e)
        return None


BATCH_SEPARATOR = "<<<SEPARATOR_a1b2c3d4e5f6>>>"

BATCH_SYSTEM_PROMPT = """You are a professional image caption writer. You will be given {batch_size} structured JSON descriptions of images, separated by "{separator}". For EACH one, rewrite it as a natural language description.

CRITICAL LENGTH REQUIREMENT: EACH of the {batch_size} descriptions MUST be approximately {target_tokens} tokens long (roughly {target_words} words each). This is NON-NEGOTIABLE. Do NOT shorten any description. Each description must be a detailed, rich, flowing paragraph of approximately {target_words} words. If the structured prompt is brief, elaborate extensively on visual details, textures, materials, colors, spatial arrangements, lighting, atmosphere, and composition to reach the required length.

Output ONLY the {batch_size} natural language descriptions, separated by exactly "{separator}". No JSON, no markdown, no numbering, no explanations. Just the {batch_size} long descriptions separated by the separator."""


def rewrite_batch(client, items, system_prompt, target_tokens, batch_size, tokenizer=None):
    """Rewrite a batch of structured prompts in a single API call."""
    try:
        # Build batched user message
        prompts = [item["structured_prompt"] for item in items]
        user_content = f"\n{BATCH_SEPARATOR}\n".join(prompts)

        result = client.rewrite(
            user_content,
            system_prompt,
            input_template="<prompt>",
        )
        raw_output = result["rewritten_prompt"]

        # Split output by separator
        parts = raw_output.split(BATCH_SEPARATOR)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) != len(items):
            log.warning("Batch split mismatch: expected %d, got %d. Falling back to individual.",
                        len(items), len(parts))
            return None

        # Build individual results
        results = []
        for item, rewritten in zip(items, parts):
            actual_tokens = None
            if tokenizer is not None:
                actual_tokens = len(tokenizer.encode(rewritten))
            results.append({
                "id": item["id"],
                "parquet_file": item["parquet_file"],
                "original_structured_prompt": item["structured_prompt"],
                "rewritten_prompt": rewritten,
                "target_tokens": target_tokens,
                "actual_tokens": actual_tokens,
                "rewrite_model": None,
                "reasoning": None,
                "usage": result.get("usage", {}),
            })
        return results

    except Exception as e:
        log.error("Batch rewrite failed (batch of %d): %s", len(items), e)
        return None


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rewrite structured JSON prompts from parquets to natural language"
    )

    # Data source (one of --datacard or --dataset_name required)
    parser.add_argument("--datacard", type=str, default=None,
                        help="DataCard name (e.g. <DATA_CARD_NAME>)")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Dataset name from dataset_info.py (legacy, requires context-scaling repo)")
    parser.add_argument("--num_parquets", type=int, default=0,
                        help="Number of parquets to sample (0=all)")
    parser.add_argument("--rows_per_parquet", type=int, default=0,
                        help="Rows per parquet (0=all)")

    # Image
    parser.add_argument("--with_image", action="store_true",
                        help="Extract image from parquet and send to LLM for vision-assisted rewrite")
    parser.add_argument("--max_image_size", type=int, default=512,
                        help="Max long side for image resize (default: 512)")

    # Target
    parser.add_argument("--target_tokens", type=int, required=True,
                        help="Target token count for the natural language caption")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL path (supports resume)")

    # Backend
    parser.add_argument("--backend", type=str, required=True,
                        choices=["gemini", "gpt", "seed", "openai", "vllm_qwen"],
                        help="Rewrite backend")
    parser.add_argument("--api_config", type=str, default=None,
                        help="API key config file (for gemini/gpt)")
    parser.add_argument("--psm", type=str, default=None,
                        help="Ray PSM address (for seed)")
    parser.add_argument("--api_url", type=str, default=None,
                        help="Base URL (for openai backend)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (for openai backend)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name override")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable thinking mode (for seed)")

    # Prompt
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="Path to system prompt file (optional, has built-in default)")
    parser.add_argument("--input_template", type=str, default="<prompt>",
                        help="Template for user message (default: raw prompt)")

    # Generation params
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--min_tokens", type=int, default=0,
                        help="Min tokens for vllm_qwen backend (forces model to generate at least N tokens)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Prompts per API call (>1 = batched mode, faster but may fail on split)")

    # Tokenizer for counting actual tokens
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer path for counting actual output tokens")

    # Multi-machine sharding
    parser.add_argument("--local_rank", type=int, default=0,
                        help="This machine's rank (0-indexed)")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of machines")

    args = parser.parse_args()

    # Validate
    if not args.datacard and not args.dataset_name:
        parser.error("Specify --datacard or --dataset_name")
    if args.backend in ("gemini", "gpt") and not args.api_config:
        parser.error(f"--api_config is required for backend={args.backend}")
    if args.backend == "seed" and not args.psm:
        parser.error("--psm is required for backend=seed")
    if args.backend == "openai" and not args.api_url:
        parser.error("--api_url is required for backend=openai")

    # System prompt
    target_words = int(args.target_tokens * 0.75)
    if args.system_prompt:
        with open(args.system_prompt, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    elif args.with_image:
        system_prompt = DEFAULT_SYSTEM_PROMPT_VISION.format(
            target_tokens=args.target_tokens, target_words=target_words)
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT_TEXT.format(
            target_tokens=args.target_tokens, target_words=target_words)

    # Tokenizer (optional)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        log.info("Loaded tokenizer: %s", args.tokenizer)

    # Auto-scale max_tokens for batch mode to prevent truncation
    if args.batch_size > 1:
        min_needed = int(args.target_tokens * args.batch_size * 1.2)  # 20% buffer
        if args.max_tokens < min_needed:
            log.info("Auto-scaling max_tokens: %d → %d (batch_size=%d × target=%d × 1.2)",
                     args.max_tokens, min_needed, args.batch_size, args.target_tokens)
            args.max_tokens = min_needed

    model_name = args.model or {
        "gemini": "gemini-3-pro-preview-new",
        "gpt": "gpt-5-chat-2025-08-07",
        "seed": "seed",
        "openai": "default",
        "vllm_qwen": "auto",
    }[args.backend]

    # Output is now a directory: one jsonl per parquet
    output_dir = args.output
    if output_dir.endswith(".jsonl"):
        output_dir = output_dir.rsplit(".", 1)[0]  # strip .jsonl extension
    os.makedirs(output_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("Parquet Structured Prompt → Natural Language Rewriter")
    log.info("=" * 60)
    log.info("Data source:   %s", args.datacard or args.dataset_name)
    log.info("Num parquets:  %s", args.num_parquets or "all")
    log.info("Rows/parquet:  %s", args.rows_per_parquet or "all")
    log.info("Target tokens: %d", args.target_tokens)
    log.info("Backend:       %s (%s)", args.backend, model_name)
    log.info("Workers:       %d", args.workers)
    log.info("Batch size:    %d", args.batch_size)
    if args.world_size > 1:
        log.info("Rank:          %d / %d", args.local_rank, args.world_size)
    log.info("With image:    %s (max_side=%d)", args.with_image, args.max_image_size)
    log.info("Output dir:    %s", output_dir)

    # 1. Resolve parquet paths
    log.info("")
    log.info("Resolving parquet paths...")
    parquet_paths = resolve_parquet_paths(args)

    # Shard parquets across machines
    if args.world_size > 1:
        total_parquets = len(parquet_paths)
        parquet_paths = sorted(parquet_paths)  # deterministic order across machines
        parquet_paths = parquet_paths[args.local_rank::args.world_size]
        log.info("Shard: rank %d gets %d parquets (of %d total)",
                 args.local_rank, len(parquet_paths), total_parquets)

    # 2. Build client
    client = build_client(args)

    # 3. Process parquets one by one, each with its own jsonl
    import time as _time
    global_start_time = _time.time()
    total_success = 0
    total_fail = 0

    for pq_global_idx, pf_path in enumerate(parquet_paths):
        pf_basename = os.path.basename(pf_path)
        pf_stem = pf_basename.rsplit(".", 1)[0]  # e.g. "shuffled_rank15_000000"
        jsonl_path = os.path.join(output_dir, f"{pf_stem}.jsonl")

        # Resume: load done IDs from this parquet's jsonl
        done_ids = load_checkpoint(jsonl_path)

        # Read items from this single parquet
        items = []
        try:
            fs = init_arrow_hdfs_fs(pf_path)
            with fs.open_input_file(pf_path) as f:
                fr = pq.ParquetFile(f)
                for rg_idx in range(fr.num_row_groups):
                    df = fr.read_row_group(rg_idx).to_pandas()
                    n_rows = len(df) if args.rows_per_parquet <= 0 else min(args.rows_per_parquet, len(df))
                    for row_idx in range(n_rows):
                        item_id = f"rg{rg_idx}_row{row_idx}"
                        if item_id in done_ids:
                            continue
                        row = df.iloc[row_idx]
                        try:
                            raw_json = json.loads(row['inputs'])[-5]["text"]
                            json.loads(raw_json)  # validate
                            item = {
                                "id": item_id,
                                "parquet_file": pf_basename,
                                "structured_prompt": raw_json,
                            }
                            if args.with_image:
                                try:
                                    img_bytes = row['images'][0]
                                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                    img = resize_image_max_side(img, args.max_image_size)
                                    item["image_base64"] = image_to_base64(img)
                                except Exception:
                                    item["image_base64"] = None
                            items.append(item)
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue
                    if args.rows_per_parquet > 0:
                        break
        except Exception as e:
            log.error("[%d/%d] Failed to read %s: %s", pq_global_idx + 1, len(parquet_paths), pf_path, e)
            continue

        if not items:
            log.info("[%d/%d] %s: skip (0 new items, %d done)",
                     pq_global_idx + 1, len(parquet_paths), pf_basename, len(done_ids))
            continue

        log.info("[%d/%d] %s: %d to process, %d done",
                 pq_global_idx + 1, len(parquet_paths), pf_basename, len(items), len(done_ids))

        # Process this parquet's items
        out_lock = threading.Lock()
        out_f = open(jsonl_path, "a", encoding="utf-8")
        success_count = 0
        fail_count = 0

        pbar = tqdm(total=len(items), desc=pf_stem, unit="prompt", dynamic_ncols=True)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for item in items:
                f = pool.submit(rewrite_one, client, item, system_prompt,
                                args.input_template, args.target_tokens, tokenizer)
                futures[f] = item
                # Drain completed futures to bound memory
                done_futures = [ft for ft in futures if ft.done()]
                for ft in done_futures:
                    result = ft.result()
                    if result is not None:
                        result["rewrite_model"] = model_name
                        with out_lock:
                            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            out_f.flush()
                        success_count += 1
                    else:
                        fail_count += 1
                    pbar.update(1)
                    del futures[ft]
            # Wait for remaining
            for ft in as_completed(futures):
                result = ft.result()
                if result is not None:
                    result["rewrite_model"] = model_name
                    with out_lock:
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_f.flush()
                    success_count += 1
                else:
                    fail_count += 1
                pbar.update(1)

        pbar.close()
        out_f.close()

        total_success += success_count
        total_fail += fail_count
        elapsed = _time.time() - global_start_time
        avg_speed = total_success / elapsed if elapsed > 0 else 0
        log.info("[%d/%d] %s: %d success, %d failed → %s  (avg %.1f prompts/min)",
                 pq_global_idx + 1, len(parquet_paths), pf_basename,
                 success_count, fail_count, jsonl_path, avg_speed * 60)

    elapsed = _time.time() - global_start_time
    avg_speed = total_success / elapsed if elapsed > 0 else 0
    log.info("Done. %d parquets, %d success, %d failed. Total %.0fs, avg %.1f prompts/min.",
             len(parquet_paths), total_success, total_fail, elapsed, avg_speed * 60)

    if args.backend in ("gemini", "gpt") and hasattr(client, "key_pool"):
        log.info("Key pool stats: %s", client.key_pool.stats())


if __name__ == "__main__":
    main()
