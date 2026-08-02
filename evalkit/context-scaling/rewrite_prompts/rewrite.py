#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Multi-threaded prompt rewrite pipeline with checkpoint/resume.

Usage:
    # GPT
    python -m rewrite_prompts.rewrite \
        --benchmark geneval2 \
        --data_path benchmarks/GenEval2/geneval2_data.jsonl \
        --backend gpt \
        --output rewritten/geneval2_gpt.jsonl \
        --system_prompt /path/to/system_prompt.txt \
        --api_config /path/to/api_config.json \
        --workers 16

    # Gemini
    python -m rewrite_prompts.rewrite \
        --benchmark oneig \
        --data_path benchmarks/OneIG-Benchmark/OneIG-Bench.csv \
        --backend gemini \
        --output rewritten/oneig_gemini.jsonl \
        --system_prompt /path/to/system_prompt.txt \
        --api_config /path/to/keys.txt \
        --workers 16

    # Seed
    python -m rewrite_prompts.rewrite \
        --benchmark dpgbench \
        --data_path benchmarks/ELLA/dpg_bench/dpg_bench.csv \
        --backend seed \
        --output rewritten/dpgbench_seed.jsonl \
        --system_prompt /path/to/system_prompt.txt \
        --psm "<RAY_PSM>" \
        --workers 16
"""

import os
import re
import json
import logging
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from rewrite_prompts.benchmark_adapters import load_prompts, BENCHMARKS, DEFAULT_DATA_PATHS
from rewrite_prompts.base import KeyPool, load_api_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_checkpoint(output_path: str, retry_failed: bool = False) -> set:
    """Load already-completed IDs from the output JSONL.

    If retry_failed=True, IDs with empty rewritten_prompt are excluded
    (treated as failed, will be retried).
    """
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
                if retry_failed and not item.get("rewritten_prompt", "").strip():
                    continue  # skip failed items so they get retried
                done.add(item["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


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
    elif args.backend == "qwen":
        from rewrite_prompts.qwen import QwenRewriteClient, resolve_model_path
        model_path = resolve_model_path(args.model or "qwen3.5-9b")
        log.info("Qwen model path: %s", model_path)
        return QwenRewriteClient(
            model_path=model_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            thinking=args.thinking,
            batch_size=getattr(args, 'batch_size', 1),
        )
    elif args.backend == "vllm_qwen":
        from rewrite_prompts.vllm_qwen import VllmQwenRewriteClient
        return VllmQwenRewriteClient(
            url=args.api_url or "http://localhost:8000",
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            thinking=args.thinking,
        )
    elif args.backend == "claude":
        from rewrite_prompts.claude import ClaudeRewriteClient
        api_key = args.api_key
        if not api_key and args.api_config:
            # Read single key from file (first non-empty, non-comment line)
            with open(args.api_config, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        api_key = line.split()[0]
                        break
        if not api_key:
            raise ValueError("Claude backend requires --api_key or --api_config with a key")
        return ClaudeRewriteClient(
            api_key=api_key,
            base_url=args.api_url or "https://openrouter.ai/api/v1",
            model=args.model or "anthropic/claude-opus-4",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    elif args.backend == "openai":
        from rewrite_prompts.openai_generic import OpenAIGenericRewriteClient
        return OpenAIGenericRewriteClient(
            base_url=args.api_url or "http://localhost:8000/v1",
            api_key=args.api_key or "test",
            model=args.model or "default",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            reasoning=args.thinking,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")


def load_from_jsonl(jsonl_path: str) -> list:
    """Load prompts from a previous rewrite JSONL for multi-stage pipelines.

    Uses rewritten_prompt as the new prompt input, preserves original_prompt.
    """
    prompts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompts.append({
                "id": item["id"],
                "benchmark": item.get("benchmark", "custom"),
                "category": item.get("category"),
                "prompt": item["rewritten_prompt"],  # stage N output → stage N+1 input
                "original_prompt": item.get("original_prompt", ""),
            })
    return prompts


def rewrite_one(client, item, system_prompt, input_template, width, height):
    """Rewrite a single prompt item. Returns the output record or None on failure."""
    prompt_id = item["id"]
    prompt = item["prompt"]

    # Per-prompt width/height override (if present in the data)
    w = item.get("width") or width
    h = item.get("height") or height

    # Support <original_prompt> placeholder in template (for multi-stage pipelines)
    template = input_template
    original_prompt = item.get("original_prompt")
    if original_prompt and "<original_prompt>" in template:
        template = template.replace("<original_prompt>", original_prompt)

    try:
        result = client.rewrite(prompt, system_prompt, input_template=template,
                                width=w, height=h)
        rewritten = result["rewritten_prompt"]
        # Strip <think>...</think>, <scene_imagine>...</scene_imagine>, and
        # <layout>...</layout> tags, keeping only the structured JSON after them.
        # Closing tags can be malformed (e.g., Gemini sometimes emits "</g>think>"),
        # so we use a tolerant regex for </think>.
        rewritten = re.sub(
            r"<think>.*?</[^>]*think>\s*",
            "", rewritten, flags=re.DOTALL,
        )
        rewritten = re.sub(
            r"<scene_imagine>.*?</scene_imagine>\s*",
            "", rewritten, flags=re.DOTALL,
        )
        rewritten = re.sub(
            r"<layout>.*?</layout>\s*",
            "", rewritten, flags=re.DOTALL,
        ).strip()
        # Fallback: if a <think> opener is still present (unclosed/malformed) and
        # the rewritten content also contains a JSON object, drop everything before
        # the first '{' so the JSON parses cleanly.
        if "<think>" in rewritten and "{" in rewritten:
            first_brace = rewritten.find("{")
            rewritten = rewritten[first_brace:].strip()
        out = {
            "id": prompt_id,
            "benchmark": item["benchmark"],
            "category": item.get("category"),
            "original_prompt": original_prompt or prompt,
            "rewritten_prompt": rewritten,
            "rewrite_model": None,  # filled by caller
            "rewrite_reasoning": result.get("reasoning"),
            "usage": result.get("usage", {}),
        }
        # Preserve per-prompt dimensions in output for downstream inference
        if item.get("width"): out["width"] = item["width"]
        if item.get("height"): out["height"] = item["height"]
        return out
    except Exception as e:
        log.error("Failed to rewrite id=%s: %s", prompt_id, e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Multi-threaded prompt rewrite pipeline")

    parser.add_argument("--benchmark", type=str, default=None,
                        choices=list(BENCHMARKS),
                        help="Benchmark name (required unless --input_jsonl is used)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to benchmark data file (required unless --input_jsonl is used)")
    parser.add_argument("--input_jsonl", type=str, default=None,
                        help="Input from a previous rewrite JSONL (for multi-stage pipelines). "
                             "Uses rewritten_prompt as input, preserves original_prompt. "
                             "Supports <original_prompt> placeholder in --input_template.")
    parser.add_argument("--backend", type=str, required=True,
                        choices=["gemini", "gpt", "seed", "qwen", "vllm_qwen", "openai", "claude"],
                        help="Rewrite backend")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL path (supports resume)")
    parser.add_argument("--system_prompt", type=str, required=True,
                        help="Path to system prompt file")
    parser.add_argument("--input_template", type=str, default="<prompt>",
                        help="Template for user message. <prompt> is replaced with the actual "
                             "prompt. Default '<prompt>' sends raw prompt. "
                             "Example: 'Rewrite this T2I prompt:\\n<prompt>'")

    # Backend-specific
    parser.add_argument("--api_config", type=str, default=None,
                        help="Path to API key config (JSON or text file, for gemini/gpt)")
    parser.add_argument("--psm", type=str, default=None,
                        help="Ray PSM address (for seed)")
    parser.add_argument("--api_url", type=str, default=None,
                        help="vLLM serving URL (for vllm_qwen backend, default: http://localhost:8000)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name override (default per backend)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (for claude/openai backends)")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable thinking mode (for seed)")

    # Generation params
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width (for <width> placeholder in input_template)")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height (for <height> placeholder in input_template)")
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel threads")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per GPU (for qwen backend)")
    parser.add_argument("--max_prompts", type=int, default=0,
                        help="Max prompts to process (0=all)")
    parser.add_argument("--retry_failed", action="store_true",
                        help="Retry items with empty rewritten_prompt")

    args = parser.parse_args()

    # Auto-resolve data_path from benchmark name if not provided
    if not args.data_path and args.benchmark and args.benchmark in DEFAULT_DATA_PATHS:
        args.data_path = DEFAULT_DATA_PATHS[args.benchmark]
        log.info("Auto-resolved data_path: %s", args.data_path)

    # Validate args
    if not args.input_jsonl and (not args.benchmark or not args.data_path):
        parser.error("--benchmark and --data_path are required unless --input_jsonl is used")
    if args.backend in ("gemini", "gpt") and not args.api_config:
        parser.error(f"--api_config is required for backend={args.backend}")
    if args.backend == "seed" and not args.psm:
        parser.error("--psm is required for backend=seed")

    # Load data
    system_prompt = load_system_prompt(args.system_prompt)
    if args.input_jsonl:
        prompts = load_from_jsonl(args.input_jsonl)
        log.info("Loaded %d prompts from previous rewrite: %s", len(prompts), args.input_jsonl)
    else:
        prompts = load_prompts(args.benchmark, args.data_path)

    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]

    # Resume
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # If retrying failed items, first clean the output file:
    # remove empty entries so they don't accumulate on repeated retries
    if args.retry_failed and os.path.exists(args.output):
        lines = []
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if item.get("rewritten_prompt", "").strip():
                        lines.append(line)
                except (json.JSONDecodeError, KeyError):
                    continue
        removed = sum(1 for line in open(args.output) if line.strip()) - len(lines)
        if removed > 0:
            with open(args.output, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
            log.info("Cleaned %d empty entries from %s", removed, args.output)

    done_ids = load_checkpoint(args.output, retry_failed=args.retry_failed)
    todo = [p for p in prompts if p["id"] not in done_ids]

    model_name = args.model or {
        "gemini": "gemini-3-pro-preview-new",
        "gpt": "gpt-4o-2024-11-20",
        "seed": "seed",
        "qwen": "qwen3.5-9b",
        "vllm_qwen": "auto",
        "claude": "anthropic/claude-opus-4",
        "openai": "default",
    }.get(args.backend, args.backend)

    log.info("=" * 60)
    log.info("Prompt Rewrite Pipeline")
    log.info("=" * 60)
    log.info("Benchmark:  %s", args.benchmark)
    log.info("Backend:    %s (%s)", args.backend, model_name)
    log.info("Prompts:    %d total, %d done, %d to process",
             len(prompts), len(done_ids), len(todo))
    log.info("Workers:    %d", args.workers)
    log.info("Template:   %s", args.input_template)
    log.info("Output:     %s", args.output)

    if not todo:
        log.info("All prompts already processed. Nothing to do.")
        return

    # Build client
    client = build_client(args)

    # Run
    out_lock = threading.Lock()
    out_f = open(args.output, "a", encoding="utf-8")
    success_count = 0
    fail_count = 0

    pbar = tqdm(total=len(todo), desc="Rewriting", unit="prompt", dynamic_ncols=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(rewrite_one, client, item, system_prompt,
                        args.input_template, args.width, args.height): item
            for item in todo
        }
        for future in as_completed(futures):
            result = future.result()
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

    log.info("Done. %d success, %d failed.", success_count, fail_count)

    # Print KeyPool stats if applicable
    if args.backend in ("gemini", "gpt"):
        configs = load_api_config(args.api_config)
        # Re-access pool stats through client
        if hasattr(client, "key_pool"):
            log.info("Key pool stats: %s", client.key_pool.stats())


if __name__ == "__main__":
    main()
