#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Inference pipeline using OpenAI-compatible /v1/images/generations API (vllm-omni).

Drop-in replacement for generate.py but targets vllm-omni serving instead of
the custom QwenImage serving endpoint.

API difference:
  - generate.py:        POST /generate        → PNG bytes
  - generate_openai.py: POST /v1/images/generations → JSON {data: [{b64_json}]}

Usage:
    python -m inference.generate_openai \
        --rewritten_jsonl rewritten/geneval2_gemini.jsonl \
        --url http://localhost:8091 \
        --output_dir outputs/geneval2_gemini_trial \
        --height 1024 --width 1024 \
        --num_steps 25 --cfg_scale 4.0 \
        --workers 8
"""

import json
import os
import base64
import logging
import argparse
import threading
from io import BytesIO
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from tqdm import tqdm

from inference.output_formatters import get_formatter
from inference.generate import compact_single_quote_json, load_rewritten_jsonl


def truncate_structured_prompt(prompt_str: str, max_chars: int) -> str:
    """Truncate a structured JSON prompt by removing elements from the end.

    If the prompt is plain text or not valid JSON, falls back to simple truncation.
    For structured prompts, removes elements one at a time until under max_chars.
    """
    if len(prompt_str) <= max_chars:
        return prompt_str

    # Try to parse as JSON and trim elements
    try:
        data = json.loads(prompt_str)
        if isinstance(data, dict) and "elements" in data and isinstance(data["elements"], list):
            while len(data["elements"]) > 1:
                data["elements"].pop()
                candidate = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
                if len(candidate) <= max_chars:
                    return candidate
            # Even with 1 element it's too long - return with 1 element
            return json.dumps(data, separators=(',', ':'), ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: simple truncation
    return prompt_str[:max_chars]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def send_generate_openai(url, prompt, height, width, num_steps, seed, cfg_scale,
                         negative_prompt, timeout=600, max_retries=10):
    """Send generation request via OpenAI /v1/images/generations API.

    Retries on 502 (backend not ready) with exponential backoff.
    Returns PIL Image.
    """
    body = {
        "prompt": prompt,
        "size": f"{width}x{height}",
        "seed": seed,
    }
    if num_steps and num_steps > 0:
        body["num_inference_steps"] = num_steps
    # For Flux/QwenImage models in vllm-omni:
    #   - guidance_scale: Flux guidance embed (not traditional CFG)
    #   - true_cfg_scale: actual classifier-free guidance (requires negative_prompt + --cfg-parallel-size 2)
    # We set both to maximize compatibility.
    if cfg_scale and cfg_scale > 1.0:
        body["guidance_scale"] = cfg_scale
        body["true_cfg_scale"] = cfg_scale
    if negative_prompt:
        body["negative_prompt"] = negative_prompt
    elif cfg_scale and cfg_scale > 1.0:
        # true_cfg requires negative_prompt; use empty string as default
        body["negative_prompt"] = ""

    import time as _time
    for attempt in range(max_retries):
        resp = requests.post(
            f"{url}/v1/images/generations",
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        if resp.status_code == 502:
            # Backend not ready yet (warmup), retry with backoff
            wait = min(2 ** attempt, 30)
            _time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        b64_str = data["data"][0]["b64_json"]
        img_bytes = base64.b64decode(b64_str)
        return Image.open(BytesIO(img_bytes))

    # Final attempt without catching
    resp.raise_for_status()


def main():
    parser = argparse.ArgumentParser(
        description="Inference pipeline using OpenAI-compatible API (vllm-omni)")

    # Input/output
    parser.add_argument("--rewritten_jsonl", type=str, required=True,
                        help="Rewritten JSONL from rewrite pipeline")
    parser.add_argument("--url", type=str, default="http://localhost:8091",
                        help="vllm-omni serving URL")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")

    # Generation params
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Override num images per prompt (default: auto per benchmark)")

    # Prompt selection
    parser.add_argument("--disable_rewritten", action="store_true",
                        help="Use original_prompt instead of rewritten_prompt")

    # Benchmark-specific
    parser.add_argument("--model_name", type=str, default="model",
                        help="Model name (used in OneIG-Bench directory structure)")
    parser.add_argument("--benchmark_data", type=str, default=None,
                        help="Path to benchmark data file (GenEval needs evaluation_metadata.jsonl)")

    # Concurrency
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_prompts", type=int, default=0,
                        help="Max prompts to process (0=all)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="HTTP request timeout in seconds")
    parser.add_argument("--max_prompt_chars", type=int, default=0,
                        help="Max prompt length in chars (0=unlimited). "
                             "Structured JSON prompts are trimmed by removing elements.")

    # Distributed sharding
    parser.add_argument("--rank", type=int, default=0,
                        help="Shard rank (0-indexed). Each rank processes items[rank::world_size].")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of shards. 1 = no sharding (default).")

    args = parser.parse_args()

    # Load items
    all_items = load_rewritten_jsonl(args.rewritten_jsonl)
    if args.max_prompts > 0:
        all_items = all_items[:args.max_prompts]

    if not all_items:
        log.error("No items to process")
        return

    # Distributed sharding: each rank processes a disjoint subset
    if args.world_size > 1:
        if args.rank < 0 or args.rank >= args.world_size:
            log.error("Invalid rank %d for world_size %d (must be 0..%d)",
                      args.rank, args.world_size, args.world_size - 1)
            return
        items = all_items[args.rank::args.world_size]
        log.info("Shard %d/%d: %d items (of %d total)",
                 args.rank, args.world_size, len(items), len(all_items))
    else:
        items = all_items

    if not items:
        log.error("No items for rank %d", args.rank)
        return

    benchmark = all_items[0]["benchmark"]
    log.info("Detected benchmark: %s", benchmark)

    formatter = get_formatter(
        benchmark=benchmark,
        output_dir=args.output_dir,
        model_name=args.model_name,
        benchmark_data=args.benchmark_data,
    )

    prompt_field = "original_prompt" if args.disable_rewritten else "rewritten_prompt"

    # Build task list with crash recovery
    tasks = []
    skip_count = 0
    recovered_count = 0
    for item in items:
        num_images = args.num_images if args.num_images is not None else formatter.get_num_images(item)
        final_path = formatter.get_final_path(item)
        if os.path.exists(final_path):
            skip_count += 1
            continue

        # Crash recovery: check if all tmp files exist
        existing_paths = []
        all_exist = True
        for i in range(num_images):
            save_path = formatter.get_save_path(item, i)
            if os.path.exists(save_path):
                existing_paths.append(save_path)
            else:
                all_exist = False

        if all_exist and len(existing_paths) == num_images:
            formatter.on_item_complete(item, existing_paths)
            recovered_count += 1
            continue

        for i in range(num_images):
            save_path = formatter.get_save_path(item, i)
            seed = args.seed + i
            raw_prompt = item.get(prompt_field, item["original_prompt"])
            prompt = compact_single_quote_json(raw_prompt)
            if args.max_prompt_chars > 0:
                prompt = truncate_structured_prompt(prompt, args.max_prompt_chars)
            tasks.append({
                "item": item,
                "image_idx": i,
                "save_path": save_path,
                "seed": seed,
                "prompt": prompt,
            })

    item_tasks = defaultdict(list)
    for t in tasks:
        item_tasks[t["item"]["id"]].append(t)
    items_to_process = {t["item"]["id"]: t["item"] for t in tasks}

    log.info("=" * 60)
    log.info("Inference Pipeline (OpenAI API / vllm-omni)")
    log.info("=" * 60)
    log.info("Benchmark:     %s", benchmark)
    log.info("Prompt field:  %s", prompt_field)
    if recovered_count > 0:
        log.info("Recovered:     %d items from incomplete tmp files", recovered_count)
    log.info("Items:         %d total, %d done, %d recovered, %d to process",
             len(items), skip_count, recovered_count, len(items_to_process))
    log.info("Tasks:         %d image generations", len(tasks))
    if args.max_prompt_chars > 0:
        log.info("Max prompt:    %d chars (truncation enabled)", args.max_prompt_chars)
    log.info("Resolution:    %dx%d", args.width, args.height)
    log.info("Workers:       %d", args.workers)
    log.info("Output:        %s", args.output_dir)

    if not tasks:
        log.info("All items already generated. Running finalize...")
        formatter.finalize(all_items)
        return

    # Health check
    try:
        resp = requests.get(f"{args.url}/health", timeout=5)
        resp.raise_for_status()
        log.info("Service health: OK")
    except Exception as e:
        log.error("Service health check failed: %s", e)
        log.error("Make sure vllm-omni is running at %s", args.url)
        return

    # Track per-item completion
    item_lock = threading.Lock()
    item_completed_paths = defaultdict(list)
    item_expected_count = {}
    for item_id, ts in item_tasks.items():
        item_expected_count[item_id] = len(ts)

    success_count = 0
    fail_count = 0
    pbar = tqdm(total=len(tasks), desc="Generating", unit="img", dynamic_ncols=True)

    def run_task(task):
        save_path = task["save_path"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img = send_generate_openai(
            url=args.url,
            prompt=task["prompt"],
            height=args.height,
            width=args.width,
            num_steps=args.num_steps,
            seed=task["seed"],
            cfg_scale=args.cfg_scale,
            negative_prompt=args.negative_prompt,
            timeout=args.timeout,
        )
        img.save(save_path, format="PNG")
        return task

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_task, t): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            item_id = task["item"]["id"]
            try:
                future.result()
                with item_lock:
                    item_completed_paths[item_id].append(task["save_path"])
                    if len(item_completed_paths[item_id]) == item_expected_count[item_id]:
                        formatter.on_item_complete(
                            task["item"],
                            item_completed_paths[item_id],
                        )
                success_count += 1
            except Exception as e:
                log.error("Failed id=%s img=%d: %s", item_id, task["image_idx"], e)
                fail_count += 1
            pbar.update(1)

    pbar.close()
    log.info("Generation done. %d success, %d failed.", success_count, fail_count)

    # Use all_items so finalize can discover files generated by other ranks
    formatter.finalize(all_items)
    log.info("Output: %s", args.output_dir)


if __name__ == "__main__":
    main()
