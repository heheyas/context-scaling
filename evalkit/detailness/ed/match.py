# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Per-(uid, caption) tuple match: (source_tuples, caption_tuples) → YES/NO
masks for source_recall and caption_precision (OARG categories).

Single matcher pass — the heart of the paper-main v4.1 form. The same
script handles the GPT-4o (paper main) and Gemini-3-pro (cross-matcher
robustness) configurations via `--matcher gemini`.

Usage:
    # GPT-4o, paper-main cache
    HOME=<RAY_SERVE_HOME> python -m detailness.ed.match \\
        --workers 1024 \\
        --out /tmp/detailness_real_n100/cache/v4_image_match.jsonl

    # Gemini-3-pro, cross-matcher robustness cache
    HOME=<RAY_SERVE_HOME> python -m detailness.ed.match \\
        --matcher gemini --workers 512 \\
        --out /tmp/detailness_real_n100/cache/v24_match_gemini.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from detailness.cache import JsonlCache
from detailness.judge import (
    DEFAULT_MODEL,
    KeyPool,
    load_api_keys,
    load_api_keys_json,
    load_prompt,
    match_tuples,
)

from .io import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CAPTION_TUPLES_CACHE,
    DEFAULT_IMAGE_SOURCE_CACHE,
    DEFAULT_POOL_PATH,
    MATCH_CACHE_FILES,
    cap_key,
)

log = logging.getLogger(__name__)

_PROMPT_PATH = os.path.join(os.path.dirname(__file__),
                            "prompts", "match_tuples.txt")


def main():
    ap = argparse.ArgumentParser(
        description="Match per-(uid, caption) source ↔ caption OARG tuples")

    # Matcher selection
    ap.add_argument("--matcher", choices=("gpt", "gemini"), default="gpt",
                    help="LLM matcher. 'gpt' (paper main) uses "
                         "gpt-4o-2024-11-20 via the Azure-compatible "
                         "endpoint; 'gemini' uses gemini-3-pro-preview-new "
                         "(cross-matcher robustness)")
    ap.add_argument("--model", default=None,
                    help="Override model name (default: per --matcher)")
    ap.add_argument("--api-config-json",
                    default=os.environ.get("GEMINI_API_CONFIG", ""),
                    help="GPT-4o keys JSON config")
    ap.add_argument("--key-file", default=None,
                    help="Gemini key file (default: judge.DEFAULT_KEY_FILE)")

    # IO
    ap.add_argument("--pool", default=DEFAULT_POOL_PATH,
                    help="16-cell evaluation pool JSONL")
    ap.add_argument("--src-cache", default=os.path.join(
        DEFAULT_CACHE_DIR, DEFAULT_IMAGE_SOURCE_CACHE),
                    help="Image-source OARG cache (from extract_image.py)")
    ap.add_argument("--cap-cache", default=os.path.join(
        DEFAULT_CACHE_DIR, DEFAULT_CAPTION_TUPLES_CACHE),
                    help="Caption OARG cache (from extract_caption.py)")
    ap.add_argument("--out", default=None,
                    help="Output match cache JSONL "
                         "(default: v4_image_match.jsonl for gpt, "
                         "v24_match_gemini.jsonl for gemini)")
    ap.add_argument("--prompt", default=_PROMPT_PATH,
                    help="Path to match_tuples.txt prompt")
    ap.add_argument("--workers", type=int, default=1024)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.matcher == "gpt":
        keys = load_api_keys_json(args.api_config_json)
        model = args.model or "gpt-4o-2024-11-20"
        default_out = MATCH_CACHE_FILES["v41"]
    else:
        keys = load_api_keys(args.key_file)
        model = args.model or DEFAULT_MODEL
        default_out = MATCH_CACHE_FILES["v24"]
    out_path = args.out or os.path.join(DEFAULT_CACHE_DIR, default_out)
    log.info("ED match: matcher=%s model=%s workers=%d out=%s",
             args.matcher, model, args.workers, out_path)

    pool        = KeyPool(keys)
    src_cache   = JsonlCache(args.src_cache)
    cap_cache   = JsonlCache(args.cap_cache)
    match_cache = JsonlCache(out_path)
    prompt_body, _ = load_prompt(args.prompt)

    with open(args.pool) as f:
        rows = [json.loads(line) for line in f]

    tasks = []
    cached = missing_src = missing_cap = 0
    for r in rows:
        k = cap_key(r["uid"], r["caption"])
        if match_cache.has(k):
            cached += 1
            continue
        src = src_cache.get(r["uid"])
        cap = cap_cache.get(k)
        if src is None:
            missing_src += 1
            continue
        if cap is None:
            missing_cap += 1
            continue
        tasks.append((k, r["uid"], r["kind"], r["level"], src, cap))
    log.info("ED match: %d to call (cached %d, missing_src %d, missing_cap %d)",
             len(tasks), cached, missing_src, missing_cap)

    def _work(t):
        k, uid, kind, level, src, cap = t
        masks = match_tuples(pool, prompt_body, src, cap, model=model)
        if masks is None:
            masks = {
                "source_recall":     {c: ["NO"] * len(src.get(c, [])) for c in "OARG"},
                "caption_precision": {c: ["NO"] * len(cap.get(c, [])) for c in "OARG"},
            }
        match_cache.put(k, {
            "uid": uid, "kind": kind, "level": level,
            "source": src, "caption": cap,
            "source_recall":     masks["source_recall"],
            "caption_precision": masks["caption_precision"],
        })
        return k

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_work, t): t[0] for t in tasks}
        pbar = tqdm(total=len(tasks), desc=f"match_{args.matcher}",
                    unit="task")
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                log.warning("match failed: %s", e)
            pbar.update(1)
        pbar.close()
    match_cache.close()
    log.info("Done. Cache: %s. Pool: %s", out_path, pool.stats())


if __name__ == "__main__":
    main()
