# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Caption text → OARG tuples via GPT-4o (closed 16-attr schema).

For each unique caption in the 16-cell pool, call GPT-4o with
`prompts/extract_tuples.txt` and parse into canonical {O, A, R, G}.
Captions are deduplicated by cap_key(uid, caption) since some cells
share identical caption strings (e.g. abl_json_full = struct/l10).

Usage:
    HOME=<RAY_SERVE_HOME> python -m detailness.ed.extract_caption \\
        --pool /tmp/detailness_n100_16cell/pool.jsonl \\
        --cache /tmp/detailness_real_n100/cache/v2_caption_tuples.jsonl \\
        --workers 512
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
    extract_tuples,
    load_api_keys,
    load_api_keys_json,
    load_prompt,
)

from .io import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CAPTION_TUPLES_CACHE,
    DEFAULT_POOL_PATH,
    cap_key,
)

log = logging.getLogger(__name__)

_PROMPT_PATH = os.path.join(os.path.dirname(__file__),
                            "prompts", "extract_tuples.txt")


def main():
    ap = argparse.ArgumentParser(
        description="Extract caption-side OARG tuples (GPT-4o text)")
    ap.add_argument("--pool", default=DEFAULT_POOL_PATH,
                    help="Path to the 16-cell pool JSONL")
    ap.add_argument("--cache", default=os.path.join(
        DEFAULT_CACHE_DIR, DEFAULT_CAPTION_TUPLES_CACHE),
                    help="Output JSONL cache (keyed by cap_key)")
    ap.add_argument("--prompt", default=_PROMPT_PATH,
                    help="Path to extract_tuples.txt prompt")
    ap.add_argument("--workers", type=int, default=512)

    # Matcher selection
    ap.add_argument("--use-gpt4o", action="store_true", default=True,
                    help="Use GPT-4o (Azure-compatible endpoint); default ON")
    ap.add_argument("--gemini", dest="use_gpt4o", action="store_false",
                    help="Use Gemini instead of GPT-4o")
    ap.add_argument("--api-config-json",
                    default=os.environ.get("GEMINI_API_CONFIG", ""),
                    help="GPT-4o keys JSON config")
    ap.add_argument("--key-file", default=None,
                    help="Gemini key file (default: judge.DEFAULT_KEY_FILE)")
    ap.add_argument("--model", default=None,
                    help="Model name override (default: gpt-4o-2024-11-20 "
                         "for GPT path, judge.DEFAULT_MODEL for Gemini)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.use_gpt4o:
        keys = load_api_keys_json(args.api_config_json)
        model = args.model or "gpt-4o-2024-11-20"
    else:
        keys = load_api_keys(args.key_file)
        model = args.model or DEFAULT_MODEL
    pool = KeyPool(keys)
    cache = JsonlCache(args.cache)
    prompt_body, _ = load_prompt(args.prompt)

    # Deduplicate captions by cap_key — abl_json_full and structured/l10 share
    # captions verbatim.
    rows = []
    with open(args.pool) as f:
        for line in f:
            rows.append(json.loads(line))

    seen = set()
    todo = []
    for r in rows:
        k = cap_key(r["uid"], r["caption"])
        if k in seen:
            continue
        seen.add(k)
        if cache.has(k):
            continue
        todo.append((k, r["caption"]))
    log.info("Caption extract: %d to call (already cached %d, pool size %d)",
             len(todo), len(seen) - len(todo), len(rows))

    def _work(t):
        k, caption = t
        out = extract_tuples(pool, prompt_body, caption, model=model)
        if out is None:
            out = {"O": [], "A": [], "R": [], "G": []}
        cache.put(k, out)
        return k

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_work, t): t[0] for t in todo}
        pbar = tqdm(total=len(todo), desc="cap_extract", unit="cap")
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                log.warning("extract failed: %s", e)
            pbar.update(1)
        pbar.close()
    cache.close()
    log.info("Done. Cache: %s. Pool: %s", args.cache, pool.stats())


if __name__ == "__main__":
    main()
