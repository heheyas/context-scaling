# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Image → OARG tuples via Gemini-vision (exhaustive enumeration).

For each uid (image file `<images_dir>/<uid>.bin`), call Gemini-3-pro
vision with `prompts/image_source.txt`, parse the JSON response into a
canonical {O, A, R, G} dict, and append to a JSONL cache.

This produces the "image-grounded source" side of ED — independent of any
caption (the prompt and image never see a caption being scored).

Threadpool 32-64 workers, ~5-10 min wall clock for 300 uids.

Usage:
    HOME=<RAY_SERVE_HOME> python -m detailness.ed.extract_image \\
        --images-dir /tmp/detailness_real_n100/images \\
        --cache /tmp/detailness_real_n100/cache/v4_image_source.jsonl \\
        --workers 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from detailness.cache import JsonlCache
from detailness.judge import (
    DEFAULT_MODEL,
    KeyPool,
    _chat,
    _strip_fences,
    image_bytes_to_data_url,
    load_api_keys,
    sniff_mime,
)

from .io import DEFAULT_CACHE_DIR, DEFAULT_IMAGE_SOURCE_CACHE, DEFAULT_IMAGES_DIR

log = logging.getLogger(__name__)

_PROMPT_PATH = os.path.join(os.path.dirname(__file__),
                            "prompts", "image_source.txt")


def parse_json_object(text):
    if text is None:
        return None
    text = _strip_fences(text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def normalize_oarg(obj):
    """Coerce parsed object into canonical {O, A, R, G} of list[dict]."""
    out = {"O": [], "A": [], "R": [], "G": []}
    if not isinstance(obj, dict):
        return out
    for k in "OARG":
        v = obj.get(k, [])
        if isinstance(v, list):
            out[k] = [x for x in v if isinstance(x, dict)]
    return out


def extract_image_source(pool: KeyPool, prompt_body: str,
                         image_bytes: bytes,
                         model: str = DEFAULT_MODEL):
    mime = sniff_mime(image_bytes)
    data_url = image_bytes_to_data_url(image_bytes, mime=mime)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt_body},
        ],
    }]
    resp = _chat(pool, messages, model=model, max_tokens=16384,
                 temperature=0.0,
                 response_format={"type": "json_object"})
    if resp is None:
        return None
    parsed = parse_json_object(resp)
    if parsed is None:
        log.warning("parse failed for response (first 300 chars): %s",
                    resp[:300])
        return None
    return normalize_oarg(parsed)


def main():
    ap = argparse.ArgumentParser(
        description="Extract image-grounded OARG tuples (Gemini-vision)")
    ap.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR,
                    help="Directory of <uid>.bin image files")
    ap.add_argument("--prompt", default=_PROMPT_PATH,
                    help="Path to image_source.txt prompt")
    ap.add_argument("--cache", default=os.path.join(
        DEFAULT_CACHE_DIR, DEFAULT_IMAGE_SOURCE_CACHE),
                    help="Output JSONL cache")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--key-file", default=None,
                    help="Optional Gemini key file path "
                         "(default: judge.DEFAULT_KEY_FILE)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    prompt_body = open(args.prompt).read()
    pool = KeyPool(load_api_keys(args.key_file))
    cache = JsonlCache(args.cache)

    uids = sorted(f[:-4] for f in os.listdir(args.images_dir)
                  if f.endswith(".bin"))
    todo = [u for u in uids if not cache.has(u)]
    log.info("Image-grounded source: %d uids (cached %d, todo %d)",
             len(uids), len(uids) - len(todo), len(todo))

    def _work(uid):
        path = os.path.join(args.images_dir, f"{uid}.bin")
        with open(path, "rb") as f:
            img = f.read()
        out = extract_image_source(pool, prompt_body, img, model=args.model)
        if out is None:
            log.warning("uid %s: extraction failed (None)", uid)
            return uid, False
        cache.put(uid, out)
        return uid, True

    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_work, u): u for u in todo}
        pbar = tqdm(total=len(todo), desc="img_src", unit="uid")
        for fut in as_completed(futs):
            try:
                uid, success = fut.result()
                if success:
                    ok += 1
                else:
                    fail += 1
            except Exception as e:
                log.warning("future failed: %s", e)
                fail += 1
            pbar.update(1)
            if pbar.n % 50 == 0:
                pbar.set_postfix_str(f"ok={ok} fail={fail}")
        pbar.close()
    cache.close()
    log.info("Done: ok=%d fail=%d  cache=%s  pool stats: %s",
             ok, fail, args.cache, pool.stats())


if __name__ == "__main__":
    main()
