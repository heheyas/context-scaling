# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Unified detailness evaluator — runs ED and/or GPG on arbitrary
(image, caption) inputs and returns a single dict per sample.

ED  (Effective Detailness) is a caption-only, model-free metric.
    Requires three LLM API calls per (image, caption):
      1. Gemini-vision: image → OARG source tuples
      2. GPT-4o text:   caption → OARG tuples
      3. GPT-4o text:   match → YES/NO masks → F_{β=0.5}(P_A, R_A)
    Per-image source extraction is cached, so N captions on the same
    image only pay step 1 once.

GPG (Grounded Perplexity Gain) is a model-conditional metric.
    Requires loading Qwen3-VL-8B-Instruct on a CUDA device and running
    two forward passes per (image, caption): grounded (with image) and
    prior (text-only). GPG = nll_prior − nll_grounded per token.

You can run either, both, or any subset:

    >>> from detailness.evaluator import Evaluator
    >>> ev = Evaluator(metrics=("ed",))                    # ED only, no GPU needed
    >>> ev.score("/path/to/img.png",
    ...          "A silver pickup truck on a street.")
    {'ed': 0.278, 'ed_P_A': 1.0, 'ed_R_A': 0.071, 'ed_F05_A': 0.278,
     'ed_n_src_A': 70, 'ed_n_cap_A': 2}

    >>> ev = Evaluator(metrics=("gpg",),
    ...                gpg_model_path="/path/to/Qwen3-VL-8B-Instruct")
    >>> ev.score(img_path, caption)
    {'gpg': 0.382, 'gpg_total_nats': 289.4, 'gpg_n_tokens': 756, ...}

CLI:
    HOME=<RAY_SERVE_HOME> PYTHONNOUSERSITE=1 \\
    python -m detailness.evaluator \\
        --jsonl /path/to/dataset.jsonl \\
        --metrics ed,gpg \\
        --gpg-model <HDFS_ROOT>/.../Qwen3-VL-8B-Instruct \\
        --out /tmp/evaluator_out.jsonl

The input JSONL is one record per line with at least:
    {"image": "<path>", "caption": "<text>", "uid": "<optional>"}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from tqdm import tqdm

log = logging.getLogger(__name__)


# ============================================================================ #
# ED back-end (LLM-API, caption-only, lazy-loaded so no GPU needed for ED-only)
# ============================================================================ #


class _EDBackend:
    """Wraps the ED pipeline: image-source extraction + caption decomposition
    + matching + per-row F0.5(P_A, R_A).

    Image-source extraction caches by uid (or image-path hash if no uid).
    """

    def __init__(self,
                 matcher: str = "gpt",
                 model_image_source: Optional[str] = None,
                 model_caption: Optional[str] = None,
                 model_match: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 api_config_json: Optional[str] = None,
                 gemini_key_file: Optional[str] = None):
        from detailness.judge import (
            DEFAULT_MODEL, KeyPool, load_api_keys, load_api_keys_json,
            load_prompt,
        )
        from .ed.extract_image import extract_image_source

        self._extract_image_source = extract_image_source
        self._KeyPool             = KeyPool

        api_config_json = api_config_json or os.environ.get("GEMINI_API_CONFIG", "")

        prompts_dir = os.path.join(os.path.dirname(__file__), "ed", "prompts")
        self._prompt_image_source, _ = load_prompt(
            os.path.join(prompts_dir, "image_source.txt"))
        self._prompt_caption,      _ = load_prompt(
            os.path.join(prompts_dir, "extract_tuples.txt"))
        self._prompt_match,        _ = load_prompt(
            os.path.join(prompts_dir, "match_tuples.txt"))

        # Vision pool (Gemini) — always used for image-source extraction
        self._gem_pool = KeyPool(load_api_keys(gemini_key_file))
        self._model_image_source = model_image_source or DEFAULT_MODEL

        # Text pools (caption decompose + match) — use chosen matcher family
        if matcher == "gemini":
            self._txt_pool = self._gem_pool
            self._model_caption = model_caption or DEFAULT_MODEL
            self._model_match   = model_match   or DEFAULT_MODEL
        else:
            self._txt_pool = KeyPool(load_api_keys_json(api_config_json))
            self._model_caption = model_caption or "gpt-4o-2024-11-20"
            self._model_match   = model_match   or "gpt-4o-2024-11-20"

        # Persistent caches (optional)
        from detailness.cache import JsonlCache
        self._JsonlCache = JsonlCache
        self._cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._src_cache = JsonlCache(
                os.path.join(cache_dir, "ed_image_source.jsonl"))
            self._cap_cache = JsonlCache(
                os.path.join(cache_dir, "ed_caption_tuples.jsonl"))
        else:
            self._src_cache = None
            self._cap_cache = None

        # In-memory per-image source cache for batch mode (keyed by uid).
        self._mem_src: Dict[str, dict] = {}

    # ----------------------- ED primitives ------------------------------ #

    def _get_image_source(self, image_path: str, uid: str) -> dict:
        if uid in self._mem_src:
            return self._mem_src[uid]
        if self._src_cache is not None and self._src_cache.has(uid):
            v = self._src_cache.get(uid)
            self._mem_src[uid] = v
            return v
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        src = self._extract_image_source(
            self._gem_pool, self._prompt_image_source,
            img_bytes, model=self._model_image_source,
        )
        if src is None:
            src = {"O": [], "A": [], "R": [], "G": []}
        if self._src_cache is not None:
            self._src_cache.put(uid, src)
        self._mem_src[uid] = src
        return src

    def _get_caption_tuples(self, cap_key: str, caption: str) -> dict:
        if self._cap_cache is not None and self._cap_cache.has(cap_key):
            return self._cap_cache.get(cap_key)
        from detailness.judge import extract_tuples
        out = extract_tuples(
            self._txt_pool, self._prompt_caption,
            caption, model=self._model_caption,
        )
        if out is None:
            out = {"O": [], "A": [], "R": [], "G": []}
        if self._cap_cache is not None:
            self._cap_cache.put(cap_key, out)
        return out

    def _match(self, src: dict, cap: dict) -> dict:
        from detailness.judge import match_tuples
        masks = match_tuples(
            self._txt_pool, self._prompt_match,
            src, cap, model=self._model_match,
        )
        if masks is None:
            masks = {
                "source_recall":     {c: ["NO"] * len(src.get(c, [])) for c in "OARG"},
                "caption_precision": {c: ["NO"] * len(cap.get(c, [])) for c in "OARG"},
            }
        return masks

    # ----------------------- Public single-pair API --------------------- #

    def score_one(self, image_path: str, caption: str,
                  uid: Optional[str] = None) -> Dict:
        """Compute ED for a single (image, caption) pair."""
        from .ed.io import cap_key as _cap_key, f_beta

        if uid is None:
            # Stable, deterministic uid from the image path
            import hashlib
            h = hashlib.sha1(image_path.encode()).hexdigest()
            uid = f"img_{h[:16]}"

        src = self._get_image_source(image_path, uid)
        k = _cap_key(uid, caption)
        cap = self._get_caption_tuples(k, caption)
        masks = self._match(src, cap)

        sr_A = masks["source_recall"].get("A", [])
        cp_A = masks["caption_precision"].get("A", [])
        n_src = len(src.get("A", []))
        n_cap = len(cap.get("A", []))
        R = sum(1 for m in sr_A if m == "YES") / n_src if n_src else 0.0
        P = sum(1 for m in cp_A if m == "YES") / n_cap if n_cap else 0.0
        F = f_beta(P, R, beta=0.5)
        return {
            "ed":         F,
            "ed_F05_A":   F,
            "ed_P_A":     P,
            "ed_R_A":     R,
            "ed_n_src_A": n_src,
            "ed_n_cap_A": n_cap,
        }

    def close(self):
        if self._src_cache is not None: self._src_cache.close()
        if self._cap_cache is not None: self._cap_cache.close()


# ============================================================================ #
# GPG back-end (Qwen3-VL forward pass, GPU)
# ============================================================================ #


class _GPGBackend:
    """Wraps Qwen3-VL-8B per-(image, caption) PMI scoring."""

    def __init__(self,
                 model_path: str,
                 device: str = "cuda:0",
                 max_pixels: int = 1024 * 28 * 28,
                 content_only: bool = True,
                 canonicalize_json: bool = True,
                 drop_keys: Optional[Sequence[str]] = None,
                 system_prompt: Optional[str] = None,
                 convert_bboxes: bool = False):
        if not model_path:
            raise ValueError("GPG requires --gpg-model / model_path")
        from .gpg.score import (
            CANONICAL_DROP_KEYS,
            canonicalize_json_caption,
            convert_bboxes_to_qwen_native,
            load_model,
            score as _gpg_score_fn,
        )
        if drop_keys is None:
            drop_keys = list(CANONICAL_DROP_KEYS)
        self._drop_keys             = list(drop_keys)
        self._canonicalize_json     = canonicalize_json
        self._convert_bboxes        = convert_bboxes
        self._content_only          = content_only
        self._max_pixels            = max_pixels
        self._system_prompt         = system_prompt
        self._canonicalize_json_fn  = canonicalize_json_caption
        self._convert_bboxes_fn     = convert_bboxes_to_qwen_native
        self._gpg_score_fn          = _gpg_score_fn

        log.info("GPG: loading model %s on %s", model_path, device)
        self._model, self._processor = load_model(model_path, device=device)
        if hasattr(self._processor, "image_processor"):
            # min_pixels is fixed at the GPG paper default; max_pixels is the
            # only knob exposed to the Evaluator.
            self._processor.image_processor.max_pixels = max_pixels

    def _content_mask_for(self, caption: str) -> Optional[list]:
        if not self._content_only:
            return None
        from .gpg.content_mask import token_content_mask
        return token_content_mask(caption, self._processor.tokenizer)

    def score_one(self, image_path: str, caption: str,
                  uid: Optional[str] = None) -> Dict:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        canon = caption
        if self._canonicalize_json:
            canon = self._canonicalize_json_fn(canon, drop_keys=self._drop_keys)
        if self._convert_bboxes:
            canon = self._convert_bboxes_fn(canon, image.width, image.height,
                                            self._max_pixels)
        mask = self._content_mask_for(canon)
        s_g, n_g, m_g = self._gpg_score_fn(
            self._model, self._processor, canon, image,
            system_prompt=self._system_prompt, content_mask=mask,
        )
        s_p, n_p, m_p = self._gpg_score_fn(
            self._model, self._processor, canon, image=None,
            system_prompt=self._system_prompt, content_mask=mask,
        )
        gpg_per_tok = m_p - m_g           # higher = more grounded
        return {
            "gpg":               gpg_per_tok,
            "gpg_total_nats":    (s_p - s_g),
            "gpg_n_tokens":      n_g,
            "gpg_nll_grounded":  m_g,
            "gpg_nll_prior":     m_p,
        }


# ============================================================================ #
# Public Evaluator
# ============================================================================ #


class Evaluator:
    """One object, one or both metrics, batch- or single-pair friendly.

    Args:
        metrics: subset of ("ed", "gpg"). Both runs in parallel for batch
                 mode (ED uses LLM-API threadpool, GPG uses GPU stream).
        ed_*:    per-call config for the ED back-end.
        gpg_*:   per-call config for the GPG back-end.
    """

    def __init__(self,
                 metrics: Sequence[str] = ("ed", "gpg"),
                 *,
                 # ED config
                 ed_matcher: str = "gpt",
                 ed_cache_dir: Optional[str] = None,
                 ed_api_config_json: Optional[str] = None,
                 ed_gemini_key_file: Optional[str] = None,
                 # GPG config
                 gpg_model_path: Optional[str] = None,
                 gpg_device: str = "cuda:0",
                 gpg_max_pixels: int = 1024 * 28 * 28,
                 gpg_content_only: bool = True,
                 gpg_canonicalize_json: bool = True,
                 gpg_drop_keys: Optional[Sequence[str]] = None,
                 gpg_convert_bboxes: bool = False,
                 gpg_system_prompt: Optional[str] = None):

        self.metrics = tuple(metrics)
        unknown = set(self.metrics) - {"ed", "gpg"}
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}; choices: ed, gpg")

        self._ed:  Optional[_EDBackend]  = None
        self._gpg: Optional[_GPGBackend] = None
        if "ed" in self.metrics:
            self._ed = _EDBackend(
                matcher=ed_matcher,
                cache_dir=ed_cache_dir,
                api_config_json=ed_api_config_json,
                gemini_key_file=ed_gemini_key_file,
            )
        if "gpg" in self.metrics:
            self._gpg = _GPGBackend(
                model_path=gpg_model_path,
                device=gpg_device,
                max_pixels=gpg_max_pixels,
                content_only=gpg_content_only,
                canonicalize_json=gpg_canonicalize_json,
                drop_keys=gpg_drop_keys,
                convert_bboxes=gpg_convert_bboxes,
                system_prompt=gpg_system_prompt,
            )

    # ------------------ single-pair --------------------------------- #

    def score(self, image_path: str, caption: str,
              uid: Optional[str] = None) -> Dict:
        """Return a flat dict with all metric values + diagnostics."""
        out: Dict = {}
        if self._ed:
            out.update(self._ed.score_one(image_path, caption, uid=uid))
        if self._gpg:
            out.update(self._gpg.score_one(image_path, caption, uid=uid))
        return out

    # ------------------ batch / dataset ----------------------------- #

    def score_dataset(self,
                      records: Iterable[Dict],
                      *,
                      ed_workers: int = 32) -> Iterator[Dict]:
        """Iterate records, yield each enriched with metric outputs.

        Each input record must have keys "image" (path), "caption" (text);
        "uid" is optional (auto-derived from image path if missing).

        ED is run with a threadpool over `ed_workers` workers (LLM API).
        GPG is run serially on the current GPU stream (model fwd-pass is
        already its own bottleneck).
        """
        records = list(records)

        # ED pass — threaded
        if self._ed:
            def _ed_one(rec):
                try:
                    s = self._ed.score_one(rec["image"], rec["caption"],
                                           uid=rec.get("uid"))
                    rec.update(s)
                except Exception as e:
                    log.warning("ED failed for %s: %s",
                                rec.get("uid") or rec.get("image"), e)
                return rec
            with ThreadPoolExecutor(max_workers=ed_workers) as ex:
                futs = {ex.submit(_ed_one, r): r for r in records}
                pbar = tqdm(total=len(records), desc="ed", unit="rec")
                for fut in as_completed(futs):
                    fut.result()
                    pbar.update(1)
                pbar.close()

        # GPG pass — serial on GPU
        if self._gpg:
            for rec in tqdm(records, desc="gpg", unit="rec"):
                try:
                    s = self._gpg.score_one(rec["image"], rec["caption"],
                                            uid=rec.get("uid"))
                    rec.update(s)
                except Exception as e:
                    log.warning("GPG failed for %s: %s",
                                rec.get("uid") or rec.get("image"), e)

        for rec in records:
            yield rec

    def close(self):
        if self._ed:
            self._ed.close()


# ============================================================================ #
# CLI
# ============================================================================ #


def _iter_jsonl(path: str) -> Iterator[Dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(
        description="Unified detailness evaluator (ED + GPG)")
    ap.add_argument("--jsonl", required=True,
                    help='Input JSONL with {"image", "caption"[, "uid"]} per line')
    ap.add_argument("--out", required=True,
                    help="Output JSONL with metric outputs added")
    ap.add_argument("--metrics", default="ed",
                    help="Comma-separated subset of {ed, gpg}; default: ed")
    # ED
    ap.add_argument("--ed-matcher", choices=("gpt", "gemini"), default="gpt")
    ap.add_argument("--ed-cache-dir", default=None,
                    help="Persistent ED cache dir "
                         "(reuses image-source / caption-tuple results)")
    ap.add_argument("--ed-api-config-json", default=None)
    ap.add_argument("--ed-gemini-key-file", default=None)
    ap.add_argument("--ed-workers", type=int, default=32)
    # GPG
    ap.add_argument("--gpg-model", default=None,
                    help="Required when 'gpg' in --metrics")
    ap.add_argument("--gpg-device", default="cuda:0")
    ap.add_argument("--gpg-max-pixels", type=int, default=1024 * 28 * 28)
    ap.add_argument("--gpg-content-only", action="store_true", default=True)
    ap.add_argument("--gpg-no-content-only", dest="gpg_content_only",
                    action="store_false")
    ap.add_argument("--gpg-canonicalize-json", action="store_true", default=True)
    ap.add_argument("--gpg-no-canonicalize-json",
                    dest="gpg_canonicalize_json", action="store_false")
    ap.add_argument("--gpg-drop-keys", default=None,
                    help='Comma-separated extra JSON keys to drop, e.g. '
                         '"atmosphere,lighting,style,photography"')
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    drop_keys = None
    if args.gpg_drop_keys:
        drop_keys = [k.strip() for k in args.gpg_drop_keys.split(",")
                     if k.strip()]

    ev = Evaluator(
        metrics=metrics,
        ed_matcher=args.ed_matcher,
        ed_cache_dir=args.ed_cache_dir,
        ed_api_config_json=args.ed_api_config_json,
        ed_gemini_key_file=args.ed_gemini_key_file,
        gpg_model_path=args.gpg_model,
        gpg_device=args.gpg_device,
        gpg_max_pixels=args.gpg_max_pixels,
        gpg_content_only=args.gpg_content_only,
        gpg_canonicalize_json=args.gpg_canonicalize_json,
        gpg_drop_keys=drop_keys,
    )

    n_written = 0
    with open(args.out, "w") as fout:
        for rec in ev.score_dataset(_iter_jsonl(args.jsonl),
                                    ed_workers=args.ed_workers):
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
    ev.close()
    log.info("Wrote %d records → %s", n_written, args.out)


if __name__ == "__main__":
    main()
