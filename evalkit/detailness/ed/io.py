# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Shared I/O for ED analysis: load per-row match records, compute
per-cell trim20 of F_{β=0.5}(P_A, R_A), join with Bagel MSE at the unified
token budget.

The 16-cell label/category convention and the Bagel MSE-side loaders are
shared with GPG — re-exported from detailness.gpg.io to
avoid duplication. ED-specific logic lives here.

Path defaults match the v4.1 paper-final run. Override via function args
or argparse flags when re-running on a different machine.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Reuse the GPG-side conventions (NO duplication of MSE loading or cell
# enumeration). These are the same 16 cells GPG validates against.
from ..gpg.io import (
    ABL_FILES,
    DEFAULT_WANDB_DIRS,
    DENSE_LEVELS,
    SPATIAL_FILES,
    STRUCT_LEVELS,
    load_all_loss_runs,
    mean_se,
    pearson,
    smoothed_mse,
    unified_budget,
)


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR     = "/tmp/detailness_real_n100/cache"
DEFAULT_AGGREGATE_DIR = "/tmp/detailness_n100_16cell"
DEFAULT_POOL_PATH     = "/tmp/detailness_n100_16cell/pool.jsonl"
DEFAULT_IMAGES_DIR    = "/tmp/detailness_real_n100/images"

# Match-cache versions:
#   v41    — paper main result (single GPT-4o matcher, F_{0.5}[A] trim20)
#   v24    — cross-matcher robustness (Gemini-3-pro matcher, same formula)
#   v25    — ensemble alternative (loads both v41 + v24)
DEFAULT_VERSION = "v41"

# Filenames within DEFAULT_CACHE_DIR. v41 reuses the v4 GPT-4o match cache;
# the historical naming reflects the iteration history rather than the
# final version label.
MATCH_CACHE_FILES = {
    "v41": "v4_image_match.jsonl",        # GPT-4o, paper main
    "v24": "v24_match_gemini.jsonl",      # Gemini-3-pro
}

DEFAULT_IMAGE_SOURCE_CACHE   = "v4_image_source.jsonl"
DEFAULT_CAPTION_TUPLES_CACHE = "v2_caption_tuples.jsonl"


# ---------------------------------------------------------------------------
# Hashing helper: pool rows are keyed by sha1(uid + \0 + caption_text)
# ---------------------------------------------------------------------------

def cap_key(uid: str, caption: str) -> str:
    """Canonical (uid, caption) → cache-key hash.

    Match caches and caption-tuples caches share this key. Must match the
    hashing used by detailness.ed.match (and historically by
    /tmp/v3_match_image.py).
    """
    h = hashlib.sha1()
    h.update(uid.encode())
    h.update(b"\0")
    h.update(caption.encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Per-row F-score helpers
# ---------------------------------------------------------------------------

def f_beta(P: float, R: float, beta: float = 0.5) -> float:
    """F_β = (1+β²) · P · R / (β² · P + R). Returns 0 when P = R = 0."""
    if P == 0 and R == 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * P + R
    return (1.0 + b2) * P * R / denom if denom else 0.0


def trim20(vs: List[float]) -> float:
    """Hampel 1974 20%-trimmed mean. Drop top 20% + bottom 20%, average
    the central 60%. Returns nan on empty input."""
    if not vs:
        return float("nan")
    vs = sorted(vs)
    t = max(1, len(vs) // 5)
    if len(vs) > 2 * t:
        vs = vs[t:len(vs) - t]
    return sum(vs) / len(vs)


def median(vs: List[float]) -> float:
    if not vs:
        return float("nan")
    vs = sorted(vs)
    return vs[len(vs) // 2]


# ---------------------------------------------------------------------------
# JSONL cache loaders
# ---------------------------------------------------------------------------

def _load_jsonl_cache(path: str) -> Dict[str, dict]:
    """Load a {"key": ..., "value": ...} JSONL cache → {key: value} dict."""
    out: Dict[str, dict] = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            k = r.get("key")
            if k is not None and "value" in r:
                out[k] = r["value"]
    return out


def load_image_source(cache_dir: str = DEFAULT_CACHE_DIR,
                      filename: str = DEFAULT_IMAGE_SOURCE_CACHE
                      ) -> Dict[str, dict]:
    """uid → OARG dict (image-grounded source tuples from Gemini-vision)."""
    return _load_jsonl_cache(os.path.join(cache_dir, filename))


def load_caption_tuples(cache_dir: str = DEFAULT_CACHE_DIR,
                        filename: str = DEFAULT_CAPTION_TUPLES_CACHE
                        ) -> Dict[str, dict]:
    """cap_key → OARG dict (caption-side tuples from GPT-4o text extractor)."""
    return _load_jsonl_cache(os.path.join(cache_dir, filename))


def load_match(version: str = DEFAULT_VERSION,
               cache_dir: str = DEFAULT_CACHE_DIR) -> Dict[str, dict]:
    """cap_key → match record. Match record has keys:
       {uid, kind, level, source, caption, source_recall, caption_precision}.
    """
    if version not in MATCH_CACHE_FILES:
        raise ValueError(f"Unknown ED version {version!r}; "
                         f"choices: {sorted(MATCH_CACHE_FILES)}")
    return _load_jsonl_cache(os.path.join(cache_dir,
                                          MATCH_CACHE_FILES[version]))


def load_pool(path: str = DEFAULT_POOL_PATH) -> List[dict]:
    """The 16-cell × 300-uid evaluation pool. Each row has
       {uid, caption, kind, level, ...}."""
    out: List[dict] = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Per-cell aggregation: pool rows + match cache → per-(kind, level)
# list of F_{β=0.5}(P_A, R_A) values
# ---------------------------------------------------------------------------

def per_row_F(match_record: dict, category: str = "A", beta: float = 0.5
              ) -> float:
    """Compute F_β(P_cat, R_cat) for a single match record on one category.

    P_cat = (count of YES caption-cat tuples) / (total caption-cat tuples)
    R_cat = (count of YES source-cat tuples) / (total source-cat tuples)
    """
    src = match_record["source"].get(category, [])
    sr  = match_record["source_recall"].get(category, [])
    cap = match_record["caption"].get(category, [])
    cp  = match_record["caption_precision"].get(category, [])
    R = sum(1 for m in sr if m == "YES") / len(src) if src else 0.0
    P = sum(1 for m in cp if m == "YES") / len(cap) if cap else 0.0
    return f_beta(P, R, beta=beta)


def aggregate_per_cell(pool: List[dict],
                       match_cache: Dict[str, dict],
                       category: str = "A",
                       beta: float = 0.5
                       ) -> Dict[Tuple[str, str], List[float]]:
    """Group per-row F values by (kind, level).

    Returns {(kind, level): [F_uid1, F_uid2, ...]}. Drops rows whose cap_key
    isn't in `match_cache` (typically caused by source-extraction timeouts).
    """
    by_cell: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in pool:
        k = cap_key(r["uid"], r["caption"])
        v = match_cache.get(k)
        if v is None:
            continue
        by_cell[(r["kind"], r["level"])].append(per_row_F(v, category, beta))
    return dict(by_cell)


# ---------------------------------------------------------------------------
# Joined 16-point dataset (mirrors GPG's build_16_points)
# ---------------------------------------------------------------------------

# Schema: (ed_value, ed_se, mse, label, category)
Point = Tuple[float, float, float, str, str]


def _bootstrap_se(vs: List[float], B: int = 1000) -> float:
    """Bootstrap SE of trim20 over `vs`. Cheap (B=1000 by default)."""
    if len(vs) < 4:
        return 0.0
    import random
    rng = random.Random(0)
    n = len(vs)
    means = []
    for _ in range(B):
        sample = [vs[rng.randrange(n)] for _ in range(n)]
        means.append(trim20(sample))
    m = sum(means) / B
    var = sum((x - m) ** 2 for x in means) / (B - 1)
    return math.sqrt(var)


def build_16_points(by_cell: Dict[Tuple[str, str], List[float]],
                    mse_at: Dict[Tuple[str, str], float],
                    excluded_abl: Optional[set] = None,
                    drop_l5: bool = False,
                    aggregator=trim20,
                    compute_se: bool = True,
                    se_bootstrap_B: int = 1000) -> List[Point]:
    """Join per-cell ED values with Bagel MSE at unified budget.

    Mirrors gpg.io.build_16_points. Drops `abl_no_depth` and
    `abl_no_atmosphere_lighting` by default (rerun pending). Keeps
    struct/l5 by default (it's a Bagel-side undertraining outlier but
    the metric ranks it correctly).
    """
    if excluded_abl is None:
        excluded_abl = {"no_depth", "no_atmosphere_lighting"}

    pts: List[Point] = []
    levels = STRUCT_LEVELS[1:] if drop_l5 else STRUCT_LEVELS

    def _se(vs):
        return _bootstrap_se(vs, se_bootstrap_B) if compute_se else 0.0

    for lv in levels:
        vs = by_cell.get(("structured", lv))
        if vs and ("s", lv) in mse_at:
            pts.append((aggregator(vs), _se(vs),
                        mse_at[("s", lv)], f"st_{lv}", "struct"))

    for lv in DENSE_LEVELS:
        vs = by_cell.get(("dense", lv))
        if vs and ("d", lv) in mse_at:
            pts.append((aggregator(vs), _se(vs),
                        mse_at[("d", lv)], f"d_{lv}", "dense"))

    for k in ABL_FILES:
        if k in excluded_abl:
            continue
        kind = f"abl_json_{k}"
        vs = by_cell.get((kind, "l10"))
        if vs and ("a", k) in mse_at:
            pts.append((aggregator(vs), _se(vs),
                        mse_at[("a", k)], k, "abl"))

    for k in SPATIAL_FILES:
        kind = f"spatial_{k}"
        vs = by_cell.get((kind, "l10"))
        if vs and ("sp", k) in mse_at:
            pts.append((aggregator(vs), _se(vs),
                        mse_at[("sp", k)], f"sp_{k}", "spatial"))

    return pts


# ---------------------------------------------------------------------------
# Spearman (rank-based) correlation for ρ headline number
# ---------------------------------------------------------------------------

def spearman(xs: List[float], ys: List[float]) -> float:
    """Spearman ρ via averaged-rank Pearson on (rank(xs), rank(ys))."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def _ranks(vs: List[float]) -> List[float]:
        idx = sorted(range(n), key=lambda i: vs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[idx[j + 1]] == vs[idx[i]]:
                j += 1
            avg = (i + j + 2) / 2
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r

    rx, ry = _ranks(xs), _ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    sx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    sy = math.sqrt(sum((r - my) ** 2 for r in ry))
    return cov / (sx * sy) if sx * sy else float("nan")


def inversions(xs: List[float], ys: List[float], sign: int = -1) -> int:
    """Count pairs (i, j) where sign(xs[i] - xs[j]) and sign(ys[i] - ys[j])
    DISAGREE with the expected `sign` (default −1 for ED → MSE)."""
    n = len(xs)
    inv = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 or dy == 0:
                continue
            if (dx * dy > 0 and sign == -1) or (dx * dy < 0 and sign == 1):
                inv += 1
    return inv


# ---------------------------------------------------------------------------
# Convenience: one-shot load (mirror gpg.io.load_v6_dropmore)
# ---------------------------------------------------------------------------

def load_v41(cache_dir: str = DEFAULT_CACHE_DIR,
             pool_path: str = DEFAULT_POOL_PATH,
             wandb_dirs: Optional[Dict[str, str]] = None,
             drop_l5: bool = False,
             se_bootstrap_B: int = 1000) -> Tuple[List[Point], float]:
    """One-liner that reproduces the 16-point paper main result.

    Returns (points, unified_budget). Each point is
    (ed_trim20, ed_se, mse, label, category).

    >>> from detailness.ed.io import load_v41
    >>> pts, budget = load_v41()
    >>> len(pts)
    16
    >>> # struct/l10 should be the lowest-MSE point
    >>> top = min(pts, key=lambda p: p[2])
    >>> assert top[3] == "st_l10"
    """
    runs   = load_all_loss_runs(wandb_dirs)
    budget = unified_budget(runs)
    mse_at = {k: smoothed_mse(t, budget) for k, t in runs.items()}

    pool        = load_pool(pool_path)
    match_cache = load_match(version="v41", cache_dir=cache_dir)
    by_cell     = aggregate_per_cell(pool, match_cache, category="A", beta=0.5)

    return build_16_points(by_cell, mse_at,
                           drop_l5=drop_l5,
                           se_bootstrap_B=se_bootstrap_B), budget
