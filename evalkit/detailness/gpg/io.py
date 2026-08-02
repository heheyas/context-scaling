# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Shared I/O for GPG analysis: load Bagel training-loss CSVs and
per-sample GPG shards, build the joined 16-point dataset.

Path defaults match the v6_dropmore paper run. Override via env vars or
function args when re-running on a different machine.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_WANDB_ROOT = os.environ.get("WANDB_ROOT", "")

DEFAULT_WANDB_DIRS = {
    "struct":  f"{_WANDB_ROOT}/wandb_exports",
    "dense":   f"{_WANDB_ROOT}/wandb_nl_exports",
    "abl":     f"{_WANDB_ROOT}/wandb_exports_ablations",
    "spatial": f"{_WANDB_ROOT}/wandb_exports_bbox_variants",
}

DEFAULT_GPG_DIR = "/tmp/detailness_real_n100/gpg_v6_dropmore"

# (kind, level) → MSE-loss-task key used in the W&B CSV
TASK_KEY = {
    "structured": "t2i_ct_json",
    "dense":      "t2i_ct_dense",
    "abl_json":   "t2i_ct_json",
    "spatial":    "t2i_ct_json_spatial",
}

STRUCT_LEVELS = ["l5", "l6", "l7", "l8", "l9", "l10"]
DENSE_LEVELS  = ["l6", "l8", "l10"]

# CSV filename → ablation key
ABL_FILES = {
    "full":                    "full.csv",
    "no_bbox":                 "no-bbox.csv",
    "no_depth":                "no_depth.csv",
    "no_atmosphere_lighting":  "no-atomsphere-lighting.csv",  # (sic — typo in source)
    "no_relationships":        "no-relations.csv",
    "no_scene":                "no-scene.csv",
}
SPATIAL_FILES = {"coarse": "coarse.csv", "fine": "fine.csv", "finer": "finer.csv"}


# ---------------------------------------------------------------------------
# Bagel training-loss CSVs → step-aligned (cum_tokens, mse) trajectory
# ---------------------------------------------------------------------------

def _flt(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def load_trajectory(csv_files: List[str], step_col: str, mse_col: str,
                    tok_col: str) -> List[Tuple[float, float]]:
    """Concat multiple W&B CSVs, sort by step, produce (cum_tokens, mse) pairs.

    `tok_col` is the per-step image-MSE token count for the target task.
    """
    traj: List[Tuple[float, float]] = []
    cum = 0.0
    for fp in csv_files:
        rows = []
        with open(fp) as f:
            for r in csv.DictReader(f):
                st = _flt(r.get(step_col, ""))
                m  = _flt(r.get(mse_col, ""))
                t  = _flt(r.get(tok_col, ""))
                if st is None:
                    continue
                rows.append((int(st), m, t))
        rows.sort()
        last_mse: Optional[float] = None
        for step, mse, tok in rows:
            if tok and tok > 0:
                cum += tok
            if mse is not None:
                last_mse = mse
            if last_mse is not None and cum > 0:
                traj.append((cum, last_mse))
    return traj


def smoothed_mse(traj: List[Tuple[float, float]], target_x: float,
                 window_frac: float = 0.02) -> Optional[float]:
    """Median MSE in a window of ±window_frac × target_x around the target
    cumulative-token budget."""
    lo, hi = target_x * (1 - window_frac), target_x * (1 + window_frac)
    ys = sorted([y for x, y in traj if lo <= x <= hi])
    return ys[len(ys) // 2] if ys else None


def load_all_loss_runs(wandb_dirs: Optional[Dict[str, str]] = None
                       ) -> Dict[Tuple[str, str], List[Tuple[float, float]]]:
    """Returns dict keyed by ('s'|'d'|'a'|'sp', level/abl/spatial_key)."""
    dirs = wandb_dirs or DEFAULT_WANDB_DIRS
    runs: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}

    for lv in STRUCT_LEVELS:
        files = [os.path.join(dirs["struct"], f"{lv}.csv")]
        cont = os.path.join(dirs["struct"], f"{lv}-from4k6.csv")
        if os.path.exists(cont):
            files.append(cont)
        files = [f for f in files if os.path.exists(f)]
        if files:
            runs[("s", lv)] = load_trajectory(
                files, "step", "mse/t2i_ct_json", "mse_token/t2i_ct_json"
            )

    for lv in DENSE_LEVELS:
        fp = os.path.join(dirs["dense"], f"{lv}-nl.csv")
        if os.path.exists(fp):
            runs[("d", lv)] = load_trajectory(
                [fp], "step", "mse/t2i_ct_dense", "mse_token/t2i_ct_dense"
            )

    for k, fn in ABL_FILES.items():
        fp = os.path.join(dirs["abl"], fn)
        if os.path.exists(fp):
            runs[("a", k)] = load_trajectory(
                [fp], "step", "mse/t2i_ct_json", "mse_token/t2i_ct_json"
            )

    for k, fn in SPATIAL_FILES.items():
        fp = os.path.join(dirs["spatial"], fn)
        if os.path.exists(fp):
            runs[("sp", k)] = load_trajectory(
                [fp], "step",
                "mse/t2i_ct_json_spatial",
                "mse_token/t2i_ct_json_spatial",
            )

    return runs


def unified_budget(runs: Dict[Tuple[str, str], List[Tuple[float, float]]]) -> float:
    """Pick the earliest run-end so every (kind, level) has an MSE there."""
    return min(t[-1][0] for t in runs.values())


# ---------------------------------------------------------------------------
# GPG shard records → per-(kind, level) total GPG (nats/caption, averaged
# over uids in the cell)
# ---------------------------------------------------------------------------

def load_gpg_records(gpg_dir: str, num_shards: int = 8) -> List[dict]:
    """Read shard{i}.jsonl files from a GPG run dir."""
    recs = []
    for i in range(num_shards):
        fp = os.path.join(gpg_dir, f"shard{i}.jsonl")
        if os.path.exists(fp):
            with open(fp) as f:
                recs += [json.loads(l) for l in f]
    return recs


def aggregate_gpg(records: List[dict]) -> Dict[Tuple[str, str], List[float]]:
    """Group per-sample records by (kind, level), return list of per-sample
    total GPG values (= gpg_mean * n_tokens) for each cell."""
    out: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in records:
        out[(r["kind"], r["level"])].append(r["gpg"] * r["n_tokens"])
    return dict(out)


def mean_se(xs: List[float]) -> Tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    s = math.sqrt(sum((v - m) ** 2 for v in xs) / (n - 1)) if n > 1 else 0.0
    return m, (s / math.sqrt(n)) if n > 1 else 0.0


# ---------------------------------------------------------------------------
# Joined 16-point dataset
# ---------------------------------------------------------------------------

# Schema: list of (gpg_mean, gpg_se, mse, label, category)
Point = Tuple[float, float, float, str, str]


def build_16_points(records_by_cell: Dict[Tuple[str, str], List[float]],
                    mse_at: Dict[Tuple[str, str], float],
                    excluded_abl=None,
                    drop_l5: bool = False) -> List[Point]:
    """Join GPG (per kind,level) with Bagel MSE (at unified budget).

    Mirrors the v6_dropmore figure recipe. Drops abl_no_depth and
    abl_no_atmosphere_lighting by default (rerun pending)."""
    if excluded_abl is None:
        excluded_abl = {"no_depth", "no_atmosphere_lighting"}

    pts: List[Point] = []
    levels = STRUCT_LEVELS[1:] if drop_l5 else STRUCT_LEVELS

    for lv in levels:
        if records_by_cell.get(("structured", lv)) and ("s", lv) in mse_at:
            g, gs = mean_se(records_by_cell[("structured", lv)])
            pts.append((g, gs, mse_at[("s", lv)], f"st_{lv}", "struct"))

    for lv in DENSE_LEVELS:
        if records_by_cell.get(("dense", lv)) and ("d", lv) in mse_at:
            g, gs = mean_se(records_by_cell[("dense", lv)])
            pts.append((g, gs, mse_at[("d", lv)], f"d_{lv}", "dense"))

    for k in ABL_FILES:
        if k in excluded_abl:
            continue
        kind = f"abl_json_{k}"
        if records_by_cell.get((kind, "l10")) and ("a", k) in mse_at:
            g, gs = mean_se(records_by_cell[(kind, "l10")])
            pts.append((g, gs, mse_at[("a", k)], k, "abl"))

    for k in SPATIAL_FILES:
        kind = f"spatial_{k}"
        if records_by_cell.get((kind, "l10")) and ("sp", k) in mse_at:
            g, gs = mean_se(records_by_cell[(kind, "l10")])
            pts.append((g, gs, mse_at[("sp", k)], f"sp_{k}", "spatial"))

    return pts


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def pearson(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """Return (r, slope, intercept) for y = slope*x + intercept."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_xx = sum((x - mx) ** 2 for x in xs)
    sx = math.sqrt(den_xx / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    r = num / (n * sx * sy) if sx * sy else 0.0
    slope = num / den_xx if den_xx else 0.0
    intercept = my - slope * mx
    return r, slope, intercept


# ---------------------------------------------------------------------------
# Convenience: one-shot load
# ---------------------------------------------------------------------------

def load_v6_dropmore(gpg_dir: str = DEFAULT_GPG_DIR,
                     wandb_dirs: Optional[Dict[str, str]] = None,
                     drop_l5: bool = False) -> Tuple[List[Point], float]:
    """Returns (points, unified_budget) for the v6_dropmore recipe.

    Drops abl_no_depth and abl_no_atmosphere_lighting by default.
    """
    runs = load_all_loss_runs(wandb_dirs)
    budget = unified_budget(runs)
    mse_at = {k: smoothed_mse(t, budget) for k, t in runs.items()}
    by_cell = aggregate_gpg(load_gpg_records(gpg_dir))
    return build_16_points(by_cell, mse_at, drop_l5=drop_l5), budget
