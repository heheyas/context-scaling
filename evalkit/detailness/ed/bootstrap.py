# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Bootstrap-resample ED ρ over N=300 uids, report mean / 95% CI / inv
distribution. Used to validate that the single-sample ρ = -0.9087 is not
an outlier on the lucky side.

Also runs a quick N-scaling experiment (sub-sample without replacement,
fit σ ∝ N^-1 to extrapolate ρ stability at larger N).

Usage:
    HOME=<RAY_SERVE_HOME> python -m detailness.ed.bootstrap \\
        --B 10000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from .io import (
    DEFAULT_AGGREGATE_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_POOL_PATH,
    DEFAULT_WANDB_DIRS,
    cap_key,
    inversions,
    load_all_loss_runs,
    load_match,
    load_pool,
    per_row_F,
    smoothed_mse,
    spearman,
    trim20,
    unified_budget,
)
from ..gpg.io import ABL_FILES, DENSE_LEVELS, SPATIAL_FILES, STRUCT_LEVELS


# (paper-cat, paper-label) → (pool-kind, pool-level). Same convention as
# build_16_points but inverted (we resample by pool-key here).
CELL_MAP = {}
for lv in STRUCT_LEVELS:
    CELL_MAP[("struct", lv)] = ("structured", lv)
for lv in DENSE_LEVELS:
    CELL_MAP[("dense", lv)] = ("dense", lv)
for k in ABL_FILES:
    CELL_MAP[("abl", k)] = (f"abl_json_{k}", "l10")
for k in SPATIAL_FILES:
    CELL_MAP[("spatial", k)] = (f"spatial_{k}", "l10")


def _gather_per_uid_F(pool: List[dict],
                      match_cache: Dict[str, dict]
                      ) -> Dict[str, Dict[Tuple[str, str], float]]:
    """{uid: {(kind, level): F_v4.1}} — pre-index for cheap bootstrap."""
    out: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(dict)
    for r in pool:
        v = match_cache.get(cap_key(r["uid"], r["caption"]))
        if v is None:
            continue
        out[r["uid"]][(r["kind"], r["level"])] = per_row_F(v, "A", 0.5)
    return out


def _compute_rho_inv(per_uid: Dict[str, Dict[Tuple[str, str], float]],
                     uids: List[str],
                     mse_at: Dict[Tuple[str, str], float],
                     excluded_abl: set = None
                     ) -> Tuple[float, int]:
    """Aggregate per-cell trim20 over `uids` (with replacement), then ρ vs MSE."""
    if excluded_abl is None:
        excluded_abl = {"no_depth", "no_atmosphere_lighting"}

    pool_lookup = {
        ("s", lv):  ("structured", lv)            for lv in STRUCT_LEVELS
    } | {
        ("d", lv):  ("dense", lv)                 for lv in DENSE_LEVELS
    } | {
        ("a", k):   (f"abl_json_{k}", "l10")       for k in ABL_FILES
    } | {
        ("sp", k):  (f"spatial_{k}", "l10")        for k in SPATIAL_FILES
    }

    cell_to_mse = {}
    cell_to_vals = defaultdict(list)
    for mse_key, kl in pool_lookup.items():
        if mse_key[0] == "a" and mse_key[1] in excluded_abl:
            continue
        if mse_key not in mse_at:
            continue
        cell_to_mse[kl] = mse_at[mse_key]

    for u in uids:
        d = per_uid.get(u, {})
        for kl, mse in cell_to_mse.items():
            f = d.get(kl)
            if f is not None:
                cell_to_vals[kl].append(f)

    xs, ys = [], []
    for kl, mse in cell_to_mse.items():
        vs = cell_to_vals.get(kl, [])
        if vs:
            xs.append(trim20(vs))
            ys.append(mse)
    return spearman(xs, ys), inversions(xs, ys, sign=-1)


def main():
    ap = argparse.ArgumentParser(
        description="Bootstrap ED ρ over the N=300 uid pool")
    ap.add_argument("--B", type=int, default=10000,
                    help="Bootstrap reps (paper: 10,000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--matcher", choices=("gpt", "gemini"), default="gpt")
    ap.add_argument("--pool",      default=DEFAULT_POOL_PATH)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--out", default=None,
                    help="Output JSON path "
                         "(default: bootstrap_<version>.json under per-cond dir)")
    ap.add_argument("--per-cond-dir", default=DEFAULT_AGGREGATE_DIR)
    ap.add_argument("--struct-dir",  default=DEFAULT_WANDB_DIRS["struct"])
    ap.add_argument("--dense-dir",   default=DEFAULT_WANDB_DIRS["dense"])
    ap.add_argument("--abl-dir",     default=DEFAULT_WANDB_DIRS["abl"])
    ap.add_argument("--spatial-dir", default=DEFAULT_WANDB_DIRS["spatial"])
    ap.add_argument("--n-scaling", action="store_true",
                    help="Also run a sub-sample N-scaling experiment")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    version = "v41" if args.matcher == "gpt" else "v24"
    out_path = args.out or os.path.join(args.per_cond_dir,
                                        f"bootstrap_{version}.json")
    os.makedirs(args.per_cond_dir, exist_ok=True)

    # MSE side
    wandb_dirs = {
        "struct":  args.struct_dir, "dense": args.dense_dir,
        "abl":     args.abl_dir,    "spatial": args.spatial_dir,
    }
    runs   = load_all_loss_runs(wandb_dirs)
    budget = unified_budget(runs)
    mse_at = {k: smoothed_mse(t, budget) for k, t in runs.items()}

    pool        = load_pool(args.pool)
    match_cache = load_match(version=version, cache_dir=args.cache_dir)
    per_uid     = _gather_per_uid_F(pool, match_cache)
    uids        = sorted(per_uid)
    n_uid       = len(uids)
    print(f"N uids = {n_uid}, B = {args.B}, matcher = {args.matcher}")

    # === full-sample baseline ===
    rho_full, inv_full = _compute_rho_inv(per_uid, uids, mse_at)
    print(f"\nFull-sample: ρ = {rho_full:+.4f}, inv = {inv_full}")

    # === bootstrap ===
    rhos: List[float] = []
    invs: List[int] = []
    for b in range(args.B):
        sample = [uids[rng.randrange(n_uid)] for _ in range(n_uid)]
        rho, inv = _compute_rho_inv(per_uid, sample, mse_at)
        rhos.append(rho)
        invs.append(inv)
        if (b + 1) % max(1, args.B // 10) == 0:
            print(f"  bootstrap {b+1}/{args.B}: "
                  f"running mean ρ = {sum(rhos)/len(rhos):+.4f}")

    rhos.sort()
    mean_rho = sum(rhos) / args.B
    var_rho  = sum((r - mean_rho) ** 2 for r in rhos) / (args.B - 1)
    std_rho  = math.sqrt(var_rho)

    def pct(p):
        return rhos[max(0, min(args.B - 1, int(args.B * p)))]

    summary = {
        "matcher": args.matcher,
        "version": version,
        "n_uid": n_uid,
        "B": args.B,
        "full_sample_rho": rho_full,
        "full_sample_inv": inv_full,
        "bootstrap_mean_rho": mean_rho,
        "bootstrap_std_rho":  std_rho,
        "bootstrap_median_rho": rhos[args.B // 2],
        "ci95":   [pct(0.025), pct(0.975)],
        "ci99":   [pct(0.005), pct(0.995)],
        "P_rho_lt_-0.90": sum(1 for r in rhos if r < -0.90) / args.B,
        "P_rho_lt_-0.93": sum(1 for r in rhos if r < -0.93) / args.B,
        "P_rho_lt_-0.95": sum(1 for r in rhos if r < -0.95) / args.B,
        "mean_inv": sum(invs) / args.B,
    }

    print(f"\n--- Bootstrap ρ (B={args.B}) ---")
    print(f"  mean ρ          = {mean_rho:+.4f}")
    print(f"  median ρ        = {summary['bootstrap_median_rho']:+.4f}")
    print(f"  std σ           = {std_rho:.4f}")
    print(f"  95 % CI         = [{summary['ci95'][0]:+.4f}, "
          f"{summary['ci95'][1]:+.4f}]")
    print(f"  99 % CI         = [{summary['ci99'][0]:+.4f}, "
          f"{summary['ci99'][1]:+.4f}]")
    print(f"  P(ρ < -0.90)    = {summary['P_rho_lt_-0.90']:.3f}")
    print(f"  P(ρ < -0.93)    = {summary['P_rho_lt_-0.93']:.3f}")
    print(f"  P(ρ < -0.95)    = {summary['P_rho_lt_-0.95']:.3f}")
    print(f"  mean inversions = {summary['mean_inv']:.2f}")

    # === N-scaling (subsample WITHOUT replacement) ===
    if args.n_scaling:
        scaling: Dict[int, List[float]] = {}
        for N in [50, 75, 100, 150, 200, 250]:
            rs = []
            for _ in range(min(args.B, 200)):
                sample = rng.sample(uids, N)
                rho, _ = _compute_rho_inv(per_uid, sample, mse_at)
                rs.append(rho)
            scaling[N] = rs
        nstats = {}
        print("\n--- N-scaling (subsample without replacement) ---")
        print(f"  {'N':>4s} {'mean ρ':>10s} {'σ':>8s}")
        for N in sorted(scaling):
            rs = scaling[N]
            m = sum(rs) / len(rs)
            s = math.sqrt(sum((r - m) ** 2 for r in rs) / max(1, len(rs) - 1))
            nstats[N] = {"mean": m, "std": s, "n_reps": len(rs)}
            print(f"  {N:>4d} {m:>+10.4f} {s:>8.4f}")
        summary["n_scaling"] = nstats

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
