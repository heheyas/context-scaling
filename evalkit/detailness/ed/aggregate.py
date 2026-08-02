# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Aggregate per-row F_{β=0.5}(P_A, R_A) into per-cell trim20 mean, join
with Bagel image-MSE at the unified token budget, fit a power law, and
emit the paper figure + 16-point CSV.

By default reproduces the v4.1 paper-main number (GPT-4o matcher,
ρ = -0.9087, inv = 15/120). Use `--matcher gemini` for the cross-matcher
robustness run; the script auto-points at the v24_match_gemini.jsonl cache
in that case.

Usage:
    # paper-main (GPT-4o matcher)
    HOME=<RAY_SERVE_HOME> python -m detailness.ed.aggregate

    # cross-matcher (Gemini)
    HOME=<RAY_SERVE_HOME> python -m detailness.ed.aggregate \\
        --matcher gemini
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import (
    DEFAULT_AGGREGATE_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_POOL_PATH,
    DEFAULT_WANDB_DIRS,
    Point,
    aggregate_per_cell,
    build_16_points,
    inversions,
    load_all_loss_runs,
    load_match,
    load_pool,
    median,
    smoothed_mse,
    spearman,
    trim20,
    unified_budget,
)


# Color / marker convention (matches gpg/analyze.py)
CAT_STYLE = {
    "struct":  ("#1f77b4", "o", "struct l5-l10"),
    "dense":   ("#d62728", "s", "dense"),
    "spatial": ("#ff7f0e", "D", "spatial"),
    "abl":     ("#2ca02c", "^", "abl"),
}


def _fit_power_law(pts: List[Point]) -> Tuple[float, float, float]:
    """ED, MSE → fit MSE = a · ED^b on log-log. Returns (a, b, R²)."""
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[2] for p in pts])
    b, log_a = np.polyfit(np.log(xs), np.log(ys), 1)
    a = math.exp(log_a)
    yhat = a * xs ** b
    ss_res = float(np.sum((ys - yhat) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return float(a), float(b), r2


def _plot_one(ax, pts: List[Point], name: str, budget: float,
              power_law: Tuple[float, float, float]):
    xs = [p[0] for p in pts]
    ys = [p[2] for p in pts]
    rho = spearman(xs, ys)
    inv = inversions(xs, ys, sign=-1)
    a, b, r2 = power_law

    for cat, (color, marker, lbl) in CAT_STYLE.items():
        sub = [p for p in pts if p[4] == cat]
        if not sub:
            continue
        if cat == "struct":
            sub = sorted(sub)
            ax.plot([p[0] for p in sub], [p[2] for p in sub],
                    "-", color=color, alpha=0.45)
        ax.errorbar([p[0] for p in sub], [p[2] for p in sub],
                    xerr=[p[1] for p in sub],
                    fmt=marker, color=color, markersize=9, capsize=2,
                    label=f"{lbl} (n={len(sub)})")
        if cat == "abl":
            for p in sub:
                ax.annotate(f" {p[3]}", (p[0], p[2]),
                            fontsize=7, color=color)

    # Power-law fit curve
    x_grid = np.linspace(min(xs) * 0.995, max(xs) * 1.005, 200)
    ax.plot(x_grid, a * x_grid ** b, "-", color="black", linewidth=1.5,
            alpha=0.8, label=f"MSE = {a:.3f}·ED^{b:.3f}  R²={r2:.3f}")

    ax.set_xlabel("ED = trim20-mean of $F_{0.5}(P_A, R_A)$ (image-grounded)")
    ax.set_ylabel(f"Bagel MSE @ {budget:.2e} tokens")
    ax.set_title(f"{name}\nn={len(pts)}, ρ={rho:+.4f}, inv={inv}/{len(xs)*(len(xs)-1)//2}",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=7)


def _print_summary(name: str, pts: List[Point], a: float, b: float, r2: float):
    xs = [p[0] for p in pts]
    ys = [p[2] for p in pts]
    rho = spearman(xs, ys)
    inv = inversions(xs, ys, sign=-1)
    print(f"\n=== {name} ===")
    print(f"  n          = {len(pts)}")
    print(f"  Spearman ρ = {rho:+.4f}")
    print(f"  inversions = {inv}/{len(xs)*(len(xs)-1)//2}")
    print(f"  power law  : MSE = {a:.4f} · ED^{b:.4f}, R² = {r2:.4f}")


def _print_sorted(pts: List[Point]):
    print("\n  sorted by MSE (best → worst):")
    print(f"  {'label':18s} {'cat':9s} {'ED':>9s} {'SE':>7s} {'MSE':>9s}")
    for p in sorted(pts, key=lambda x: x[2]):
        print(f"  {p[3]:18s} {p[4]:9s} {p[0]:>9.4f} {p[1]:>7.4f} {p[2]:>9.5f}")


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate ED 16-point table + power-law fit + figure")
    ap.add_argument("--matcher", choices=("gpt", "gemini"), default="gpt",
                    help="Which match cache to aggregate. gpt = paper main "
                         "(v4.1, v4_image_match.jsonl); gemini = cross-matcher "
                         "robustness (v24_match_gemini.jsonl)")
    ap.add_argument("--pool", default=DEFAULT_POOL_PATH)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--aggregator", choices=("trim20", "median"),
                    default="trim20",
                    help="Per-cell aggregation of per-uid F values "
                         "(paper main: trim20)")
    ap.add_argument("--struct-dir",  default=DEFAULT_WANDB_DIRS["struct"])
    ap.add_argument("--dense-dir",   default=DEFAULT_WANDB_DIRS["dense"])
    ap.add_argument("--abl-dir",     default=DEFAULT_WANDB_DIRS["abl"])
    ap.add_argument("--spatial-dir", default=DEFAULT_WANDB_DIRS["spatial"])
    ap.add_argument("--fig-dir",
                    default=os.environ.get("FIGURES_DIR", "./figures"))
    ap.add_argument("--per-cond-dir", default=DEFAULT_AGGREGATE_DIR)
    ap.add_argument("--tag", default=None,
                    help="Figure / CSV filename suffix "
                         "(default: 'v41' for gpt, 'v24' for gemini)")
    ap.add_argument("--se-bootstrap-B", type=int, default=1000,
                    help="Bootstrap reps for per-cell SE")
    ap.add_argument("--drop-l5", action="store_true",
                    help="Exclude struct/l5 outlier")
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(args.per_cond_dir, exist_ok=True)

    # MSE side — re-uses GPG's loader
    wandb_dirs = {
        "struct":  args.struct_dir,
        "dense":   args.dense_dir,
        "abl":     args.abl_dir,
        "spatial": args.spatial_dir,
    }
    runs   = load_all_loss_runs(wandb_dirs)
    budget = unified_budget(runs)
    mse_at = {k: smoothed_mse(t, budget) for k, t in runs.items()}
    print(f"unified budget = {budget:.4e} image-MSE tokens")

    # ED side
    version = "v41" if args.matcher == "gpt" else "v24"
    tag     = args.tag or version
    pool        = load_pool(args.pool)
    match_cache = load_match(version=version, cache_dir=args.cache_dir)
    by_cell     = aggregate_per_cell(pool, match_cache,
                                     category="A", beta=0.5)

    aggregator = trim20 if args.aggregator == "trim20" else median
    pts = build_16_points(by_cell, mse_at,
                          drop_l5=args.drop_l5,
                          aggregator=aggregator,
                          se_bootstrap_B=args.se_bootstrap_B)

    a, b, r2 = _fit_power_law(pts)
    name = (f"v4.1 paper-main (GPT-4o matcher)"
            if args.matcher == "gpt"
            else "v24 cross-matcher (Gemini-3-pro)")
    _print_summary(name, pts, a, b, r2)
    _print_sorted(pts)

    # CSV
    csv_path = os.path.join(args.per_cond_dir, f"per_cond_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "label", "ED", "SE", "MSE"])
        for p in sorted(pts, key=lambda x: x[2]):
            w.writerow([p[4], p[3], p[0], p[1], p[2]])
    print(f"\nSaved {csv_path}")

    # Figure
    fig, ax = plt.subplots(figsize=(11, 7))
    _plot_one(ax, pts, name, budget, (a, b, r2))
    plt.tight_layout()
    fig_path = os.path.join(args.fig_dir, f"ed_{tag}.png")
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
