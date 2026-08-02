# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Aggregate GPG runs vs Bagel training MSE, produce the 16-point fit
figure and per-experiment plots.

By default compares the 5 overnight experiments (v3, v5, v6, v7, v8). The
v6_dropmore one is the paper version. Single-experiment mode is also
supported via --gpg-dir.

Usage:
    # Default: sweep all 5 overnight experiments + write grid figure
    python -m detailness.gpg.analyze

    # Single experiment, custom output dir
    python -m detailness.gpg.analyze \\
        --gpg-dir /tmp/detailness_real_n100/gpg_v6_dropmore \\
        --fig-dir /tmp/figures \\
        --tag v6_dropmore
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io import (
    ABL_FILES,
    SPATIAL_FILES,
    DEFAULT_GPG_DIR,
    DEFAULT_WANDB_DIRS,
    aggregate_gpg,
    build_16_points,
    load_all_loss_runs,
    load_gpg_records,
    pearson,
    smoothed_mse,
    unified_budget,
)


# Standard "overnight" experiments (5 settings swept against the same pool)
DEFAULT_EXPERIMENTS = [
    ("v3_canonical (baseline)",         "/tmp/detailness_real_n100/gpg_qwen3vl_v3"),
    ("v5_schema (system prompt)",       "/tmp/detailness_real_n100/gpg_v5_schema"),
    ("v6_dropmore (extra canonicalize)","/tmp/detailness_real_n100/gpg_v6_dropmore"),
    ("v7_combined (schema + dropmore)", "/tmp/detailness_real_n100/gpg_v7_combined"),
    ("v8_super (super aggressive)",     "/tmp/detailness_real_n100/gpg_v8_super"),
]

# Color/marker per category
CAT_STYLE = {
    "struct":  ("#1f77b4", "o", "struct l5-l10"),
    "dense":   ("#d62728", "s", "dense"),
    "spatial": ("#ff7f0e", "D", "spatial"),
    "abl":     ("#2ca02c", "^", "abl"),
}


def _pearson_pts(pts):
    xs = [p[0] for p in pts]
    ys = [p[2] for p in pts]
    return pearson(xs, ys)


def collect_pts_for_dir(gpg_dir: str, mse_at: Dict[Tuple[str, str], float],
                        drop_l5: bool = False):
    recs = load_gpg_records(gpg_dir)
    if not recs:
        return None
    return build_16_points(aggregate_gpg(recs), mse_at, drop_l5=drop_l5)


def plot_one(ax, pts, name: str, budget: float):
    r, slope, intercept = _pearson_pts(pts)

    for cat, (color, marker, lbl) in CAT_STYLE.items():
        if cat == "abl":
            sub = [p for p in pts if p[4] == "abl"]
        else:
            sub = [p for p in pts if p[4] == cat]
        if not sub:
            continue
        if cat == "struct":
            sub = sorted(sub)
            ax.plot([p[0] for p in sub], [p[2] for p in sub],
                    "-", color=color, alpha=0.5)
        ax.errorbar([p[0] for p in sub], [p[2] for p in sub],
                    xerr=[p[1] for p in sub],
                    fmt=marker, color=color, markersize=10, capsize=2,
                    label=f"{lbl} (n={len(sub)})")
        if cat == "abl":
            for p in sub:
                ax.annotate(f" {p[3]}", (p[0], p[2]), fontsize=7, color=color)

    xline = [min(p[0] for p in pts) - 3, max(p[0] for p in pts) + 3]
    ax.plot(xline, [intercept + slope * x for x in xline],
            ":", color="gray", linewidth=1.2, label=f"r={r:+.4f}")
    ax.set_xlabel("Total GPG (nats/caption)")
    ax.set_ylabel(f"Bagel MSE @ {budget:.2e} tokens")
    ax.set_title(f"{name}\nn={len(pts)}, r={r:+.4f}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=7)


def print_summary_table(named_pts: List[Tuple[str, List]]):
    print(f"\n{'experiment':40s} {'n':>4s} {'r':>9s} {'l10 max?':>10s} "
          f"{'sp gap':>8s} {'no_bbox Δ':>16s}")
    print("-" * 100)
    for name, pts in named_pts:
        if pts is None or len(pts) < 10:
            print(f"{name:40s} (no data)")
            continue
        r, _, _ = _pearson_pts(pts)
        gpg_l10 = next((p[0] for p in pts if p[3] == "st_l10"), None)
        gpg_max = max(p[0] for p in pts)
        l10_max = (gpg_l10 is not None
                   and (gpg_l10 == gpg_max or abs(gpg_l10 - gpg_max) < 0.01))
        sp_co = next((p[0] for p in pts if p[3] == "sp_coarse"), None)
        sp_fr = next((p[0] for p in pts if p[3] == "sp_finer"), None)
        sp_gap = (sp_fr - sp_co) if (sp_co and sp_fr) else None
        g_full = next((p[0] for p in pts if p[3] == "full"), None)
        g_nob = next((p[0] for p in pts if p[3] == "no_bbox"), None)
        nob = (g_nob - g_full) if (g_nob and g_full) else None
        print(f"{name:40s} {len(pts):>4d} {r:>+9.4f} "
              f"{('✓' if l10_max else '✗'):>10s} "
              f"{(f'{sp_gap:+.1f}' if sp_gap else '—'):>8s} "
              f"{(f'{nob:+.1f}' if nob else '—'):>16s}")


def print_sorted_with_violations(named_pts: List[Tuple[str, List]]):
    print("\n\n=== FULL SORTED GPG by experiment ===")
    for name, pts in named_pts:
        if pts is None:
            continue
        r, _, _ = _pearson_pts(pts)
        print(f"\n--- {name} (r={r:+.4f}) ---")
        ps = sorted(pts)
        for p in ps:
            print(f"  GPG={p[0]:7.2f}  MSE={p[2]:.5f}  {p[3]}")
        viol = 0
        viol_sum = 0.0
        for i in range(len(ps) - 1):
            d = ps[i + 1][2] - ps[i][2]
            if d > 0:
                viol += 1
                viol_sum += d
        print(f"  monotonic violations: {viol} (total Δ={viol_sum:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpg-dir", default=None,
                    help="If set, score this single dir instead of sweeping experiments")
    ap.add_argument("--tag", default="single",
                    help="Used in figure filename when --gpg-dir is set")
    ap.add_argument("--struct-dir", default=DEFAULT_WANDB_DIRS["struct"])
    ap.add_argument("--dense-dir",  default=DEFAULT_WANDB_DIRS["dense"])
    ap.add_argument("--abl-dir",    default=DEFAULT_WANDB_DIRS["abl"])
    ap.add_argument("--spatial-dir",default=DEFAULT_WANDB_DIRS["spatial"])
    ap.add_argument("--fig-dir",
                    default=os.environ.get("FIGURES_DIR", "./figures"))
    ap.add_argument("--drop-l5", action="store_true",
                    help="Exclude struct_l5 outlier from the fit")
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    wandb_dirs = {
        "struct":  args.struct_dir,
        "dense":   args.dense_dir,
        "abl":     args.abl_dir,
        "spatial": args.spatial_dir,
    }
    runs = load_all_loss_runs(wandb_dirs)
    budget = unified_budget(runs)
    mse_at = {k: smoothed_mse(t, budget) for k, t in runs.items()}
    print(f"unified budget = {budget:.4e} image-MSE tokens")

    if args.gpg_dir:
        experiments = [(args.tag, args.gpg_dir)]
    else:
        experiments = DEFAULT_EXPERIMENTS

    named_pts = [(name, collect_pts_for_dir(d, mse_at, args.drop_l5))
                 for name, d in experiments]
    print_summary_table(named_pts)

    # Grid figure (one panel per experiment)
    n = len(experiments)
    cols = 3 if n > 3 else n
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows),
                             squeeze=False)
    axes = axes.flatten()
    for ax, (name, pts) in zip(axes, named_pts):
        if pts is None:
            ax.set_title(f"{name}\nNO DATA")
            continue
        plot_one(ax, pts, name, budget)
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    suffix = args.tag if args.gpg_dir else "overnight"
    out = os.path.join(args.fig_dir, f"gpg_{suffix}_grid.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved {out}")

    # Per-experiment standalone plots
    for name, pts in named_pts:
        if pts is None:
            continue
        fig, ax = plt.subplots(figsize=(11, 7))
        plot_one(ax, pts, name, budget)
        safe = (name.replace(" ", "_").replace("(", "")
                    .replace(")", "").replace(",", ""))
        out = os.path.join(args.fig_dir, f"gpg_{suffix}_{safe}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=160, bbox_inches="tight")
        print(f"  saved {out}")

    print_sorted_with_violations(named_pts)


if __name__ == "__main__":
    main()
