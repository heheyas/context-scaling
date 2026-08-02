# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Compare GPG ↔ Bagel-MSE fit across multiple judge VLMs.

Reads `gpg_v6_<tag>/` directories produced by `cross_judge.sh`, joins
each with the same Bagel MSE side, and reports:

    - per-judge r, R² (linear fit raw GPG)
    - per-judge saturation fit (floor, amp, k) on raw GPG
    - per-judge l10_max sanity check
    - paper-appendix table + multi-panel figure

Usage:
    # default: scan all gpg_v6_* dirs in DEFAULT_GPG_PARENT
    python -m detailness.gpg.judge_robustness

    # explicit list of (judge_tag, gpg_dir) pairs
    python -m detailness.gpg.judge_robustness \\
        --judges qwen3_vl:/tmp/.../gpg_v6_dropmore \\
                 qwen2_5_vl:/tmp/.../gpg_v6_qwen2_5_vl \\
                 internvl3:/tmp/.../gpg_v6_internvl3
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import (
    DEFAULT_WANDB_DIRS,
    aggregate_gpg,
    build_16_points,
    load_all_loss_runs,
    load_gpg_records,
    smoothed_mse,
    unified_budget,
)


DEFAULT_GPG_PARENT = "/tmp/detailness_real_n100"

# Cosmetic labels for the appendix table — keep in sync with cross_judge.sh
JUDGE_LABELS = {
    "qwen2_vl":         "Qwen2-VL-7B",
    "qwen2_5_vl":       "Qwen2.5-VL-7B",
    "qwen3_vl":         "Qwen3-VL-8B ★",
    "dropmore":         "Qwen3-VL-8B ★",  # legacy tag for paper v6
    "qwen3_5_moe_35b":  "Qwen3.5-35B-A3B (MoE)",
    "qwen3_5_moe_122b": "Qwen3.5-122B-A10B (MoE)",
    "internvl3":        "InternVL3-8B",
}

# Heuristic order for table rows / figure panels
JUDGE_ORDER = ["qwen2_vl", "qwen2_5_vl", "qwen3_vl", "dropmore",
               "qwen3_5_moe_35b", "qwen3_5_moe_122b", "internvl3"]

CAT_STYLE = {
    "struct":  ("#1f77b4", "o"),
    "dense":   ("#d62728", "s"),
    "spatial": ("#ff7f0e", "D"),
    "abl":     ("#2ca02c", "^"),
}


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _linfit(x: np.ndarray, y: np.ndarray):
    n = len(x); mx = x.mean(); my = y.mean()
    sxx = ((x - mx) ** 2).sum()
    sxy = ((x - mx) * (y - my)).sum()
    b = sxy / sxx
    a = my - b * mx
    yhat = a + b * x
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - my) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    sx = np.sqrt(((x - mx) ** 2).sum() / n)
    sy = np.sqrt(((y - my) ** 2).sum() / n)
    r = sxy / (n * sx * sy) if sx * sy else 0.0
    return float(a), float(b), float(r), float(r2)


def _fit_saturation(x: np.ndarray, y: np.ndarray):
    from scipy.optimize import curve_fit
    p0 = [y.min(), y.max() - y.min(), 0.01]
    try:
        popt, _ = curve_fit(
            lambda x_, f, a, k: f + a * np.exp(-k * x_),
            x, y, p0=p0, maxfev=20000,
        )
    except Exception:
        return None
    floor, amp, k = popt
    yhat = floor + amp * np.exp(-k * x)
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(floor), float(amp), float(k), float(r2)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_judges(parent: str) -> List[Tuple[str, str]]:
    """Find all gpg_v6_<tag>/ dirs under parent. Returns [(tag, path)]."""
    dirs = sorted(glob.glob(os.path.join(parent, "gpg_v6_*")))
    out: List[Tuple[str, str]] = []
    for d in dirs:
        # Skip if no shard files
        if not glob.glob(os.path.join(d, "shard*.jsonl")):
            continue
        tag = os.path.basename(d).removeprefix("gpg_v6_")
        out.append((tag, d))
    return out


def order_judges(judges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Sort by JUDGE_ORDER, then alphabetical for unknowns."""
    rank = {t: i for i, t in enumerate(JUDGE_ORDER)}
    return sorted(judges, key=lambda j: (rank.get(j[0], 99), j[0]))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_judge(gpg_dir: str,
                    mse_at: Dict[Tuple[str, str], float]):
    """Returns (pts, n_records). pts is None if no data."""
    recs = load_gpg_records(gpg_dir)
    if not recs:
        return None, 0
    by_cell = aggregate_gpg(recs)
    pts = build_16_points(by_cell, mse_at)
    return pts, len(recs)


def per_judge_stats(pts) -> Dict[str, float]:
    if pts is None or len(pts) < 6:
        return {}
    GPG = np.array([p[0] for p in pts])
    MSE = np.array([p[2] for p in pts])
    a, b, r, r2 = _linfit(GPG, MSE)
    sat = _fit_saturation(GPG, MSE)
    sat_floor, sat_r2 = (sat[0], sat[3]) if sat else (None, None)
    # l10 max?
    gpg_l10 = next((p[0] for p in pts if p[3] == "st_l10"), None)
    gpg_max = float(GPG.max())
    l10_max = (gpg_l10 is not None
               and abs(gpg_l10 - gpg_max) < 0.5)
    # Bbox-direction sanity
    g_full = next((p[0] for p in pts if p[3] == "full"), None)
    g_nob  = next((p[0] for p in pts if p[3] == "no_bbox"), None)
    bbox_dir = (g_nob - g_full) if (g_nob is not None and g_full is not None) else None
    return {
        "n_pts": len(pts),
        "GPG_min": float(GPG.min()),
        "GPG_max": gpg_max,
        "GPG_l10": gpg_l10 if gpg_l10 is not None else float("nan"),
        "MSE_min": float(MSE.min()),
        "MSE_max": float(MSE.max()),
        "lin_a": a, "lin_b": b, "r": r, "R2": r2,
        "sat_floor": sat_floor, "sat_R2": sat_r2,
        "l10_max": l10_max,
        "bbox_dir": bbox_dir,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_one_judge(ax, pts, name: str, budget: float):
    GPG = np.array([p[0] for p in pts])
    MSE = np.array([p[2] for p in pts])
    a, b, r, r2 = _linfit(GPG, MSE)

    for cat, (color, marker) in CAT_STYLE.items():
        sub = [p for p in pts if p[4] == cat]
        if not sub:
            continue
        ax.errorbar([p[0] for p in sub], [p[2] for p in sub],
                    xerr=[p[1] for p in sub],
                    fmt=marker, color=color, markersize=8, capsize=2,
                    label=f"{cat} (n={len(sub)})")
    xs = np.linspace(GPG.min() - 3, GPG.max() + 3, 100)
    ax.plot(xs, a + b * xs, "--", color="gray", linewidth=1.1,
            label=f"r={r:+.3f}, R²={r2:.3f}")
    sat = _fit_saturation(GPG, MSE)
    if sat:
        f_, am, kk, sr2 = sat
        ax.plot(xs, f_ + am * np.exp(-kk * xs), "-", color="black",
                linewidth=1.3, label=f"sat: floor={f_:.4f}")
    ax.set_xlabel("GPG (nats/caption)", fontsize=9)
    ax.set_ylabel("Bagel MSE", fontsize=9)
    ax.set_title(name, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=7)


def grid_figure(judges_with_pts, budget: float, out: str):
    n = len(judges_with_pts)
    cols = 3 if n > 3 else n
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.5 * rows),
                             squeeze=False)
    axes = axes.flatten()
    for ax, (tag, pts) in zip(axes, judges_with_pts):
        label = JUDGE_LABELS.get(tag, tag)
        plot_one_judge(ax, pts, label, budget)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"GPG cross-judge robustness (Bagel MSE budget = {budget:.2e})",
                 fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default=DEFAULT_GPG_PARENT,
                    help="parent dir containing gpg_v6_<tag>/ subdirs")
    ap.add_argument("--judges", nargs="*", default=None,
                    help="explicit list 'tag:/path/to/gpg_dir' overriding discovery")
    ap.add_argument("--fig-dir",
                    default=os.environ.get("FIGURES_DIR", "./figures"))
    ap.add_argument("--csv", default=None,
                    help="optional path to write the appendix table as CSV")
    ap.add_argument("--struct-dir",  default=DEFAULT_WANDB_DIRS["struct"])
    ap.add_argument("--dense-dir",   default=DEFAULT_WANDB_DIRS["dense"])
    ap.add_argument("--abl-dir",     default=DEFAULT_WANDB_DIRS["abl"])
    ap.add_argument("--spatial-dir", default=DEFAULT_WANDB_DIRS["spatial"])
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    wandb_dirs = {"struct": args.struct_dir, "dense": args.dense_dir,
                  "abl": args.abl_dir, "spatial": args.spatial_dir}
    runs = load_all_loss_runs(wandb_dirs)
    budget = unified_budget(runs)
    mse_at = {k: smoothed_mse(t, budget) for k, t in runs.items()}
    print(f"unified MSE budget = {budget:.4e}")

    if args.judges:
        judges = []
        for j in args.judges:
            if ":" in j:
                tag, path = j.split(":", 1)
            else:
                tag, path = os.path.basename(j).removeprefix("gpg_v6_"), j
            judges.append((tag, path))
    else:
        judges = discover_judges(args.parent)
        if not judges:
            print(f"[err] no gpg_v6_* dirs found under {args.parent}")
            return
    judges = order_judges(judges)

    judges_with_pts: List[Tuple[str, list]] = []
    rows = []
    for tag, path in judges:
        pts, n_recs = aggregate_judge(path, mse_at)
        stats = per_judge_stats(pts)
        if not stats:
            print(f"[skip] {tag}: pts={pts}, recs={n_recs}")
            continue
        judges_with_pts.append((tag, pts))
        rows.append({"tag": tag, "label": JUDGE_LABELS.get(tag, tag),
                     "path": path, "n_recs": n_recs, **stats})

    # Print appendix table
    print(f"\n{'judge':28s} {'n_pts':>5s} {'recs':>6s} {'r':>9s} {'R²':>7s} "
          f"{'sat_floor':>10s} {'sat_R²':>7s} {'l10max':>7s} {'no_bbox Δ':>11s}")
    print("-" * 100)
    for r in rows:
        l10 = "✓" if r["l10_max"] else "✗"
        bdir = (f"{r['bbox_dir']:+.1f}" if r["bbox_dir"] is not None else "—")
        floor = f"{r['sat_floor']:.4f}" if r["sat_floor"] is not None else "—"
        sR2 = f"{r['sat_R2']:.4f}" if r["sat_R2"] is not None else "—"
        print(f"{r['label']:28s} {r['n_pts']:>5d} {r['n_recs']:>6d} "
              f"{r['r']:>+9.4f} {r['R2']:>7.4f} {floor:>10s} {sR2:>7s} "
              f"{l10:>7s} {bdir:>11s}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            fieldnames = ["tag", "label", "n_pts", "n_recs", "GPG_min", "GPG_max",
                          "GPG_l10", "MSE_min", "MSE_max", "lin_a", "lin_b", "r",
                          "R2", "sat_floor", "sat_R2", "l10_max", "bbox_dir", "path"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fieldnames})
        print(f"\nSaved table → {args.csv}")

    out = os.path.join(args.fig_dir, "gpg_judge_robustness_grid.png")
    grid_figure(judges_with_pts, budget, out)


if __name__ == "__main__":
    main()
