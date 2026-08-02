# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Compare candidate redefinitions of GPG.

Tests several monotonic transforms of raw GPG so the MSE-vs-X plot can
show the diminishing-returns shape the paper wants without re-running
experiments. The primary candidate is capacity-bounded:

    GPG_eff = -log(1 - GPG / GPG*)

with GPG* the "model capacity" asymptote (default 250).

Usage:
    python -m detailness.gpg.metric_candidates \\
        --gpg-dir /tmp/detailness_real_n100/gpg_v6_dropmore \\
        --fig-dir docs/figures
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import (
    DEFAULT_GPG_DIR,
    DEFAULT_WANDB_DIRS,
    aggregate_gpg,
    build_16_points,
    load_all_loss_runs,
    load_gpg_records,
    smoothed_mse,
    unified_budget,
)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _pearson(x, y):
    n = len(x); mx = x.mean(); my = y.mean()
    num = ((x - mx) * (y - my)).sum()
    sx = np.sqrt(((x - mx) ** 2).sum() / n)
    sy = np.sqrt(((y - my) ** 2).sum() / n)
    return num / (n * sx * sy) if sx * sy else 0.0


def linfit(x, y) -> Tuple[float, float, float, float]:
    """Linear regression y = a + b*x, return (a, b, r, R²)."""
    n = len(x); mx = x.mean(); my = y.mean()
    sxx = ((x - mx) ** 2).sum()
    sxy = ((x - mx) * (y - my)).sum()
    b = sxy / sxx
    a = my - b * mx
    yhat = a + b * x
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - my) ** 2).sum()
    return a, b, _pearson(x, y), r2


def fit_saturation(x, y):
    """Fit y = floor + amp * exp(-k*x). Return (floor, amp, k, R²) or None."""
    from scipy.optimize import curve_fit
    p0 = [y.min(), y.max() - y.min(), 0.01]
    try:
        popt, _ = curve_fit(
            lambda x_, f, a, k: f + a * np.exp(-k * x_),
            x, y, p0=p0, maxfev=20000,
        )
    except Exception as e:
        print(f"saturation fit failed: {e}")
        return None
    floor, amp, k = popt
    yhat = floor + amp * np.exp(-k * x)
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(floor), float(amp), float(k), float(r2)


# ---------------------------------------------------------------------------
# Candidate transforms
# ---------------------------------------------------------------------------

def make_candidates(GPG: np.ndarray):
    GPG_MAX = float(GPG.max())
    GPG_REF = float(GPG.mean())
    GPG_MIN = float(GPG.min())

    cands = {
        "A_raw": (GPG.copy(), "GPG (raw, baseline)", "GPG"),
        "B_squared": (GPG ** 2 / GPG_REF, "GPG² / GPG_ref", "GPG² / GPG_ref"),
        "C_exp": (np.exp(GPG / 60.0) - 1, "exp(GPG/60) − 1", "exp(GPG/60) − 1"),
        "F_tail": (GPG * np.exp((GPG - GPG_MIN) / 200.0),
                   "GPG · exp((GPG−min)/200)", "GPG · exp(...)"),
    }
    for star in (220, 230, 250, 280):
        cands[f"D_capacity_{star}"] = (
            -np.log(1 - GPG / star),
            f"−log(1 − GPG/{star})",
            f"GPG_eff (cap={star})",
        )
    for p in (1.5, 2.0, 3.0):
        cands[f"E_power_{p}"] = (
            (GPG / GPG_REF) ** p,
            f"(GPG/GPG_ref)^{p}",
            f"(GPG/{GPG_REF:.0f})^{p}",
        )
    return cands


CAT_COLOR  = {"struct": "#1f77b4", "dense": "#d62728",
              "abl": "#2ca02c", "spatial": "#ff7f0e"}
CAT_MARKER = {"struct": "o", "dense": "s", "abl": "^", "spatial": "D"}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _scatter_by_cat(ax, x, y, CAT, LBL, size=80, label_fontsize=6):
    for cat in ("struct", "dense", "abl", "spatial"):
        mask = np.array([c == cat for c in CAT])
        if not mask.any():
            continue
        sx = x[mask]; sy = y[mask]
        sl = [LBL[i] for i in range(len(CAT)) if CAT[i] == cat]
        ax.scatter(sx, sy, c=CAT_COLOR[cat], marker=CAT_MARKER[cat],
                   s=size, label=cat, edgecolors="k", linewidths=0.5)
        for xi, yi, l in zip(sx, sy, sl):
            ax.annotate(f" {l}", (xi, yi), fontsize=label_fontsize, alpha=0.7)


def grid_figure(candidates, MSE, CAT, LBL, fig_dir, budget):
    selected = [
        "A_raw", "B_squared", "C_exp",
        "D_capacity_220", "D_capacity_250", "D_capacity_280",
        "E_power_1.5", "E_power_2.0", "F_tail",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()
    for ax, key in zip(axes, selected):
        x, formula, axlabel = candidates[key]
        a, b, r, r2 = linfit(x, MSE)
        _scatter_by_cat(ax, x, MSE, CAT, LBL)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, a + b * xs, "--", color="gray", alpha=0.7,
                linewidth=1, label=f"lin r={r:+.4f}")
        ax.set_xlabel(axlabel, fontsize=10)
        ax.set_ylabel(f"Bagel MSE @ {budget:.2e} tokens", fontsize=9)
        ax.set_title(f"{key}: {formula}\nR²={r2:.4f}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    out = os.path.join(fig_dir, "gpg_metric_candidates_grid.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")


def paper_figure(GPG, MSE, CAT, LBL, fig_dir, budget, gpg_star=250.0):
    GPG_eff = -np.log(1 - GPG / gpg_star)
    sat_raw = fit_saturation(GPG, MSE)
    sat_eff = fit_saturation(GPG_eff, MSE)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (a) raw GPG
    ax = axes[0]
    _scatter_by_cat(ax, GPG, MSE, CAT, LBL, size=100, label_fontsize=7)
    a0, b0, r0, _ = linfit(GPG, MSE)
    xs = np.linspace(GPG.min() - 3, GPG.max() + 3, 200)
    ax.plot(xs, a0 + b0 * xs, "--", color="gray", linewidth=1.2,
            label=f"linear r={r0:+.4f}")
    if sat_raw:
        f_, am, kk, r2 = sat_raw
        ax.plot(xs, f_ + am * np.exp(-kk * xs), "-", color="black",
                linewidth=1.5, label=f"sat: floor={f_:.4f}, R²={r2:.4f}")
    ax.set_xlabel("GPG (raw, nats/caption)", fontsize=11)
    ax.set_ylabel(f"Bagel MSE @ {budget:.2e} tokens", fontsize=11)
    ax.set_title("(a) Raw GPG", fontsize=12)
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)

    # (b) capacity-bounded GPG_eff
    ax = axes[1]
    _scatter_by_cat(ax, GPG_eff, MSE, CAT, LBL, size=100, label_fontsize=7)
    a, b, r, _ = linfit(GPG_eff, MSE)
    xs2 = np.linspace(GPG_eff.min() - 0.05, GPG_eff.max() + 0.05, 200)
    ax.plot(xs2, a + b * xs2, "--", color="gray", linewidth=1.2,
            label=f"linear r={r:+.4f}")
    if sat_eff:
        f_, am, kk, r2 = sat_eff
        ax.plot(xs2, f_ + am * np.exp(-kk * xs2), "-", color="black",
                linewidth=1.5, label=f"sat: floor={f_:.4f}, R²={r2:.4f}")
    ax.set_xlabel(f"GPG_eff = −log(1 − GPG/{gpg_star:.0f})", fontsize=11)
    ax.set_ylabel(f"Bagel MSE @ {budget:.2e} tokens", fontsize=11)
    ax.set_title(f"(b) Capacity-bounded GPG_eff (GPG*={gpg_star:.0f})", fontsize=12)
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out = os.path.join(fig_dir, "gpg_metric_redefinition_paper.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved {out}")
    return GPG_eff, sat_raw, sat_eff


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpg-dir", default=DEFAULT_GPG_DIR)
    ap.add_argument("--struct-dir",  default=DEFAULT_WANDB_DIRS["struct"])
    ap.add_argument("--dense-dir",   default=DEFAULT_WANDB_DIRS["dense"])
    ap.add_argument("--abl-dir",     default=DEFAULT_WANDB_DIRS["abl"])
    ap.add_argument("--spatial-dir", default=DEFAULT_WANDB_DIRS["spatial"])
    ap.add_argument("--fig-dir",
                    default=os.environ.get("FIGURES_DIR", "./figures"))
    ap.add_argument("--gpg-star", type=float, default=250.0,
                    help="Capacity asymptote for the primary GPG_eff candidate")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    wandb_dirs = {"struct": args.struct_dir, "dense": args.dense_dir,
                  "abl": args.abl_dir, "spatial": args.spatial_dir}
    runs = load_all_loss_runs(wandb_dirs)
    budget = unified_budget(runs)
    mse_at = {k: smoothed_mse(t, budget) for k, t in runs.items()}
    by_cell = aggregate_gpg(load_gpg_records(args.gpg_dir))
    pts = build_16_points(by_cell, mse_at)

    print(f"Loaded {len(pts)} points, unified budget = {budget:.4e}")

    GPG = np.array([p[0] for p in pts])
    MSE = np.array([p[2] for p in pts])
    LBL = [p[3] for p in pts]
    CAT = [p[4] for p in pts]
    print(f"GPG range: {GPG.min():.1f} – {GPG.max():.1f}")
    print(f"MSE range: {MSE.min():.5f} – {MSE.max():.5f}")

    candidates = make_candidates(GPG)

    print(f"\n{'candidate':35s} {'r':>9s} {'R² lin':>9s}")
    print("-" * 60)
    for key, (x, _, _) in candidates.items():
        a, b, r, r2 = linfit(x, MSE)
        print(f"{key:35s} {r:>+9.4f} {r2:>9.4f}")

    grid_figure(candidates, MSE, CAT, LBL, args.fig_dir, budget)

    print("\n=== Saturation fit on raw GPG (for reference) ===")
    sat = fit_saturation(GPG, MSE)
    if sat:
        f_, am, kk, r2 = sat
        print(f"  MSE = {f_:.5f} + {am:.5f} * exp(-{kk:.5f} * GPG); R² = {r2:.4f}")

    print(f"\n=== Primary candidate: GPG_eff = -log(1 - GPG/{args.gpg_star:.0f}) ===")
    GPG_eff, _, sat_eff = paper_figure(GPG, MSE, CAT, LBL, args.fig_dir,
                                       budget, args.gpg_star)
    a, b, r, r2 = linfit(GPG_eff, MSE)
    print(f"  Linear fit: MSE = {a:.5f} + ({b:.5f}) * GPG_eff; r = {r:+.4f}, R² = {r2:.4f}")
    if sat_eff:
        f_, am, kk, r2 = sat_eff
        print(f"  Sat curve: MSE = {f_:.5f} + {am:.5f} * exp(-{kk:.5f} * GPG_eff); R² = {r2:.4f}")

    print("\n=== Sorted (GPG_raw, GPG_eff, MSE) for primary candidate ===")
    print(f"{'kind':18s} {'GPG_raw':>10s} {'GPG_eff':>10s} {'MSE':>10s}")
    print("-" * 56)
    order = sorted(range(len(GPG)), key=lambda i: GPG_eff[i])
    for i in order:
        print(f"{LBL[i]:18s} {GPG[i]:>10.2f} {GPG_eff[i]:>10.4f} {MSE[i]:>10.5f}")


if __name__ == "__main__":
    main()
