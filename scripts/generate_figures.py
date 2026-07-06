"""Generate all publication figures from pre-computed outputs.

Reads CSVs from output/{kalshi,bayesian,cross_platform}/ and saves
PNG + PDF to output/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import COLORS, DOMAINS, OUTPUT_DIR
from src.plotting import (
    fig_hero_decomposition,
    fig_observed_vs_fitted,
    fig_slope_trajectories,
    fig_whale_effect,
)

KALSHI = OUTPUT_DIR / "kalshi"
REVISION = OUTPUT_DIR / "revision"
FIGURES = OUTPUT_DIR / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
# Manuscript root (two levels up) holds the \includegraphics targets.
MANUSCRIPT_DIR = Path(__file__).resolve().parent.parent.parent.parent


def fig_reliability_curves(out_stem):
    """Figure 8: nonparametric reliability curves by domain (binned observed
    frequency vs raw price, with the isotonic fit and the 45-degree line)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rel_path = REVISION / "binned_reliability_by_domain.csv"
    iso_path = REVISION / "isotonic_calibration_by_domain.csv"
    if not rel_path.exists():
        print("  SKIP figure8_reliability (run revision-diagnostics first)")
        return
    rel = pd.read_csv(rel_path)
    iso = pd.read_csv(iso_path) if iso_path.exists() else None

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7), sharex=True, sharey=True)
    for ax, dom in zip(axes.ravel(), DOMAINS):
        ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", zorder=1)
        r = rel[rel["domain"] == dom]
        if len(r):
            ax.plot(r["mean_price"] / 100.0, r["observed_rate"], "o-",
                    color=COLORS.get(dom, "C0"), ms=4, lw=1.5, zorder=3, label="binned")
        if iso is not None:
            g = iso[iso["domain"] == dom]
            if len(g):
                ax.plot(g["price"], g["isotonic_probability"], "-",
                        color=COLORS.get(dom, "C0"), alpha=0.5, lw=2.5, zorder=2, label="isotonic")
        ax.set_title(dom, fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
    for ax in axes[-1]:
        ax.set_xlabel("Raw market price")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed frequency")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=150, bbox_inches="tight")
        # also place the PDF where the manuscript includes it
        if ext == "pdf":
            fig.savefig(str(MANUSCRIPT_DIR / "figure8_reliability.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("  GENERATING PUBLICATION FIGURES")
    print("=" * 70)

    # Load decomposed calibration matrix
    cal_path = KALSHI / "calibration_matrix_decomposed.csv"
    if not cal_path.exists():
        cal_path = KALSHI / "calibration_matrix.csv"
    cal = pd.read_csv(cal_path)
    print(f"  Loaded {len(cal)} cells from {cal_path.name}")

    # Figure 1: Slope trajectories
    fig_slope_trajectories(cal, str(FIGURES / "figure1_slope_trajectories"))
    print("  saved figure1_slope_trajectories.{png,pdf}")

    # Figure 2: Hero decomposition (requires decomposition columns)
    if "mu" in cal.columns:
        fig_hero_decomposition(cal, str(FIGURES / "figure2_hero"))
        print("  saved figure2_hero.{png,pdf}")
    else:
        print("  SKIP figure2_hero (no decomposition columns — run run_kalshi.py first)")

    # Figure 3: Observed vs fitted
    if "fitted" in cal.columns:
        r2 = fig_observed_vs_fitted(cal, str(FIGURES / "figure3_observed_vs_fitted"))
        print(f"  saved figure3_observed_vs_fitted.{{png,pdf}}  R2={r2:.4f}")
    else:
        print("  SKIP figure3_observed_vs_fitted (no 'fitted' column)")

    # Figure 4: Whale effect
    fig_whale_effect(cal, str(FIGURES / "figure4_whale_effect"))
    print("  saved figure4_whale_effect.{png,pdf}")

    # Figure 8: nonparametric reliability curves by domain
    fig_reliability_curves(str(FIGURES / "figure8_reliability"))
    print("  saved figure8_reliability.{png,pdf} (+ manuscript root)")

    # Cross-platform figures (if available)
    cp_dir = OUTPUT_DIR / "cross_platform"
    if (cp_dir / "polymarket_slopes_by_domain_time.csv").exists():
        from src.plotting import (
            fig_cross_platform_trajectories,
            fig_politics_comparison,
            fig_scale_effect_comparison,
        )

        pm_dt = pd.read_csv(cp_dir / "polymarket_slopes_by_domain_time.csv")
        kalshi_dt = pd.read_csv(KALSHI / "calibration_slopes_by_domain_time.csv")
        fig_cross_platform_trajectories(pm_dt, kalshi_dt, str(FIGURES / "figure_cp1_slope_trajectories"))
        print("  saved figure_cp1_slope_trajectories.{png,pdf}")

        fig_politics_comparison(kalshi_dt, pm_dt, str(FIGURES / "figure_cp2_politics_comparison"))
        print("  saved figure_cp2_politics_comparison.{png,pdf}")

        if (cp_dir / "polymarket_slopes_by_domain_size.csv").exists():
            pm_ds = pd.read_csv(cp_dir / "polymarket_slopes_by_domain_size.csv")
            kalshi_ds = pd.read_csv(KALSHI / "calibration_slopes_by_domain_size.csv")
            fig_scale_effect_comparison(kalshi_ds, pm_ds, str(FIGURES / "figure_cp3_scale_effect"))
            print("  saved figure_cp3_scale_effect.{png,pdf}")
    else:
        print("  SKIP cross-platform figures (run run_cross_platform.py first)")

    print("\n" + "=" * 70)
    print(f"  DONE — figures in {FIGURES}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
