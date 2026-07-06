"""Matplotlib setup and all figure-generation functions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BIN_LABELS, COLORS, DOMAINS, SIZE_LABELS


def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    return plt


def save_figure(fig, path, plt):
    """Save figure as PNG (300 DPI) and PDF, then close."""
    for ext in ["png", "pdf"]:
        fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Main Kalshi figures (from final_pipeline.py Steps 4 & 6)
# ═══════════════════════════════════════════════════════════════════

def fig_slope_trajectories(cal, out_path):
    """Figure 1: Calibration slope trajectories by domain and time horizon."""
    plt = setup_matplotlib()
    x = np.arange(len(BIN_LABELS))

    fig, ax = plt.subplots(figsize=(8, 5))
    for d in DOMAINS:
        means = []
        for tl in BIN_LABELS:
            sub = cal[(cal["domain"] == d) & (cal["time_bin"] == tl)]
            means.append(sub["slope_b"].mean() if len(sub) > 0 else np.nan)
        ax.plot(x, means, "o-", color=COLORS[d], label=d, lw=2, ms=5)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Calibration slope $b$")
    ax.set_xlabel("Time to close")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    save_figure(fig, out_path, plt)


def fig_hero_decomposition(cal, out_path):
    """Figure 2: 2x2 decomposition (mu, alpha, beta, gamma)."""
    plt = setup_matplotlib()
    x = np.arange(len(BIN_LABELS))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) mu(tau)
    ax = axes[0, 0]
    mu_vals = [cal[cal["time_bin"] == tl]["mu"].iloc[0] if len(cal[cal["time_bin"] == tl]) > 0
               else np.nan for tl in BIN_LABELS]
    ax.plot(x, mu_vals, "ko-", lw=2, ms=6)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("$\\mu(\\tau)$")
    ax.set_title("(a) Universal horizon $\\mu(\\tau)$", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.15)

    # (b) alpha_d
    ax = axes[0, 1]
    alpha_vals = {d: cal[cal["domain"] == d]["alpha"].iloc[0] for d in DOMAINS if len(cal[cal["domain"] == d]) > 0}
    sorted_d = sorted(alpha_vals, key=alpha_vals.get)
    y_pos = np.arange(len(sorted_d))
    ax.barh(y_pos, [alpha_vals[d] for d in sorted_d],
            color=[COLORS[d] for d in sorted_d], height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_d)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("$\\alpha_d$")
    ax.set_title("(b) Domain intercepts $\\alpha_d$", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.15, axis="x")

    # (c) beta_d(tau)
    ax = axes[1, 0]
    for d in DOMAINS:
        beta_vals = []
        for tl in BIN_LABELS:
            sub = cal[(cal["domain"] == d) & (cal["time_bin"] == tl)]
            beta_vals.append(sub["beta"].iloc[0] if len(sub) > 0 else np.nan)
        ax.plot(x, beta_vals, "o-", color=COLORS[d], label=d, lw=1.5, ms=4)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("$\\beta_d(\\tau)$")
    ax.set_title("(c) Domain deviations $\\beta_d(\\tau)$", fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.15)

    # (d) gamma_d(s)
    ax = axes[1, 1]
    xs = np.arange(len(SIZE_LABELS))
    for d in DOMAINS:
        gamma_vals = []
        for sl in SIZE_LABELS:
            sub = cal[(cal["domain"] == d) & (cal["size_bin"] == sl)]
            gamma_vals.append(sub["gamma"].iloc[0] if len(sub) > 0 else np.nan)
        ax.plot(xs, gamma_vals, "o-", color=COLORS[d], label=d, lw=1.5, ms=5)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(SIZE_LABELS)
    ax.set_ylabel("$\\gamma_d(s)$")
    ax.set_title("(d) Scale effects $\\gamma_d(s)$", fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    save_figure(fig, out_path, plt)


def fig_observed_vs_fitted(cal, out_path):
    """Figure 3: Observed vs fitted calibration slopes."""
    plt = setup_matplotlib()

    fig, ax = plt.subplots(figsize=(6, 6))
    for d in DOMAINS:
        sub = cal[cal["domain"] == d]
        ax.scatter(sub["fitted"], sub["slope_b"], color=COLORS[d], label=d,
                   s=25, alpha=0.7, zorder=3)
    lims = [cal[["fitted", "slope_b"]].min().min() - 0.05,
            cal[["fitted", "slope_b"]].max().max() + 0.05]
    ax.plot(lims, lims, "k--", lw=0.8, zorder=0)
    ss_tot = np.sum((cal["slope_b"] - cal["slope_b"].mean()) ** 2)
    ss_res = np.sum((cal["slope_b"] - cal["fitted"]) ** 2)
    r2 = 1 - ss_res / ss_tot
    ax.annotate(f"$R^2 = {r2:.3f}$", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=12)
    ax.set_xlabel("Fitted $\\theta$")
    ax.set_ylabel("Observed $\\theta$")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.15)
    ax.set_aspect("equal")
    fig.tight_layout()
    save_figure(fig, out_path, plt)
    return r2


def fig_whale_effect(cal, out_path):
    """Figure 4: Scale effect bar charts for Politics and Sports."""
    plt = setup_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for idx, d in enumerate(["Politics", "Sports"]):
        ax = axes[idx]
        means, errs = [], []
        for sl in SIZE_LABELS:
            sub = cal[(cal["domain"] == d) & (cal["size_bin"] == sl)]
            if len(sub) > 0:
                means.append(sub["slope_b"].mean())
                errs.append(sub["slope_b"].std() / np.sqrt(max(len(sub), 1)))
            else:
                means.append(np.nan)
                errs.append(0)
        ax.bar(np.arange(len(SIZE_LABELS)), means, yerr=errs,
               color=COLORS[d], alpha=0.8, capsize=4)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_xticks(np.arange(len(SIZE_LABELS)))
        ax.set_xticklabels(SIZE_LABELS, fontsize=9)
        ax.set_title(d, fontweight="bold", color=COLORS[d])
        ax.set_ylabel("Mean slope $b$" if idx == 0 else "")
        ax.grid(True, alpha=0.15, axis="y")
    fig.suptitle("Scale effect: calibration slope by trade size", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_path, plt)


def fig_posterior_predictive(cal, samples, out_path):
    """Figure 5: Posterior predictive check with 95% intervals."""
    plt = setup_matplotlib()
    from src.config import SIZE_LOG_MEDIANS

    d2i = {d: i for i, d in enumerate(DOMAINS)}
    t2i = {l: i for i, l in enumerate(BIN_LABELS)}

    mu_post = np.array(samples["mu"])
    alpha_post = np.array(samples["alpha"])
    beta_post = np.array(samples["beta"])
    delta_post = np.array(samples["delta"])
    sigma_post = np.array(samples["sigma"])

    raw_log_s = np.array([SIZE_LOG_MEDIANS[s] for s in cal["size_bin"]])
    log_s_c = raw_log_s - raw_log_s.mean()

    rng = np.random.default_rng(123)
    ppc_rows = []
    for idx, row in cal.iterrows():
        di = d2i[row["domain"]]
        ti = t2i[row["time_bin"]]
        ls = log_s_c[idx]

        theta_pred = mu_post[:, ti] + alpha_post[:, di] + beta_post[:, di, ti] + delta_post[:, di] * ls
        theta_rep = theta_pred + rng.normal(0, sigma_post)

        obs = row["slope_b"]
        pm = float(np.mean(theta_pred))
        lo = float(np.percentile(theta_rep, 2.5))
        hi = float(np.percentile(theta_rep, 97.5))
        within = 1 if lo <= obs <= hi else 0

        ppc_rows.append(dict(
            domain=row["domain"], time_bin=row["time_bin"], size_bin=row["size_bin"],
            observed=round(obs, 4), post_mean=round(pm, 4),
            post_2_5=round(lo, 4), post_97_5=round(hi, 4), within_95=within,
        ))

    ppc = pd.DataFrame(ppc_rows)

    fig, ax = plt.subplots(figsize=(8, 6))
    ppc_sorted = ppc.sort_values("post_mean").reset_index(drop=True)
    for i, row in ppc_sorted.iterrows():
        col = COLORS.get(row["domain"], "#999")
        w = 1 if row["within_95"] else 0
        ax.plot([row["post_2_5"], row["post_97_5"]], [i, i],
                color=col, alpha=0.3 + 0.4 * w, lw=1)
        ax.plot(row["observed"], i, "o", color=col, ms=2, alpha=0.8)
    ax.set_xlabel("Calibration slope")
    ax.set_ylabel("Cell (sorted by posterior mean)")
    ax.set_title("Posterior predictive check", fontweight="bold")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COLORS[d], lw=2, label=d) for d in DOMAINS]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    save_figure(fig, out_path, plt)

    return ppc


# ═══════════════════════════════════════════════════════════════════
# Bayesian figures (from fit_calibration_model.py)
# ═══════════════════════════════════════════════════════════════════

def fig_forest_plot(trace, cat_names, out_path, label="M1"):
    """Forest plot of category-level alpha and beta with 90% HDI."""
    plt = setup_matplotlib()
    import arviz as az

    GROUP_COLORS = {
        "Sports": "#1f77b4", "Politics": "#d62728", "Crypto": "#ff7f0e",
        "Finance": "#2ca02c", "Science/Tech": "#9467bd", "Weather": "#17becf",
        "Entertainment": "#e377c2", "Media": "#bcbd22", "World Events": "#8c564b",
        "Esports": "#7f7f7f", "Other": "#aaaaaa", "Polymarket": "#D65F5F",
    }

    J = len(cat_names)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, J * 0.5)))

    for ax_idx, (param, ref_val, param_label) in enumerate([
        ("alpha_j", 0, r"$\alpha_j$ (Bias)"),
        ("beta_j", 1, r"$\beta_j$ (Slope)"),
    ]):
        ax = axes[ax_idx]
        draws = trace.posterior[param].values.reshape(-1, J)
        means = draws.mean(axis=0)
        hdis = np.array([az.hdi(draws[:, j], prob=0.90) for j in range(J)])
        y_pos = np.arange(J)
        colors = [GROUP_COLORS.get(cat_names[j], "#999") for j in range(J)]

        for j in range(J):
            ax.plot(hdis[j], [y_pos[j], y_pos[j]], color=colors[j], linewidth=2, solid_capstyle="round")
            ax.plot(means[j], y_pos[j], "o", color=colors[j], markersize=8, zorder=5)

        ax.axvline(ref_val, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
                   label=f"Perfect calibration = {ref_val}")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cat_names, fontsize=9)
        ax.set_xlabel(param_label, fontsize=12)
        ax.set_title(f"Category-Level {param_label} ({label})", fontsize=13)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()

    fig.tight_layout()
    save_figure(fig, out_path, plt)


def fig_calibration_by_category(trace, data, cat_names, out_path, label="M1"):
    """Grid of calibration curves by category with posterior bands."""
    plt = setup_matplotlib()

    GROUP_COLORS = {
        "Sports": "#1f77b4", "Politics": "#d62728", "Crypto": "#ff7f0e",
        "Finance": "#2ca02c", "Science/Tech": "#9467bd", "Weather": "#17becf",
        "Entertainment": "#e377c2", "Media": "#bcbd22", "World Events": "#8c564b",
        "Esports": "#7f7f7f", "Other": "#aaaaaa", "Polymarket": "#D65F5F",
    }

    J = len(cat_names)
    ncols = min(4, J)
    nrows = (J + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)

    alpha_draws = trace.posterior["alpha_j"].values.reshape(-1, J)
    beta_draws = trace.posterior["beta_j"].values.reshape(-1, J)
    p_grid = np.linspace(0.02, 0.98, 200)
    logit_grid = np.log(p_grid / (1 - p_grid))
    df = data["df"]

    for j in range(J):
        row_idx, col_idx = divmod(j, ncols)
        ax = axes[row_idx][col_idx]
        cat = cat_names[j]
        color = GROUP_COLORS.get(cat, "#999")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)

        n_draws = min(200, alpha_draws.shape[0])
        rng = np.random.RandomState(42)
        idx = rng.choice(alpha_draws.shape[0], size=n_draws, replace=False)

        curves = np.zeros((n_draws, len(p_grid)))
        for i, k in enumerate(idx):
            eta = alpha_draws[k, j] + beta_draws[k, j] * logit_grid
            curves[i] = 1 / (1 + np.exp(-eta))

        lo = np.percentile(curves, 5, axis=0)
        hi = np.percentile(curves, 95, axis=0)
        mean_curve = curves.mean(axis=0)

        ax.fill_between(p_grid, lo, hi, alpha=0.2, color=color)
        ax.plot(p_grid, mean_curve, color=color, lw=2)

        cat_df = df[df["category"] == cat]
        if len(cat_df) >= 10:
            n_bins = min(10, max(5, len(cat_df) // 50))
            cat_df = cat_df.copy()
            cat_df["bin"] = pd.cut(cat_df["implied_prob"], bins=np.linspace(0, 1, n_bins + 1))
            binned = cat_df.groupby("bin", observed=True).agg(
                pred=("implied_prob", "mean"), obs=("outcome", "mean"), n=("outcome", "count")
            ).dropna()
            if len(binned) > 0:
                sizes = binned["n"] / binned["n"].max() * 100 + 20
                ax.scatter(binned["pred"], binned["obs"], s=sizes, color=color,
                           edgecolors="white", lw=0.5, zorder=5, alpha=0.8)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"{cat} (n={len(cat_df):,})", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)

    for j in range(J, nrows * ncols):
        row_idx, col_idx = divmod(j, ncols)
        axes[row_idx][col_idx].set_visible(False)

    fig.supxlabel("Predicted Probability", fontsize=12)
    fig.supylabel("Observed Frequency", fontsize=12)
    fig.suptitle(f"Calibration Curves by Category ({label})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    save_figure(fig, out_path, plt)


def fig_platform_effect(trace, out_path):
    """Posterior of gamma (platform effect) from M2."""
    plt = setup_matplotlib()
    import arviz as az

    fig, ax = plt.subplots(figsize=(8, 5))
    draws = trace.posterior["gamma"].values.flatten()
    ax.hist(draws, bins=60, density=True, alpha=0.7, color="#D65F5F", edgecolor="white", lw=0.5)

    mean_val = draws.mean()
    hdi = az.hdi(draws, prob=0.90)
    ax.axvline(mean_val, color="black", lw=1.5, label=f"Posterior mean: {mean_val:.4f}")
    ax.axvline(0, color="gray", ls="--", lw=1.5, alpha=0.7, label="No platform effect")
    ax.axvspan(hdi[0], hdi[1], alpha=0.15, color="#D65F5F",
               label=f"90% HDI: [{hdi[0]:.4f}, {hdi[1]:.4f}]")

    prob_positive = (draws > 0).mean()
    ax.set_xlabel(r"$\gamma$ (Platform Effect: Polymarket vs Kalshi)", fontsize=12)
    ax.set_title(f"Platform Effect Posterior (M2) — P(gamma > 0) = {prob_positive:.1%}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_path, plt)


# ═══════════════════════════════════════════════════════════════════
# Cross-platform figures
# ═══════════════════════════════════════════════════════════════════

def fig_cross_platform_trajectories(pm_dt, kalshi_dt, out_path, domains=None):
    """Figure CP1: Side-by-side slope trajectories for Kalshi vs Polymarket."""
    plt = setup_matplotlib()
    if domains is None:
        from src.config import CROSS_PLATFORM_DOMAINS
        domains = CROSS_PLATFORM_DOMAINS

    x = np.arange(len(BIN_LABELS))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (dt, title) in zip(axes, [(kalshi_dt, "Kalshi"), (pm_dt, "Polymarket")]):
        for d in domains:
            means = []
            for tl in BIN_LABELS:
                sub = dt[(dt["domain"] == d) & (dt["time_bin"] == tl)]
                means.append(float(sub["slope_b"].iloc[0]) if len(sub) > 0 else np.nan)
            ax.plot(x, means, "o-", color=COLORS[d], label=d, lw=2, ms=5)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Calibration slope $b$")
        ax.set_xlabel("Time to close")
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.15)

    fig.suptitle("Cross-Platform Calibration Slope Trajectories", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_path, plt)


def fig_politics_comparison(kalshi_dt, pm_dt, out_path):
    """Figure CP2: Politics calibration slope by horizon (Kalshi vs Polymarket)."""
    plt = setup_matplotlib()

    kalshi_pol = kalshi_dt[kalshi_dt["domain"] == "Politics"].sort_values("time_bin_order")
    pm_pol = pm_dt[pm_dt["domain"] == "Politics"].sort_values("time_bin_order")

    common_bins = sorted(
        set(kalshi_pol["time_bin"]) & set(pm_pol["time_bin"]),
        key=lambda t: BIN_LABELS.index(t),
    )
    if len(common_bins) == 0:
        common_bins = BIN_LABELS

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_x = np.arange(len(common_bins))
    width = 0.35

    k_vals, p_vals = [], []
    for tl in common_bins:
        k_sub = kalshi_pol[kalshi_pol["time_bin"] == tl]
        p_sub = pm_pol[pm_pol["time_bin"] == tl]
        k_vals.append(float(k_sub["slope_b"].iloc[0]) if len(k_sub) > 0 else np.nan)
        p_vals.append(float(p_sub["slope_b"].iloc[0]) if len(p_sub) > 0 else np.nan)

    ax.bar(bar_x - width / 2, k_vals, width, label="Kalshi", color=COLORS["Politics"], alpha=0.7)
    ax.bar(bar_x + width / 2, p_vals, width, label="Polymarket", color=COLORS["Politics"], alpha=0.35,
           edgecolor=COLORS["Politics"], linewidth=1.5)
    ax.axhline(1.0, color="gray", ls="--", lw=1.0, zorder=0, label="Perfect calibration")
    ax.set_xticks(bar_x)
    ax.set_xticklabels(common_bins, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Calibration slope $b$")
    ax.set_xlabel("Time to close")
    ax.set_title("Politics: Calibration Slope by Horizon (Kalshi vs Polymarket)",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15, axis="y")
    fig.tight_layout()
    save_figure(fig, out_path, plt)


def fig_scale_effect_comparison(kalshi_ds, pm_ds, out_path):
    """Figure CP3: Scale effect comparison (Politics) for both platforms."""
    plt = setup_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    xs = np.arange(len(SIZE_LABELS))

    for ax, (ds, title) in zip(axes, [(kalshi_ds, "Kalshi"), (pm_ds, "Polymarket")]):
        pol = ds[ds["domain"] == "Politics"]
        if len(pol) == 0:
            ax.set_title(f"{title}: No Politics data")
            continue
        means = []
        for sl in SIZE_LABELS:
            sub = pol[pol["size_bin"] == sl]
            means.append(float(sub["slope_b"].iloc[0]) if len(sub) > 0 else np.nan)
        ax.bar(xs, means, color=COLORS["Politics"], alpha=0.8)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(SIZE_LABELS, fontsize=9)
        ax.set_title(f"{title} — Politics", fontweight="bold", color=COLORS["Politics"])
        ax.set_ylabel("Calibration slope $b$" if title == "Kalshi" else "")
        ax.grid(True, alpha=0.15, axis="y")

        large_v = pol[pol["size_bin"] == "Large"]["slope_b"].values
        single_v = pol[pol["size_bin"] == "Single"]["slope_b"].values
        if len(large_v) > 0 and len(single_v) > 0:
            delta = float(large_v[0]) - float(single_v[0])
            ax.annotate(f"$\\Delta$ = {delta:+.3f}", xy=(0.95, 0.95),
                        xycoords="axes fraction", ha="right", va="top", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    fig.suptitle("Scale Effect: Politics Slopes by Trade Size", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_path, plt)
