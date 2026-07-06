"""Main Kalshi analysis: base data -> calibration matrix -> decomposition -> summary.

Outputs CSVs to output/kalshi/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import f as fdist

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration import bootstrap_whale_effect, decompose, fit_logistic
from src.config import (
    BIN_LABELS,
    CELL_MIN,
    DOMAINS,
    OUTPUT_DIR,
    SIZE_LABELS,
)
from src.pipeline import (
    fit_calibration_matrix,
    fit_slopes_by_domain_size,
    fit_slopes_by_domain_time,
    load_kalshi_market_stats,
    load_kalshi_trades,
)

OUT = OUTPUT_DIR / "kalshi"
OUT.mkdir(parents=True, exist_ok=True)


def step1_base_data():
    """Load trades and fit calibration matrix."""
    print("\n" + "=" * 70)
    print("  STEP 1: BASE DATA — Full calibration matrix")
    print("=" * 70)

    conn = duckdb.connect()
    df = load_kalshi_trades(conn)
    conn.close()

    total_trades = int(df["n_trades"].sum())
    print(f"  Loaded {total_trades:,} trades")
    for d in DOMAINS:
        n = int(df[df["domain"] == d]["n_trades"].sum())
        print(f"    {d:>15s}: {n:>12,} trades")

    # Fit calibration matrix (domain x time x size)
    cal = fit_calibration_matrix(df)
    cal.to_csv(OUT / "calibration_matrix.csv", index=False)
    print(f"  {len(cal)} cells -> calibration_matrix.csv")

    # Domain x time
    dt = fit_slopes_by_domain_time(df)
    dt.to_csv(OUT / "calibration_slopes_by_domain_time.csv", index=False)
    print(f"  {len(dt)} cells -> calibration_slopes_by_domain_time.csv")

    # Domain x size
    ds = fit_slopes_by_domain_size(df)
    ds.to_csv(OUT / "calibration_slopes_by_domain_size.csv", index=False)
    print(f"  {len(ds)} cells -> calibration_slopes_by_domain_size.csv")

    return cal, df


def step2_summary(raw_df):
    """Table 1: summary statistics."""
    print("\n" + "=" * 70)
    print("  STEP 2: TABLE 1 — Summary statistics")
    print("=" * 70)

    conn = duckdb.connect()
    all_mkts, resolved = load_kalshi_market_stats(conn)
    conn.close()

    rows = []
    for d in DOMAINS:
        d_all = all_mkts[all_mkts["domain"] == d]
        d_res = resolved[resolved["domain"] == d]
        d_resolved_closed = d_all[(d_all["status"] == "finalized") & (d_all["result"].isin(["yes", "no"]))]
        n_all = len(d_all)
        pct = 100.0 * len(d_resolved_closed) / max(n_all, 1)
        rows.append(dict(
            domain=d,
            n_markets=len(d_res),
            n_trades=int(d_res["n_trades"].sum()),
            n_contracts=int(d_res["n_contracts"].sum()),
            pct_resolved=round(pct, 1),
            median_volume=int(d_res["n_trades"].median()) if len(d_res) > 0 else 0,
            mean_price=round(float(d_res["mean_price"].mean()), 1) if len(d_res) > 0 else 0,
            base_rate=round(100.0 * (d_res["result"] == "yes").mean(), 1) if len(d_res) > 0 else 0,
        ))

    # Totals row
    tot_all = all_mkts[all_mkts["domain"].isin(DOMAINS)]
    tot_resolved_closed = tot_all[(tot_all["status"] == "finalized") & (tot_all["result"].isin(["yes", "no"]))]
    tot_res = resolved[resolved["domain"].isin(DOMAINS)]
    rows.append(dict(
        domain="TOTAL",
        n_markets=len(tot_res),
        n_trades=int(tot_res["n_trades"].sum()),
        n_contracts=int(tot_res["n_contracts"].sum()),
        pct_resolved=round(100.0 * len(tot_resolved_closed) / max(len(tot_all), 1), 1),
        median_volume=int(tot_res["n_trades"].median()),
        mean_price=round(float(tot_res["mean_price"].mean()), 1),
        base_rate=round(100.0 * (tot_res["result"] == "yes").mean(), 1),
    ))

    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "table1_summary.csv", index=False)
    print("  saved table1_summary.csv")
    print(tbl.to_string(index=False))
    return tbl


def step3_decomposition(cal):
    """Variance decomposition + F-tests + whale effect bootstrap."""
    print("\n" + "=" * 70)
    print("  STEP 3: DECOMPOSITION — Variance analysis")
    print("=" * 70)

    cal = decompose(cal)
    theta = cal["slope_b"].values
    ss_total = np.sum((theta - theta.mean()) ** 2)

    components = ["mu", "alpha", "kappa", "beta", "gamma"]
    fitted_cumul = np.zeros(len(theta))
    vd_rows = []
    prev_r2 = 0.0
    for comp in components:
        fitted_cumul = fitted_cumul + cal[comp].values
        ss_fit = np.sum((fitted_cumul - theta.mean()) ** 2)
        cumul_r2 = ss_fit / ss_total if ss_total > 0 else 0
        marginal = cumul_r2 - prev_r2
        vd_rows.append(dict(component=comp, marginal_r2=round(marginal, 4),
                            cumulative_r2=round(cumul_r2, 4)))
        prev_r2 = cumul_r2
        print(f"  {comp:>10s}  marginal={marginal:.4f}  cumul={cumul_r2:.4f}")

    ss_resid = np.sum(cal["residual"].values ** 2)
    vd_rows.append(dict(component="residual", marginal_r2=round(1 - prev_r2, 4),
                        cumulative_r2=1.0))
    vd = pd.DataFrame(vd_rows)
    vd.to_csv(OUT / "table3_variance_decomposition.csv", index=False)
    print("  saved table3_variance_decomposition.csv")

    # F-tests (Type I; df_resid fixed by the full 5-term model)
    n = len(theta)
    df_mu = len(BIN_LABELS) - 1          # one df absorbed by the grand mean
    df_alpha = len(DOMAINS) - 1
    df_kappa = len(SIZE_LABELS) - 1
    df_beta = (len(DOMAINS) - 1) * (len(BIN_LABELS) - 1)
    df_gamma = (len(DOMAINS) - 1) * (len(SIZE_LABELS) - 1)  # doubly-centered interaction
    df_resid = n - 1 - df_mu - df_alpha - df_kappa - df_beta - df_gamma

    ss_alpha = np.sum(cal["alpha"].values ** 2)
    ss_kappa = np.sum(cal["kappa"].values ** 2)
    ss_beta = np.sum(cal["beta"].values ** 2)
    ss_gamma = np.sum(cal["gamma"].values ** 2)

    tests = []
    for name, ss, df_num in [("domain_alpha", ss_alpha, df_alpha),
                              ("size_kappa", ss_kappa, df_kappa),
                              ("domain_time_beta", ss_beta, df_beta),
                              ("domain_size_gamma", ss_gamma, df_gamma)]:
        if df_resid > 0 and df_num > 0:
            ms_comp = ss / df_num
            ms_resid = ss_resid / df_resid
            F = ms_comp / ms_resid if ms_resid > 0 else np.inf
            p_val = 1 - fdist.cdf(F, df_num, df_resid)
        else:
            F, p_val = np.nan, np.nan
        tests.append(dict(test=name, F_statistic=round(F, 2), df_num=df_num,
                          df_den=df_resid, p_value=f"{p_val:.2e}"))
        print(f"  F-test {name}: F={F:.2f}, df=({df_num},{df_resid}), p={p_val:.2e}")

    # Bootstrap whale effects
    for domain in ["Politics", "Sports"]:
        obs_diff, ci_lo, ci_hi = bootstrap_whale_effect(cal, domain)
        tests.append(dict(test=f"whale_effect_{domain}",
                          F_statistic=round(obs_diff, 4),
                          df_num=0, df_den=0,
                          p_value=f"[{ci_lo:.4f}, {ci_hi:.4f}]"))
        print(f"  Whale effect {domain}: {obs_diff:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    pd.DataFrame(tests).to_csv(OUT / "table4_statistical_tests.csv", index=False)
    print("  saved table4_statistical_tests.csv")

    cal[["domain", "time_bin", "size_bin", "mu", "alpha", "kappa", "beta", "gamma",
         "residual", "fitted", "slope_b"]].rename(
        columns={"slope_b": "observed_theta", "fitted": "fitted_theta"}
    ).to_csv(OUT / "decomposition_components.csv", index=False)
    print("  saved decomposition_components.csv")

    return cal


def step4_weighting(raw_df):
    """Contract-weighted vs trade-weighted comparison."""
    print("\n" + "=" * 70)
    print("  STEP 4: CONTRACT-WEIGHTED vs TRADE-WEIGHTED")
    print("=" * 70)

    rows = []
    for (domain, tbin), cell in raw_df.groupby(["domain", "tbin"]):
        n = int(cell["n_trades"].sum())
        if n < CELL_MIN:
            continue
        prices = cell["yes_price"].values.astype(float)
        outcomes = cell["is_yes"].values.astype(float)
        contracts = cell["total_contracts"].values.astype(float)
        trade_w = cell["n_trades"].values.astype(float)

        res_tw = fit_logistic(prices, outcomes, trade_w)
        res_cw = fit_logistic(prices, outcomes, contracts)

        if res_tw and res_cw:
            rows.append(dict(
                domain=domain, time_bin=BIN_LABELS[int(tbin)],
                slope_trade_weighted=round(res_tw[0], 4),
                slope_contract_weighted=round(res_cw[0], 4),
                difference=round(res_cw[0] - res_tw[0], 4)))

    wt = pd.DataFrame(rows)
    wt.to_csv(OUT / "weighting_comparison.csv", index=False)
    print("  saved weighting_comparison.csv")

    for d in ["Politics", "Sports"]:
        sub = wt[wt["domain"] == d]
        if len(sub) > 0:
            md = sub["difference"].mean()
            print(f"  {d} mean gap (contract - trade): {md:+.4f}")


def main():
    print("=" * 70)
    print("  KALSHI CALIBRATION ANALYSIS")
    print("=" * 70)

    cal, raw_df = step1_base_data()
    tbl1 = step2_summary(raw_df)
    cal = step3_decomposition(cal)
    step4_weighting(raw_df)

    # Save full decomposed calibration matrix
    cal.to_csv(OUT / "calibration_matrix_decomposed.csv", index=False)

    print("\n" + "=" * 70)
    print(f"  DONE — outputs in {OUT}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
