"""Cross-platform Polymarket replication and comparison with Kalshi.

Outputs to output/cross_platform/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration import bootstrap_whale_effect, fit_logistic, fit_slope
from src.classify import classify_polymarket_domain
from src.config import (
    BIN_LABELS,
    CELL_MIN,
    COLORS,
    CROSS_PLATFORM_DOMAINS,
    DATE_CUTOFF,
    OUTPUT_DIR,
    PM_MARKETS,
    PM_TRADES,
    SIZE_LABELS,
)
from src.pipeline import size_bin_sql, time_bin_sql
from src.plotting import (
    fig_cross_platform_trajectories,
    fig_politics_comparison,
    fig_scale_effect_comparison,
)

DOMAINS = CROSS_PLATFORM_DOMAINS
OUT = OUTPUT_DIR / "cross_platform"
OUT.mkdir(parents=True, exist_ok=True)
KALSHI_OUT = OUTPUT_DIR / "kalshi"


def step1_base_data():
    """Load and aggregate Polymarket trade data."""
    print("\n" + "=" * 70)
    print("  STEP 1: PRE-AGGREGATE POLYMARKET TRADE DATA")
    print("=" * 70)

    conn = duckdb.connect()
    tb = time_bin_sql()
    sb = size_bin_sql()

    pm_markets = str(PM_MARKETS).replace("\\", "/")
    pm_trades = str(PM_TRADES).replace("\\", "/")

    # Load markets and classify in Python
    markets_df = conn.execute(f"""
        SELECT ticker, event_ticker, result, close_time, title, _domain, status
        FROM '{pm_markets}'
    """).df()
    markets_df["domain"] = markets_df["title"].apply(classify_polymarket_domain)
    conn.register("pm_markets_classified", markets_df)

    df = conn.execute(f"""
        WITH resolved AS (
            SELECT ticker, event_ticker, result, close_time, domain
            FROM pm_markets_classified
            WHERE status = 'finalized' AND result IN ('yes', 'no')
              AND domain IN ({','.join(f"'{d}'" for d in DOMAINS)})
        ),
        trade_data AS (
            SELECT t.yes_price, t.count AS trade_count,
                   CASE WHEN m.result = 'yes' THEN 1 ELSE 0 END AS is_yes,
                   m.domain,
                   EXTRACT(EPOCH FROM (m.close_time - t.created_time)) / 3600.0 AS hours_to_close,
                   ({sb}) AS sbin,
                   m.ticker
            FROM '{pm_trades}' t
            INNER JOIN resolved m ON t.ticker = m.ticker
            WHERE t.created_time <= TIMESTAMP '{DATE_CUTOFF}'
              AND m.close_time > t.created_time
        ),
        market_counts AS (
            SELECT ticker, COUNT(*) AS ntrades
            FROM trade_data
            GROUP BY ticker
            HAVING COUNT(*) >= 10
        )
        SELECT td.domain, ({tb}) AS tbin, td.sbin, td.yes_price, td.is_yes,
               SUM(td.trade_count) AS total_contracts, COUNT(*) AS n_trades
        FROM trade_data td
        INNER JOIN market_counts mc ON td.ticker = mc.ticker
        WHERE td.yes_price BETWEEN 5 AND 95 AND ({tb}) >= 0 AND td.sbin >= 0
        GROUP BY td.domain, ({tb}), td.sbin, td.yes_price, td.is_yes
    """).df()
    conn.close()

    total_trades = int(df["n_trades"].sum())
    print(f"  Loaded {total_trades:,} trades")
    for d in DOMAINS:
        n = int(df[df["domain"] == d]["n_trades"].sum())
        print(f"    {d:>15s}: {n:>12,} trades")

    return df, markets_df


def step2_slopes(df):
    """Fit calibration slopes by domain x time and domain x size."""
    print("\n" + "=" * 70)
    print("  STEP 2: CALIBRATION SLOPES")
    print("=" * 70)

    # Domain x time
    dt_rows = []
    for (domain, tbin), cell in df.groupby(["domain", "tbin"]):
        n_t = int(cell["n_trades"].sum())
        if n_t < CELL_MIN:
            continue
        result = fit_logistic(
            cell["yes_price"].values.astype(float),
            cell["is_yes"].values.astype(float),
            cell["total_contracts"].values.astype(float),
        )
        if result:
            b, a, se = result
            dt_rows.append(dict(
                domain=domain, time_bin=BIN_LABELS[int(tbin)],
                time_bin_order=int(tbin) + 1, n_trades=n_t,
                slope_b=b, intercept_a=a, slope_stderr=se,
            ))
    dt = pd.DataFrame(dt_rows)
    dt.to_csv(OUT / "polymarket_slopes_by_domain_time.csv", index=False)
    print(f"  {len(dt)} cells -> polymarket_slopes_by_domain_time.csv")

    # Full matrix for whale bootstrap
    full_rows = []
    for (domain, tbin, sbin), cell in df.groupby(["domain", "tbin", "sbin"]):
        n_t = int(cell["n_trades"].sum())
        if n_t < CELL_MIN:
            continue
        result = fit_logistic(
            cell["yes_price"].values.astype(float),
            cell["is_yes"].values.astype(float),
            cell["total_contracts"].values.astype(float),
        )
        if result:
            b, a, se = result
            full_rows.append(dict(
                domain=domain, time_bin=BIN_LABELS[int(tbin)],
                time_bin_order=int(tbin) + 1,
                size_bin=SIZE_LABELS[int(sbin)],
                size_bin_order=int(sbin) + 1,
                n_trades=n_t, slope_b=b, intercept_a=a, slope_stderr=se,
            ))
    cal = pd.DataFrame(full_rows)

    # Domain x size
    ds_rows = []
    for (domain, sbin), cell in df.groupby(["domain", "sbin"]):
        n_t = int(cell["n_trades"].sum())
        if n_t < CELL_MIN:
            continue
        result = fit_logistic(
            cell["yes_price"].values.astype(float),
            cell["is_yes"].values.astype(float),
            cell["total_contracts"].values.astype(float),
        )
        if result:
            b, a, se = result
            ds_rows.append(dict(
                domain=domain, size_bin=SIZE_LABELS[int(sbin)],
                size_bin_order=int(sbin) + 1, n_trades=n_t,
                slope_b=b, intercept_a=a, slope_stderr=se,
            ))
    ds = pd.DataFrame(ds_rows)
    ds.to_csv(OUT / "polymarket_slopes_by_domain_size.csv", index=False)
    print(f"  {len(ds)} cells -> polymarket_slopes_by_domain_size.csv")

    # Bootstrap whale effect
    print("\n  Bootstrap whale effect:")
    boot_rows = []
    for domain in DOMAINS:
        obs_diff, ci_lo, ci_hi = bootstrap_whale_effect(cal, domain)
        if np.isnan(obs_diff):
            print(f"    {domain}: insufficient data")
            continue
        significant = "YES" if (ci_lo > 0 or ci_hi < 0) else "no"
        boot_rows.append(dict(
            domain=domain, obs_delta=round(obs_diff, 4),
            ci_2_5=round(ci_lo, 4), ci_97_5=round(ci_hi, 4),
            significant=significant,
        ))
        print(f"    {domain}: {obs_diff:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}] {significant}")
    pd.DataFrame(boot_rows).to_csv(OUT / "polymarket_whale_bootstrap.csv", index=False)

    return dt, ds, cal


def step3_weighting(df):
    """Contract-weighted vs trade-weighted comparison."""
    print("\n" + "=" * 70)
    print("  STEP 3: WEIGHTING COMPARISON")
    print("=" * 70)

    rows = []
    for (domain, tbin), cell in df.groupby(["domain", "tbin"]):
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
                difference=round(res_cw[0] - res_tw[0], 4),
            ))

    wt = pd.DataFrame(rows)
    wt.to_csv(OUT / "polymarket_weighting_comparison.csv", index=False)
    print("  saved polymarket_weighting_comparison.csv")
    return wt


def step4_comparison(pm_dt, pm_ds):
    """Cross-platform comparison tables."""
    print("\n" + "=" * 70)
    print("  STEP 4: CROSS-PLATFORM COMPARISON")
    print("=" * 70)

    kalshi_dt = pd.read_csv(KALSHI_OUT / "calibration_slopes_by_domain_time.csv")
    kalshi_ds = pd.read_csv(KALSHI_OUT / "calibration_slopes_by_domain_size.csv")

    # Time comparison
    comp_rows = []
    for d in DOMAINS:
        for tl in BIN_LABELS:
            k_sub = kalshi_dt[(kalshi_dt["domain"] == d) & (kalshi_dt["time_bin"] == tl)]
            p_sub = pm_dt[(pm_dt["domain"] == d) & (pm_dt["time_bin"] == tl)]
            k_slope = float(k_sub["slope_b"].iloc[0]) if len(k_sub) > 0 else np.nan
            p_slope = float(p_sub["slope_b"].iloc[0]) if len(p_sub) > 0 else np.nan
            comp_rows.append(dict(
                domain=d, time_bin=tl,
                kalshi_slope=round(k_slope, 4) if not np.isnan(k_slope) else np.nan,
                polymarket_slope=round(p_slope, 4) if not np.isnan(p_slope) else np.nan,
                difference=round(p_slope - k_slope, 4) if not (np.isnan(k_slope) or np.isnan(p_slope)) else np.nan,
            ))
    pd.DataFrame(comp_rows).to_csv(OUT / "cross_platform_comparison_time.csv", index=False)
    print("  saved cross_platform_comparison_time.csv")

    return kalshi_dt, kalshi_ds


def step5_figures(pm_dt, pm_ds, kalshi_dt, kalshi_ds):
    """Generate cross-platform figures."""
    print("\n" + "=" * 70)
    print("  STEP 5: FIGURES")
    print("=" * 70)

    fig_cross_platform_trajectories(pm_dt, kalshi_dt, str(OUT / "figure_cp1_slope_trajectories"), DOMAINS)
    print("  saved figure_cp1_slope_trajectories.{png,pdf}")

    fig_politics_comparison(kalshi_dt, pm_dt, str(OUT / "figure_cp2_politics_comparison"))
    print("  saved figure_cp2_politics_comparison.{png,pdf}")

    fig_scale_effect_comparison(kalshi_ds, pm_ds, str(OUT / "figure_cp3_scale_effect"))
    print("  saved figure_cp3_scale_effect.{png,pdf}")


def step6_clustered_and_size(markets_df):
    """Polymarket market-clustered whale bootstrap (parity with Kalshi; the
    submitted manuscript flagged its absence) and cross-platform trade-size
    distribution (Reviewer 1: show whether Kalshi political bets are larger)."""
    print("\n" + "=" * 70)
    print("  STEP 6: PM MARKET-CLUSTERED BOOTSTRAP + CROSS-PLATFORM SIZE")
    print("=" * 70)

    pm_trades = str(PM_TRADES).replace("\\", "/")
    conn = duckdb.connect()
    conn.register("pm_markets_classified", markets_df)

    # ticker-level cell weights for clustered bootstrap
    tick = conn.execute(f"""
        WITH resolved AS (
            SELECT ticker, result, close_time, domain FROM pm_markets_classified
            WHERE status='finalized' AND result IN ('yes','no')
              AND domain IN ({','.join(f"'{d}'" for d in DOMAINS)})
        ),
        td AS (
            SELECT t.ticker, m.domain, t.yes_price,
                   CASE WHEN m.result='yes' THEN 1 ELSE 0 END AS is_yes,
                   CASE WHEN t.count=1 THEN 0 WHEN t.count<=10 THEN 1
                        WHEN t.count<=100 THEN 2 ELSE 3 END AS sbin,
                   t.count AS weight
            FROM '{pm_trades}' t INNER JOIN resolved m ON t.ticker=m.ticker
            WHERE t.created_time <= TIMESTAMP '{DATE_CUTOFF}'
              AND m.close_time > t.created_time AND t.yes_price BETWEEN 5 AND 95
        ),
        mc AS (SELECT ticker FROM td GROUP BY ticker HAVING COUNT(*) >= 10)
        SELECT td.domain, td.ticker, td.sbin, td.yes_price, td.is_yes, SUM(td.weight) AS weight
        FROM td INNER JOIN mc ON td.ticker=mc.ticker
        GROUP BY td.domain, td.ticker, td.sbin, td.yes_price, td.is_yes
    """).df()

    # trade-size frequency by domain
    size_freq = conn.execute(f"""
        WITH resolved AS (
            SELECT ticker, domain FROM pm_markets_classified
            WHERE status='finalized' AND result IN ('yes','no')
              AND domain IN ({','.join(f"'{d}'" for d in DOMAINS)})
        )
        SELECT m.domain, t.count AS trade_size, COUNT(*) AS freq
        FROM '{pm_trades}' t INNER JOIN resolved m ON t.ticker=m.ticker
        WHERE t.created_time <= TIMESTAMP '{DATE_CUTOFF}' AND t.yes_price BETWEEN 5 AND 95
        GROUP BY m.domain, t.count
    """).df()
    conn.close()

    prices_flat = np.repeat(np.arange(5, 96), 2).astype(float)
    outcomes_flat = np.tile(np.array([0.0, 1.0]), 91)
    N_ITER = 5000
    boot_rows = []
    for domain in ["Politics", "Sports", "Crypto"]:
        dom = tick[tick["domain"] == domain]
        tickers = np.sort(dom["ticker"].unique())  # deterministic index mapping
        n_u = len(tickers)
        if n_u < 2:
            continue
        idxmap = {t: i for i, t in enumerate(tickers)}
        tensor = np.zeros((n_u, 91, 2, 4))
        np.add.at(tensor, (dom["ticker"].map(idxmap).values, dom["yes_price"].astype(int).values - 5,
                           dom["is_yes"].astype(int).values, dom["sbin"].astype(int).values),
                  dom["weight"].astype(float).values)
        rng = np.random.default_rng(42)
        diffs = np.full(N_ITER, np.nan)
        for i in range(N_ITER):
            cnt = np.bincount(rng.integers(0, n_u, n_u), minlength=n_u).astype(float)
            tw = np.einsum("i,ijkl->jkl", cnt, tensor)
            wl, ws = tw[:, :, 3].ravel(), tw[:, :, 0].ravel()
            ml, ms = wl > 0, ws > 0
            sl = fit_slope(prices_flat[ml], outcomes_flat[ml], wl[ml]) if ml.sum() >= 5 else np.nan
            ss = fit_slope(prices_flat[ms], outcomes_flat[ms], ws[ms]) if ms.sum() >= 5 else np.nan
            diffs[i] = sl - ss
        v = diffs[~np.isnan(diffs)]
        lo, hi = np.percentile(v, [2.5, 97.5])
        boot_rows.append(dict(platform="Polymarket", domain=domain, method="market_clustered",
                              n_clusters=n_u, mean_diff=round(float(v.mean()), 4),
                              ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4),
                              sig="YES" if (lo > 0 or hi < 0) else "no"))
        print(f"    {domain}: {v.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] ({n_u} markets)")
    pd.DataFrame(boot_rows).to_csv(OUT / "polymarket_whale_clustered_bootstrap.csv", index=False)

    # cross-platform trade-size distribution
    def wq(s, f, q):
        o = np.argsort(s); s, f = s[o], f[o]
        return float(np.interp(q * f.sum(), np.cumsum(f) - 0.5 * f, s))
    rows = []
    for d, g in size_freq[size_freq["domain"].isin(DOMAINS)].groupby("domain"):
        s = g["trade_size"].to_numpy(float); f = g["freq"].to_numpy(float)
        rows.append(dict(platform="Polymarket", domain=d, n_trades=int(f.sum()),
                         median_size=round(wq(s, f, .5), 1), p90_size=round(wq(s, f, .9), 1),
                         p99_size=round(wq(s, f, .99), 1), mean_size=round((s * f).sum() / f.sum(), 1),
                         frac_vol_from_large=round((s[s > 100] * f[s > 100]).sum() / (s * f).sum(), 4)))
    pd.DataFrame(rows).to_csv(OUT / "polymarket_tradesize_distribution.csv", index=False)
    print("  saved polymarket_whale_clustered_bootstrap.csv, polymarket_tradesize_distribution.csv")


def main():
    print("=" * 70)
    print("  CROSS-PLATFORM CALIBRATION: POLYMARKET vs KALSHI")
    print("=" * 70)

    for path, label in [(PM_MARKETS, "Polymarket markets"), (PM_TRADES, "Polymarket CTF trades")]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}")
            sys.exit(1)

    df, markets_df = step1_base_data()
    pm_dt, pm_ds, pm_cal = step2_slopes(df)
    step3_weighting(df)
    kalshi_dt, kalshi_ds = step4_comparison(pm_dt, pm_ds)
    step5_figures(pm_dt, pm_ds, kalshi_dt, kalshi_ds)
    step6_clustered_and_size(markets_df)

    print("\n" + "=" * 70)
    print(f"  DONE — outputs in {OUT}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
