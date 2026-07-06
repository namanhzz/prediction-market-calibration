"""All robustness checks: clustered bootstrap, price range, weighted decomposition, ANOVA, confound.

Outputs to output/robustness/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration import (
    compute_weighted_decomposition,
    decompose,
    fit_logistic,
    fit_slope,
    fit_slope_with_se,
)
from src.classify import get_group
from src.config import (
    BIN_LABELS,
    CELL_MIN,
    DATE_CUTOFF,
    DOMAINS,
    KALSHI_MARKETS,
    KALSHI_TRADES,
    OUTPUT_DIR,
    SIZE_LABELS,
    TIME_BINS,
)
from src.pipeline import size_bin_sql, time_bin_sql

OUT = OUTPUT_DIR / "robustness"
OUT.mkdir(parents=True, exist_ok=True)
KALSHI_OUT = OUTPUT_DIR / "kalshi"


# ═══════════════════════════════════════════════════════════════════
# FIX 1: Market-clustered bootstrap for whale effect
# ═══════════════════════════════════════════════════════════════════

def fix1_clustered_bootstrap():
    """Clustered bootstrap for the whale effect at two cluster levels:
    market (ticker, ~contract) and event (event_ticker, groups related yes/no
    and sibling contracts). Addresses R1's request to cluster across both the
    contract and event dimensions."""
    print("\n" + "=" * 70)
    print("  FIX 1: MARKET- AND EVENT-CLUSTERED BOOTSTRAP")
    print("=" * 70)

    markets = str(KALSHI_MARKETS).replace("\\", "/")
    trades = str(KALSHI_TRADES).replace("\\", "/")
    conn = duckdb.connect()

    df = conn.execute(f"""
        WITH resolved AS (
            SELECT ticker, event_ticker, result, close_time
            FROM '{markets}/*.parquet'
            WHERE status='finalized' AND result IN ('yes','no')
        ),
        trade_data AS (
            SELECT t.yes_price, t.count AS weight,
                   CASE WHEN m.result='yes' THEN 1 ELSE 0 END AS is_yes,
                   CASE WHEN t.count = 1 THEN 0 WHEN t.count <= 10 THEN 1
                        WHEN t.count <= 100 THEN 2 ELSE 3 END AS sbin,
                   m.ticker, m.event_ticker,
                   regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) AS cat_prefix
            FROM '{trades}/*.parquet' t
            INNER JOIN resolved m ON t.ticker = m.ticker
            WHERE t.created_time <= TIMESTAMP '{DATE_CUTOFF}'
              AND m.close_time > t.created_time
              AND t.yes_price BETWEEN 5 AND 95
        ),
        market_counts AS (
            SELECT ticker FROM trade_data GROUP BY ticker HAVING COUNT(*) >= 10
        )
        SELECT td.ticker, td.event_ticker, td.sbin, td.yes_price, td.is_yes,
               SUM(td.weight) AS weight, td.cat_prefix
        FROM trade_data td
        INNER JOIN market_counts mc ON td.ticker = mc.ticker
        GROUP BY td.ticker, td.event_ticker, td.sbin, td.yes_price, td.is_yes, td.cat_prefix
    """).df()
    conn.close()

    df["domain"] = df["cat_prefix"].apply(get_group)
    print(f"  Loaded {len(df):,} aggregated rows")

    N_ITER = 5000
    prices_flat = np.repeat(np.arange(5, 96), 2).astype(float)
    outcomes_flat = np.tile(np.array([0.0, 1.0]), 91)

    def cluster_bootstrap(dom_df, cluster_col):
        units = np.sort(dom_df[cluster_col].unique())  # deterministic index mapping
        n_units = len(units)
        unit_to_idx = {u: i for i, u in enumerate(units)}
        weight_tensor = np.zeros((n_units, 91, 2, 4), dtype=np.float64)
        ui = dom_df[cluster_col].map(unit_to_idx).values
        pi = dom_df["yes_price"].values.astype(int) - 5
        oi = dom_df["is_yes"].values.astype(int)
        si = dom_df["sbin"].values.astype(int)
        wv = dom_df["weight"].values.astype(float)
        np.add.at(weight_tensor, (ui, pi, oi, si), wv)

        rng = np.random.default_rng(42)
        diffs = np.full(N_ITER, np.nan)
        for i in range(N_ITER):
            idx = rng.integers(0, n_units, size=n_units)
            counts = np.bincount(idx, minlength=n_units).astype(np.float64)
            total_w = np.einsum("i,ijkl->jkl", counts, weight_tensor)
            w_large = total_w[:, :, 3].ravel()
            w_single = total_w[:, :, 0].ravel()
            mask_l = w_large > 0
            mask_s = w_single > 0
            sl = fit_slope(prices_flat[mask_l], outcomes_flat[mask_l], w_large[mask_l]) if mask_l.sum() >= 5 else np.nan
            ss = fit_slope(prices_flat[mask_s], outcomes_flat[mask_s], w_single[mask_s]) if mask_s.sum() >= 5 else np.nan
            diffs[i] = sl - ss
        valid = diffs[~np.isnan(diffs)]
        return n_units, valid

    results_rows = []
    for domain in ["Politics", "Sports"]:
        print(f"\n  {domain}:")
        dom_df = df[df["domain"] == domain].copy()
        for cluster_col, method in [("ticker", "market_clustered"), ("event_ticker", "event_clustered")]:
            n_units, valid = cluster_bootstrap(dom_df, cluster_col)
            ci_lo, ci_hi = np.percentile(valid, [2.5, 97.5])
            mean_diff = float(np.mean(valid))
            print(f"    {method:>16s} ({n_units} {cluster_col}s): {mean_diff:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
            results_rows.append(dict(
                domain=domain, method=method, n_clusters=n_units,
                mean_diff=round(mean_diff, 4), ci_lo=round(ci_lo, 4),
                ci_hi=round(ci_hi, 4), n_valid=len(valid),
            ))

    result = pd.DataFrame(results_rows)
    result.to_csv(OUT / "whale_effect_clustered_bootstrap.csv", index=False)
    print("\n  saved whale_effect_clustered_bootstrap.csv")


# ═══════════════════════════════════════════════════════════════════
# FIX 5: Extended price range robustness
# ═══════════════════════════════════════════════════════════════════

def fix5_price_range():
    """Test decomposition stability across different price ranges."""
    print("\n" + "=" * 70)
    print("  FIX 5: PRICE RANGE ROBUSTNESS")
    print("=" * 70)

    markets = str(KALSHI_MARKETS).replace("\\", "/")
    trades = str(KALSHI_TRADES).replace("\\", "/")
    tb = time_bin_sql()
    sb = size_bin_sql()

    def load_and_fit(price_lo, price_hi, C=10.0):
        conn = duckdb.connect()
        raw = conn.execute(f"""
            WITH resolved AS (
                SELECT ticker, event_ticker, result, close_time
                FROM '{markets}/*.parquet'
                WHERE status='finalized' AND result IN ('yes','no')
            ),
            trade_data AS (
                SELECT t.yes_price, t.count AS trade_count,
                       CASE WHEN m.result='yes' THEN 1 ELSE 0 END AS is_yes,
                       regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) AS cat_prefix,
                       EXTRACT(EPOCH FROM (m.close_time - t.created_time))/3600.0 AS hours_to_close,
                       ({sb}) AS sbin, m.ticker
                FROM '{trades}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
                WHERE t.created_time <= TIMESTAMP '{DATE_CUTOFF}'
                  AND m.close_time > t.created_time
            ),
            market_counts AS (
                SELECT ticker FROM trade_data GROUP BY ticker HAVING COUNT(*) >= 10
            )
            SELECT td.cat_prefix, ({tb}) AS tbin, td.sbin, td.yes_price, td.is_yes,
                   SUM(td.trade_count) AS total_contracts, COUNT(*) AS n_trades
            FROM trade_data td
            INNER JOIN market_counts mc ON td.ticker = mc.ticker
            WHERE td.yes_price BETWEEN {price_lo} AND {price_hi} AND ({tb}) >= 0 AND td.sbin >= 0
            GROUP BY td.cat_prefix, ({tb}), td.sbin, td.yes_price, td.is_yes
        """).df()
        conn.close()

        raw["domain"] = raw["cat_prefix"].apply(get_group)
        raw = raw[raw["domain"].isin(DOMAINS)].copy()

        rows = []
        for (domain, tbin, sbin), cell in raw.groupby(["domain", "tbin", "sbin"]):
            n_t = int(cell["n_trades"].sum())
            if n_t < CELL_MIN:
                continue
            result = fit_logistic(
                cell["yes_price"].values.astype(float),
                cell["is_yes"].values.astype(float),
                cell["total_contracts"].values.astype(float),
                C=C,
            )
            if result is None:
                continue
            b, a, se = result
            rows.append(dict(
                domain=domain, time_bin=BIN_LABELS[int(tbin)],
                size_bin=SIZE_LABELS[int(sbin)],
                n_trades=n_t, slope_b=b, slope_stderr=se,
            ))
        return pd.DataFrame(rows)

    def block_r2(cal):
        """Four-block grouping (size block = kappa+gamma) of the five-term fit."""
        cal = decompose(cal.copy())
        theta = cal["slope_b"].values
        ss_total = np.sum((theta - theta.mean()) ** 2)
        fitted_cumul = np.zeros(len(theta)); prev = 0.0; marg = {}
        for comp in ["mu", "alpha", "kappa", "beta", "gamma"]:
            fitted_cumul = fitted_cumul + cal[comp].values
            c_r2 = np.sum((fitted_cumul - theta.mean()) ** 2) / ss_total if ss_total > 0 else 0
            marg[comp] = c_r2 - prev
            prev = c_r2
        return {"mu": marg["mu"], "alpha": marg["alpha"], "beta": marg["beta"],
                "gamma": marg["kappa"] + marg["gamma"]}, prev, len(cal)

    # Price-range robustness (C=10) and L2-regularization sweep (C=1,10,100 at [5,95]).
    price_configs = [(5, 95, 10.0, "Baseline [5,95], C=10"), (2, 98, 10.0, "Price [2,98]"),
                     (1, 99, 10.0, "Price [1,99]"), (10, 90, 10.0, "Price [10,90]")]
    c_configs = [(5, 95, 1.0, "C=1"), (5, 95, 100.0, "C=100")]
    decomp_rows = []
    for lo, hi, C, label in price_configs + c_configs:
        print(f"\n  Computing {label}...")
        blocks, total, ncell = block_r2(load_and_fit(lo, hi, C=C))
        cum = 0.0
        for comp in ["mu", "alpha", "beta", "gamma"]:
            cum += blocks[comp]
            decomp_rows.append(dict(check_name=label, component=comp,
                                    marginal_r2=round(blocks[comp], 4), cumulative_r2=round(cum, 4)))
        decomp_rows.append(dict(check_name=label, component="total",
                                marginal_r2=round(total, 4), cumulative_r2=round(total, 4)))
        print(f"    {ncell} cells, total R²={total:.4f}")

    pd.DataFrame(decomp_rows).to_csv(OUT / "price_range_robustness.csv", index=False)
    print("  saved price_range_robustness.csv")


# ═══════════════════════════════════════════════════════════════════
# FIX 2: Weighted variance decomposition
# ═══════════════════════════════════════════════════════════════════

def fix2_weighted_decomposition():
    """Compare unweighted vs inverse-variance weighted decomposition."""
    print("\n" + "=" * 70)
    print("  FIX 2: WEIGHTED DECOMPOSITION")
    print("=" * 70)

    cal = pd.read_csv(KALSHI_OUT / "calibration_matrix.csv")
    slopes = cal["slope_b"].values
    time_bins = cal["time_bin"].values
    domains_arr = cal["domain"].values
    size_bins = cal["size_bin"].values
    se = cal["slope_stderr"].values

    uw = compute_weighted_decomposition(slopes, time_bins, domains_arr, size_bins, np.ones(len(slopes)))
    weights = 1.0 / (se ** 2)
    wt = compute_weighted_decomposition(slopes, time_bins, domains_arr, size_bins, weights)

    print(f"\n  {'Component':<12} {'Unweighted':>12} {'Weighted':>12}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 12}")
    for comp in ["mu", "alpha", "kappa", "beta", "gamma"]:
        print(f"  {comp:<12} {uw[comp]:>12.4f} {wt[comp]:>12.4f}")
    print(f"  {'TOTAL':<12} {uw['total']:>12.4f} {wt['total']:>12.4f}")
    print("  Note: weighted total is an inverse-variance weighted in-sample")
    print("  fit, not an out-of-sample R²; see leave-one-cell-out for OOS.")

    rows = []
    for comp in ["mu", "alpha", "kappa", "beta", "gamma"]:
        rows.append({"type": "Unweighted", "component": comp, "marginal_r2": round(uw[comp], 4)})
        rows.append({"type": "Weighted", "component": comp, "marginal_r2": round(wt[comp], 4)})
    rows.append({"type": "Unweighted", "component": "total", "marginal_r2": round(uw["total"], 4)})
    rows.append({"type": "Weighted", "component": "total", "marginal_r2": round(wt["total"], 4)})
    pd.DataFrame(rows).to_csv(OUT / "weighted_variance_decomposition.csv", index=False)
    print("  saved weighted_variance_decomposition.csv")


# ═══════════════════════════════════════════════════════════════════
# FIX 3: Type I/II/III ANOVA
# ═══════════════════════════════════════════════════════════════════

def fix3_anova():
    """Compare Type I, II, III sums of squares."""
    print("\n" + "=" * 70)
    print("  FIX 3: TYPE I/II/III ANOVA")
    print("=" * 70)

    cal = pd.read_csv(KALSHI_OUT / "calibration_matrix.csv")
    y = cal["slope_b"].values
    n = len(y)

    def one_hot(values):
        levels = sorted(set(values))
        mat = np.zeros((len(values), len(levels) - 1))
        for j, level in enumerate(levels[1:]):
            mat[:, j] = (np.array(values) == level).astype(float)
        return mat

    def interaction(v1, v2):
        oh1, oh2 = one_hot(v1), one_hot(v2)
        cols = []
        for i in range(oh1.shape[1]):
            for j in range(oh2.shape[1]):
                cols.append(oh1[:, i] * oh2[:, j])
        return np.column_stack(cols) if cols else np.zeros((n, 0))

    X_time = one_hot(cal["time_bin"].values)
    X_domain = one_hot(cal["domain"].values)
    X_td = interaction(cal["time_bin"].values, cal["domain"].values)
    X_ds = interaction(cal["domain"].values, cal["size_bin"].values)
    intercept = np.ones((n, 1))

    def ss_res(X):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(np.sum((y - X @ beta) ** 2))

    models = {
        "null": intercept,
        "T": np.hstack([intercept, X_time]),
        "T+D": np.hstack([intercept, X_time, X_domain]),
        "T+D+TD": np.hstack([intercept, X_time, X_domain, X_td]),
        "full": np.hstack([intercept, X_time, X_domain, X_td, X_ds]),
    }
    ss = {k: ss_res(v) for k, v in models.items()}
    ss_total = float(np.sum((y - y.mean()) ** 2))

    type1 = {
        "time_bin": ss["null"] - ss["T"],
        "domain": ss["T"] - ss["T+D"],
        "time:domain": ss["T+D"] - ss["T+D+TD"],
        "domain:size": ss["T+D+TD"] - ss["full"],
    }

    models_drop = {
        "time_bin": np.hstack([intercept, X_domain, X_td, X_ds]),
        "domain": np.hstack([intercept, X_time, X_td, X_ds]),
        "time:domain": np.hstack([intercept, X_time, X_domain, X_ds]),
        "domain:size": np.hstack([intercept, X_time, X_domain, X_td]),
    }
    type3 = {k: ss_res(v) - ss["full"] for k, v in models_drop.items()}

    models_type2 = {
        "time_bin": (np.hstack([intercept, X_domain, X_ds]),
                     np.hstack([intercept, X_time, X_domain, X_ds])),
        "domain": (np.hstack([intercept, X_time, X_ds]),
                   np.hstack([intercept, X_time, X_domain, X_ds])),
        "time:domain": (np.hstack([intercept, X_time, X_domain, X_ds]),
                        np.hstack([intercept, X_time, X_domain, X_td, X_ds])),
        "domain:size": (np.hstack([intercept, X_time, X_domain, X_td]),
                        np.hstack([intercept, X_time, X_domain, X_td, X_ds])),
    }
    type2 = {k: ss_res(v[0]) - ss_res(v[1]) for k, v in models_type2.items()}

    rows = []
    for term in ["time_bin", "domain", "time:domain", "domain:size"]:
        rows.append(dict(
            term=term,
            type_I_SS=round(type1[term], 6), type_I_pct=round(100 * type1[term] / ss_total, 2),
            type_II_SS=round(type2[term], 6), type_II_pct=round(100 * type2[term] / ss_total, 2),
            type_III_SS=round(type3[term], 6), type_III_pct=round(100 * type3[term] / ss_total, 2),
        ))

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "anova_type_comparison.csv", index=False)
    print("  saved anova_type_comparison.csv")
    print(f"\n  {'Term':<15} {'Type I %':>10} {'Type II %':>10} {'Type III %':>10}")
    for _, r in result.iterrows():
        print(f"  {r['term']:<15} {r['type_I_pct']:>10.2f} {r['type_II_pct']:>10.2f} {r['type_III_pct']:>10.2f}")


# ═══════════════════════════════════════════════════════════════════
# FIX 4: Size x Horizon confound
# ═══════════════════════════════════════════════════════════════════

def fix4_size_horizon_confound():
    """Check if size effect persists after controlling for horizon."""
    print("\n" + "=" * 70)
    print("  FIX 4: SIZE x HORIZON CONFOUND")
    print("=" * 70)

    cal = pd.read_csv(KALSHI_OUT / "calibration_matrix.csv")
    y = cal["slope_b"].values
    n = len(y)

    def one_hot(values):
        levels = sorted(set(values))
        mat = np.zeros((len(values), len(levels) - 1))
        for j, level in enumerate(levels[1:]):
            mat[:, j] = (np.array(values) == level).astype(float)
        return mat

    def interaction(v1, v2):
        oh1, oh2 = one_hot(v1), one_hot(v2)
        cols = []
        for i in range(oh1.shape[1]):
            for j in range(oh2.shape[1]):
                cols.append(oh1[:, i] * oh2[:, j])
        return np.column_stack(cols) if cols else np.zeros((n, 0))

    intercept = np.ones((n, 1))
    X_time = one_hot(cal["time_bin"].values)
    X_domain = one_hot(cal["domain"].values)
    X_td = interaction(cal["time_bin"].values, cal["domain"].values)
    X_ds = interaction(cal["domain"].values, cal["size_bin"].values)
    X_ts = interaction(cal["time_bin"].values, cal["size_bin"].values)

    def ss_res(X):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(np.sum((y - X @ beta) ** 2))

    ss_total = float(np.sum((y - y.mean()) ** 2))

    ss_base = ss_res(np.hstack([intercept, X_time, X_domain, X_td, X_ds]))
    ss_ext = ss_res(np.hstack([intercept, X_time, X_domain, X_td, X_ds, X_ts]))
    ss_no_ds = ss_res(np.hstack([intercept, X_time, X_domain, X_td]))
    ss_with_ts = ss_res(np.hstack([intercept, X_time, X_domain, X_td, X_ts]))

    gamma_without_ts = (ss_no_ds - ss_base) / ss_total
    gamma_with_ts = (ss_with_ts - ss_ext) / ss_total
    ts_marginal = (ss_base - ss_ext) / ss_total

    print(f"  gamma_d(s) without tau x s: {gamma_without_ts:.4f}")
    print(f"  gamma_d(s) with tau x s:    {gamma_with_ts:.4f}")
    print(f"  tau x s marginal R2:        {ts_marginal:.4f}")

    # Within-time-bin whale effect for Politics
    pol = cal[cal["domain"] == "Politics"]
    whale_rows = []
    for tb in BIN_LABELS:
        single = pol[(pol["time_bin"] == tb) & (pol["size_bin"] == "Single")]
        large = pol[(pol["time_bin"] == tb) & (pol["size_bin"] == "Large")]
        s_val = float(single["slope_b"].iloc[0]) if len(single) > 0 else np.nan
        l_val = float(large["slope_b"].iloc[0]) if len(large) > 0 else np.nan
        whale_rows.append(dict(time_bin=tb, single_slope=round(s_val, 4),
                               large_slope=round(l_val, 4), difference=round(l_val - s_val, 4)))

    all_rows = [
        dict(metric="gamma_without_ts", value=round(gamma_without_ts, 4)),
        dict(metric="gamma_with_ts", value=round(gamma_with_ts, 4)),
        dict(metric="ts_marginal", value=round(ts_marginal, 4)),
    ]
    pd.DataFrame(all_rows).to_csv(OUT / "size_horizon_confound.csv", index=False)
    pd.DataFrame(whale_rows).to_csv(OUT / "politics_whale_by_time.csv", index=False)
    print("  saved size_horizon_confound.csv, politics_whale_by_time.csv")


# ═══════════════════════════════════════════════════════════════════
# FIX 6: Permutation null and k-fold CV for the decomposition R²
# ═══════════════════════════════════════════════════════════════════

def fix6_permutation_and_cv(n_perm=5000, n_folds=10, seed=42):
    """Assess whether the 87.3% in-sample R² exceeds what model flexibility
    alone (72 parameters on 216 cells) would produce under no structure.

    - Permutation null: shuffle the slope vector across cells, refit the full
      five-term design, record R². p-value = P(null R² >= observed).
    - k-fold CV R²: out-of-sample predictive R² to complement leave-one-cell-out.
    """
    print("\n" + "=" * 70)
    print("  FIX 6: PERMUTATION NULL + K-FOLD CV FOR R²")
    print("=" * 70)

    cal = pd.read_csv(KALSHI_OUT / "calibration_matrix.csv")
    y = cal["slope_b"].values.astype(float)
    n = len(y)

    def one_hot(values):
        levels = sorted(set(values))
        mat = np.zeros((len(values), len(levels) - 1))
        for j, level in enumerate(levels[1:]):
            mat[:, j] = (np.array(values) == level).astype(float)
        return mat

    def interaction(v1, v2):
        oh1, oh2 = one_hot(v1), one_hot(v2)
        cols = [oh1[:, i] * oh2[:, j] for i in range(oh1.shape[1]) for j in range(oh2.shape[1])]
        return np.column_stack(cols) if cols else np.zeros((n, 0))

    # Full five-term design: intercept + time + domain + size + time:domain + domain:size
    X = np.hstack([
        np.ones((n, 1)),
        one_hot(cal["time_bin"].values),
        one_hot(cal["domain"].values),
        one_hot(cal["size_bin"].values),
        interaction(cal["time_bin"].values, cal["domain"].values),
        interaction(cal["domain"].values, cal["size_bin"].values),
    ])
    p = X.shape[1]
    ss_total = float(np.sum((y - y.mean()) ** 2))

    def r2_of(yvec):
        beta = np.linalg.lstsq(X, yvec, rcond=None)[0]
        return 1.0 - float(np.sum((yvec - X @ beta) ** 2)) / float(np.sum((yvec - yvec.mean()) ** 2))

    r2_obs = r2_of(y)
    adj_r2 = 1 - (1 - r2_obs) * (n - 1) / (n - p)

    rng = np.random.default_rng(seed)
    null_r2 = np.empty(n_perm)
    for i in range(n_perm):
        null_r2[i] = r2_of(y[rng.permutation(n)])
    p_value = float((np.sum(null_r2 >= r2_obs) + 1) / (n_perm + 1))

    # k-fold CV R² (averaged over several shuffles for stability). Uses its
    # own RNG so the value does not depend on the number of permutations above.
    rng_cv = np.random.default_rng(seed + 1)
    cv_scores = []
    for rep in range(5):
        order = rng_cv.permutation(n)
        folds = np.array_split(order, n_folds)
        sse = 0.0
        for fold in folds:
            test = np.zeros(n, dtype=bool)
            test[fold] = True
            train = ~test
            beta = np.linalg.lstsq(X[train], y[train], rcond=None)[0]
            sse += float(np.sum((y[test] - X[test] @ beta) ** 2))
        cv_scores.append(1 - sse / ss_total)
    cv_r2 = float(np.mean(cv_scores))

    print(f"  Observed R²            : {r2_obs:.4f}  (p={p} params, n={n})")
    print(f"  Adjusted R²            : {adj_r2:.4f}")
    print(f"  Permutation null R²    : mean={null_r2.mean():.4f}, 95th pct={np.percentile(null_r2,95):.4f}, max={null_r2.max():.4f}")
    print(f"  Permutation p-value    : {p_value:.4g}")
    print(f"  {n_folds}-fold CV R² (5 reps) : {cv_r2:.4f}")

    pd.DataFrame([
        dict(metric="observed_r2", value=round(r2_obs, 4)),
        dict(metric="adjusted_r2", value=round(adj_r2, 4)),
        dict(metric="n_params", value=p),
        dict(metric="perm_null_mean_r2", value=round(float(null_r2.mean()), 4)),
        dict(metric="perm_null_p95_r2", value=round(float(np.percentile(null_r2, 95)), 4)),
        dict(metric="perm_null_max_r2", value=round(float(null_r2.max()), 4)),
        dict(metric="perm_p_value", value=round(p_value, 6)),
        dict(metric="kfold_cv_r2", value=round(cv_r2, 4)),
        dict(metric="n_perm", value=n_perm),
        dict(metric="n_folds", value=n_folds),
    ]).to_csv(OUT / "r2_permutation_null.csv", index=False)
    print("  saved r2_permutation_null.csv")


def main():
    print("=" * 70)
    print("  ROBUSTNESS CHECKS")
    print("=" * 70)

    fix1_clustered_bootstrap()
    fix5_price_range()
    fix2_weighted_decomposition()
    fix3_anova()
    fix4_size_horizon_confound()
    fix6_permutation_and_cv()

    print("\n" + "=" * 70)
    print(f"  DONE — outputs in {OUT}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
