"""Reviewer-requested diagnostics derived from existing calibration outputs.

This script is intentionally lightweight: it does not re-fit trade-level
calibration models. It summarizes first-stage intercepts, cell sizes, a
decomposition that separates the common trade-size main effect, and simple
out-of-sample checks for the cell-level decomposition.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as fdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration import bootstrap_whale_effect, cluster_robust_slope_se, fit_logistic
from src.classify import get_group
from src.config import BIN_LABELS, CELL_MIN, DATE_CUTOFF, DOMAINS, KALSHI_MARKETS, KALSHI_TRADES, OUTPUT_DIR, SIZE_LABELS
from src.pipeline import load_kalshi_trades, size_bin_sql, time_bin_sql


KALSHI_OUT = OUTPUT_DIR / "kalshi"
OUT = OUTPUT_DIR / "revision"
OUT.mkdir(parents=True, exist_ok=True)


def _one_hot(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    levels = sorted(pd.unique(values))
    if len(levels) <= 1:
        return np.empty((len(values), 0)), levels
    mat = np.column_stack([(values.to_numpy() == level).astype(float) for level in levels[1:]])
    return mat, levels


def _interaction(a: pd.Series, b: pd.Series) -> np.ndarray:
    a_mat, _ = _one_hot(a)
    b_mat, _ = _one_hot(b)
    cols = []
    for i in range(a_mat.shape[1]):
        for j in range(b_mat.shape[1]):
            cols.append(a_mat[:, i] * b_mat[:, j])
    return np.column_stack(cols) if cols else np.empty((len(a), 0))


def _ss_res(y: np.ndarray, x: np.ndarray) -> float:
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return float(np.sum((y - x @ beta) ** 2))


def write_intercept_and_sample_summaries(cal: pd.DataFrame) -> None:
    intercept_summary = (
        cal.groupby("domain")["intercept_a"]
        .agg(
            mean="mean",
            median="median",
            mean_abs=lambda x: float(np.mean(np.abs(x))),
            max_abs=lambda x: float(np.max(np.abs(x))),
        )
        .reset_index()
        .round(4)
    )
    intercept_summary.to_csv(OUT / "intercept_summary_by_domain.csv", index=False)

    cell_summary = (
        cal.groupby("domain")["n_trades"]
        .agg(cells="count", min="min", median="median", mean="mean", max="max")
        .reset_index()
    )
    for col in ["min", "median", "mean", "max"]:
        cell_summary[col] = cell_summary[col].round(0).astype(int)
    cell_summary.to_csv(OUT / "cell_size_summary_by_domain.csv", index=False)

    cols = [
        "domain",
        "time_bin",
        "size_bin",
        "n_trades",
        "slope_b",
        "intercept_a",
        "slope_stderr",
    ]
    cal[cols].to_csv(OUT / "cell_intercepts_216.csv", index=False)


def write_decomposition_with_size_main(cal: pd.DataFrame) -> None:
    y = cal["slope_b"].to_numpy()
    n = len(y)
    intercept = np.ones((n, 1))
    ss_total = float(np.sum((y - y.mean()) ** 2))

    terms = [
        ("time_bin", _one_hot(cal["time_bin"])[0]),
        ("domain", _one_hot(cal["domain"])[0]),
        ("size_bin", _one_hot(cal["size_bin"])[0]),
        ("time:domain", _interaction(cal["time_bin"], cal["domain"])),
        ("domain:size", _interaction(cal["domain"], cal["size_bin"])),
    ]

    x = intercept
    ss_prev = _ss_res(y, x)
    rows = []
    for term, matrix in terms:
        x = np.hstack([x, matrix])
        ss_after = _ss_res(y, x)
        ss_term = ss_prev - ss_after
        rows.append(
            {
                "term": term,
                "df": matrix.shape[1],
                "sum_sq": ss_term,
                "marginal_r2": ss_term / ss_total,
                "cumulative_r2": 1 - ss_after / ss_total,
            }
        )
        ss_prev = ss_after

    ss_resid = ss_prev
    df_resid = n - x.shape[1]
    for row in rows:
        ms_term = row["sum_sq"] / row["df"]
        ms_resid = ss_resid / df_resid
        row["F_statistic"] = ms_term / ms_resid
        row["p_value"] = fdist.sf(row["F_statistic"], row["df"], df_resid)

    rows.append(
        {
            "term": "residual",
            "df": df_resid,
            "sum_sq": ss_resid,
            "marginal_r2": ss_resid / ss_total,
            "cumulative_r2": 1.0,
            "F_statistic": np.nan,
            "p_value": np.nan,
        }
    )
    pd.DataFrame(rows).round(6).to_csv(OUT / "decomposition_with_size_main.csv", index=False)

    loo_sse = 0.0
    for i in range(n):
        train = np.ones(n, dtype=bool)
        train[i] = False
        beta = np.linalg.lstsq(x[train], y[train], rcond=None)[0]
        pred = float(x[i] @ beta)
        loo_sse += (y[i] - pred) ** 2

    cv_rows = [
        {
            "check": "leave_one_cell_out",
            "sse": loo_sse,
            "r2": 1 - loo_sse / ss_total,
        }
    ]
    for group in ["domain", "time_bin", "size_bin"]:
        sse = 0.0
        for level in pd.unique(cal[group]):
            test = (cal[group] == level).to_numpy()
            train = ~test
            beta = np.linalg.lstsq(x[train], y[train], rcond=None)[0]
            pred = x[test] @ beta
            sse += float(np.sum((y[test] - pred) ** 2))
        cv_rows.append({"check": f"leave_{group}_out", "sse": sse, "r2": 1 - sse / ss_total})
    pd.DataFrame(cv_rows).round(6).to_csv(OUT / "decomposition_cv_checks.csv", index=False)


def write_flexible_calibration_diagnostics() -> None:
    import duckdb
    from sklearn.isotonic import IsotonicRegression

    conn = duckdb.connect()
    try:
        raw = load_kalshi_trades(conn)
    finally:
        conn.close()

    bins = np.arange(5, 106, 10)
    raw["price_bin"] = pd.cut(raw["yes_price"], bins=bins, right=False, include_lowest=True)

    reliability_rows = []
    for (domain, price_bin), cell in raw.groupby(["domain", "price_bin"], observed=True):
        weights = cell["total_contracts"].astype(float).to_numpy()
        if weights.sum() <= 0:
            continue
        reliability_rows.append(
            {
                "domain": domain,
                "price_bin": str(price_bin),
                "n_trades": int(cell["n_trades"].sum()),
                "mean_price": float(np.average(cell["yes_price"], weights=weights)),
                "observed_rate": float(np.average(cell["is_yes"], weights=weights)),
            }
        )
    pd.DataFrame(reliability_rows).round(6).to_csv(OUT / "binned_reliability_by_domain.csv", index=False)

    grid = np.linspace(0.05, 0.95, 19)
    isotonic_rows = []
    for domain in DOMAINS:
        cell = raw[raw["domain"] == domain]
        if cell.empty:
            continue
        weights = cell["total_contracts"].astype(float).to_numpy()
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(cell["yes_price"].astype(float).to_numpy() / 100.0, cell["is_yes"].astype(float).to_numpy(), sample_weight=weights)
        preds = model.predict(grid)
        for price, pred in zip(grid, preds):
            isotonic_rows.append({"domain": domain, "price": price, "isotonic_probability": pred})
    pd.DataFrame(isotonic_rows).round(6).to_csv(OUT / "isotonic_calibration_by_domain.csv", index=False)


def write_nonparametric_metrics() -> None:
    """Nonparametric calibration metrics (addresses AE: 'consider more flexible
    nonparametric approaches ... a much richer picture').

    Computes, per domain and per domain x coarse-horizon, the expected and
    maximum calibration error (ECE/MCE), the Brier score and its Murphy
    decomposition (reliability - resolution + uncertainty), and an isotonic
    fit. Also ranks domains by nonparametric ECE vs the slope-based summary to
    show the slope captures the same ordering.
    """
    import duckdb
    from sklearn.isotonic import IsotonicRegression

    conn = duckdb.connect()
    try:
        raw = load_kalshi_trades(conn)
    finally:
        conn.close()
    raw["coarse_h"] = pd.cut(
        raw["tbin"].astype(int), bins=[-1, 4, 6, 8],
        labels=["<24h", "1d-1w", ">1w"])

    edges = np.linspace(5, 95, 10)  # 9 price bins on the 5-95 range

    def metrics(cell: pd.DataFrame) -> dict:
        p = cell["yes_price"].to_numpy(float) / 100.0
        y = cell["is_yes"].to_numpy(float)
        w = cell["total_contracts"].to_numpy(float)
        W = w.sum()
        if W <= 0:
            return {}
        obar = float(np.average(y, weights=w))
        brier = float(np.average((y - p) ** 2, weights=w))
        # bin by raw price
        binidx = np.clip(np.digitize(cell["yes_price"].to_numpy(float), edges) - 1, 0, len(edges) - 2)
        ece = mce = reliability = resolution = 0.0
        for k in range(len(edges) - 1):
            m = binidx == k
            wk = w[m].sum()
            if wk <= 0:
                continue
            pk = np.average(p[m], weights=w[m])
            ok = np.average(y[m], weights=w[m])
            gap = abs(ok - pk)
            ece += (wk / W) * gap
            mce = max(mce, gap)
            reliability += (wk / W) * (pk - ok) ** 2
            resolution += (wk / W) * (ok - obar) ** 2
        return dict(n_trades=int(cell["n_trades"].sum()), base_rate=round(obar, 4),
                    ECE=round(ece, 4), MCE=round(mce, 4), Brier=round(brier, 4),
                    reliability=round(reliability, 4), resolution=round(resolution, 4),
                    uncertainty=round(obar * (1 - obar), 4))

    # per domain
    dom_rows = []
    for d in DOMAINS:
        m = metrics(raw[raw["domain"] == d])
        if m:
            dom_rows.append(dict(domain=d, **m))
    dom = pd.DataFrame(dom_rows)
    dom.to_csv(OUT / "calibration_metrics_by_domain.csv", index=False)

    # per domain x coarse horizon
    dh_rows = []
    for (d, h), cell in raw.groupby(["domain", "coarse_h"], observed=True):
        m = metrics(cell)
        if m:
            dh_rows.append(dict(domain=d, coarse_horizon=str(h), **m))
    pd.DataFrame(dh_rows).to_csv(OUT / "nonparametric_calibration_by_domain_horizon.csv", index=False)

    # isotonic by domain x coarse horizon
    grid = np.linspace(0.05, 0.95, 19)
    iso_rows = []
    for (d, h), cell in raw.groupby(["domain", "coarse_h"], observed=True):
        if cell.empty:
            continue
        w = cell["total_contracts"].to_numpy(float)
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(cell["yes_price"].to_numpy(float) / 100.0, cell["is_yes"].to_numpy(float), sample_weight=w)
        for price, pred in zip(grid, model.predict(grid)):
            iso_rows.append(dict(domain=d, coarse_horizon=str(h), price=round(price, 3),
                                 isotonic_probability=round(float(pred), 4)))
    pd.DataFrame(iso_rows).to_csv(OUT / "isotonic_by_domain_horizon.csv", index=False)

    # ranking: slope-based vs ECE-based
    cal = pd.read_csv(KALSHI_OUT / "calibration_matrix.csv")
    slope_dev = cal.assign(absdev=(cal["slope_b"] - 1.0).abs()).groupby("domain")["absdev"].mean()
    rank = dom.set_index("domain")[["ECE"]].copy()
    rank["mean_abs_slope_dev"] = slope_dev
    rank["rank_by_ECE"] = rank["ECE"].rank(ascending=False).astype(int)
    rank["rank_by_slope"] = rank["mean_abs_slope_dev"].rank(ascending=False).astype(int)
    rank.reset_index().round(4).to_csv(OUT / "calibration_metrics_comparison.csv", index=False)
    print("  calibration_metrics_by_domain.csv + nonparametric_* + comparison")
    print(dom.to_string(index=False))


def write_burst_aggregation_robustness() -> None:
    import duckdb

    markets = str(KALSHI_MARKETS).replace("\\", "/")
    trades = str(KALSHI_TRADES).replace("\\", "/")
    tb = time_bin_sql()
    sb = """
        CASE
            WHEN burst_count = 1 THEN 0
            WHEN burst_count <= 10 THEN 1
            WHEN burst_count <= 100 THEN 2
            ELSE 3
        END
    """

    conn = duckdb.connect()
    try:
        raw = conn.execute(
            f"""
            WITH resolved AS (
                SELECT ticker, event_ticker, result, close_time
                FROM '{markets}/*.parquet'
                WHERE status='finalized' AND result IN ('yes','no')
            ),
            trade_data AS (
                SELECT
                    t.ticker,
                    t.created_time,
                    COALESCE(t.taker_side, '') AS taker_side,
                    t.yes_price,
                    t.count AS trade_count,
                    CASE WHEN m.result='yes' THEN 1 ELSE 0 END AS is_yes,
                    regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) AS cat_prefix,
                    EXTRACT(EPOCH FROM (m.close_time - t.created_time))/3600.0 AS hours_to_close
                FROM '{trades}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
                WHERE t.created_time <= TIMESTAMP '{DATE_CUTOFF}'
                  AND m.close_time > t.created_time
                  AND t.yes_price BETWEEN 5 AND 95
            ),
            market_counts AS (
                SELECT ticker FROM trade_data GROUP BY ticker HAVING COUNT(*) >= 10
            ),
            burst AS (
                SELECT
                    td.ticker,
                    td.created_time,
                    td.taker_side,
                    td.yes_price,
                    td.is_yes,
                    td.cat_prefix,
                    td.hours_to_close,
                    SUM(td.trade_count) AS burst_count,
                    COUNT(*) AS raw_trade_count
                FROM trade_data td
                INNER JOIN market_counts mc ON td.ticker = mc.ticker
                GROUP BY
                    td.ticker,
                    td.created_time,
                    td.taker_side,
                    td.yes_price,
                    td.is_yes,
                    td.cat_prefix,
                    td.hours_to_close
            )
            SELECT
                cat_prefix,
                ({tb}) AS tbin,
                ({sb}) AS sbin,
                yes_price,
                is_yes,
                SUM(burst_count) AS total_contracts,
                COUNT(*) AS n_trades,
                SUM(raw_trade_count) AS raw_trade_count
            FROM burst
            WHERE ({tb}) >= 0
            GROUP BY cat_prefix, ({tb}), ({sb}), yes_price, is_yes
            """
        ).df()
    finally:
        conn.close()

    raw["domain"] = raw["cat_prefix"].apply(get_group)
    raw = raw[raw["domain"].isin(DOMAINS)].copy()

    rows = []
    for (domain, tbin, sbin), cell in raw.groupby(["domain", "tbin", "sbin"]):
        n_t = int(cell["n_trades"].sum())
        if n_t < CELL_MIN:
            continue
        result = fit_logistic(
            cell["yes_price"].astype(float).to_numpy(),
            cell["is_yes"].astype(float).to_numpy(),
            cell["total_contracts"].astype(float).to_numpy(),
        )
        if result is None:
            continue
        b, a, se = result
        rows.append(
            {
                "domain": domain,
                "time_bin": BIN_LABELS[int(tbin)],
                "time_bin_order": int(tbin) + 1,
                "size_bin": SIZE_LABELS[int(sbin)],
                "size_bin_order": int(sbin) + 1,
                "n_bursts": n_t,
                "raw_trade_count": int(cell["raw_trade_count"].sum()),
                "slope_b": b,
                "intercept_a": a,
                "slope_stderr": se,
            }
        )

    cal = pd.DataFrame(rows)
    cal.to_csv(OUT / "burst_aggregated_calibration_matrix.csv", index=False)

    boot_rows = []
    for domain in ["Politics", "Sports"]:
        obs_diff, ci_lo, ci_hi = bootstrap_whale_effect(cal.rename(columns={"n_bursts": "n_trades"}), domain)
        boot_rows.append(
            {
                "domain": domain,
                "delta_large_minus_single": obs_diff,
                "ci_2_5": ci_lo,
                "ci_97_5": ci_hi,
            }
        )
    pd.DataFrame(boot_rows).round(6).to_csv(OUT / "burst_aggregated_whale_effect.csv", index=False)


def write_intercept_and_ice() -> None:
    """Make the intercept a first-class object (addresses AE / Reviewer 2 #7:
    slope-only does not capture calibration; systematic bias via the intercept
    is also miscalibration).

    (1) Decomposes the first-stage intercept a(d,tau,s) in the same five-term
        Type I framework as the slope, so directional-bias structure is visible.
    (2) Computes an Integrated Calibration Error (ICE) per cell from the FULL
        two-parameter curve sigma(a + b*logit(p)), weighted by the cell's
        contract-weighted price distribution, and ranks domains by ICE to show
        the slope-based domain ranking survives when the intercept is included.
    """
    import duckdb
    from scipy.special import expit, logit as _logit
    from src.calibration import decompose

    # ── (1) intercept decomposition (reuse decompose on intercept_a) ──
    cal = pd.read_csv(KALSHI_OUT / "calibration_matrix.csv").copy()
    cal_i = cal.copy()
    cal_i["slope_b"] = cal_i["intercept_a"]          # decompose operates on slope_b
    dec_i = decompose(cal_i)
    theta = dec_i["slope_b"].to_numpy()
    sst = float(np.sum((theta - theta.mean()) ** 2))
    fitc = np.zeros(len(theta)); prev = 0.0; rows = []
    for comp in ["mu", "alpha", "kappa", "beta", "gamma"]:
        fitc = fitc + dec_i[comp].to_numpy()
        cum = float(np.sum((fitc - theta.mean()) ** 2) / sst)
        rows.append(dict(component=comp, marginal_r2=round(cum - prev, 4), cumulative_r2=round(cum, 4)))
        prev = cum
    rows.append(dict(component="residual", marginal_r2=round(1 - prev, 4), cumulative_r2=1.0))
    pd.DataFrame(rows).to_csv(OUT / "intercept_decomposition.csv", index=False)

    # ── (2) ICE per cell from the full (a, b) curve ──
    conn = duckdb.connect()
    try:
        raw = load_kalshi_trades(conn)
    finally:
        conn.close()
    # price distribution per cell (contract-weighted)
    price_hist = (raw.groupby(["domain", "tbin", "sbin", "yes_price"])["total_contracts"]
                  .sum().reset_index())
    price_hist["time_bin"] = price_hist["tbin"].apply(lambda i: BIN_LABELS[int(i)])
    price_hist["size_bin"] = price_hist["sbin"].apply(lambda i: SIZE_LABELS[int(i)])

    ab = cal.set_index(["domain", "time_bin", "size_bin"])[["slope_b", "intercept_a"]]
    ice_rows = []
    for (d, tb, sb), g in price_hist.groupby(["domain", "time_bin", "size_bin"]):
        key = (d, tb, sb)
        if key not in ab.index:
            continue
        b, a = float(ab.loc[key, "slope_b"]), float(ab.loc[key, "intercept_a"])
        p = np.clip(g["yes_price"].to_numpy(float) / 100.0, 0.01, 0.99)
        w = g["total_contracts"].to_numpy(float)
        w = w / w.sum()
        lp = _logit(p)
        p_full = expit(a + b * lp)            # full two-parameter recalibration
        p_slope = expit(b * lp)               # slope-only (intercept forced to 0)
        ice_full = float(np.sum(w * np.abs(p_full - p)))
        ice_slope = float(np.sum(w * np.abs(p_slope - p)))
        ice_rows.append(dict(domain=d, time_bin=tb, size_bin=sb,
                             ice_full=ice_full, ice_slope_only=ice_slope))
    ice = pd.DataFrame(ice_rows).merge(
        cal[["domain", "time_bin", "size_bin", "n_trades"]],
        on=["domain", "time_bin", "size_bin"], how="left")
    ice.round(6).to_csv(OUT / "calibration_error_by_cell.csv", index=False)

    # domain ranking: slope-based (mean |b-1|) vs ICE-based (full curve).
    # Volume-weighted ICE is the representative aggregate (equal-weighting cells
    # overweights thin, high-intercept cells); both are reported.
    slope_dev = cal.assign(absdev=(cal["slope_b"] - 1.0).abs()).groupby("domain")["absdev"].mean()
    ice_eq = ice.groupby("domain")["ice_full"].mean()
    ice_vw = ice.groupby("domain").apply(
        lambda g: np.average(g["ice_full"], weights=g["n_trades"]), include_groups=False)
    rank = pd.DataFrame({
        "mean_abs_slope_dev": slope_dev,
        "ICE_full_equal_wt": ice_eq,
        "ICE_full_volume_wt": ice_vw,
    })
    rank["rank_by_slope"] = rank["mean_abs_slope_dev"].rank(ascending=False).astype(int)
    rank["rank_by_ICE_volwt"] = rank["ICE_full_volume_wt"].rank(ascending=False).astype(int)
    rank = rank.reset_index().round(4)
    rank.to_csv(OUT / "domain_ranking_slope_vs_ice.csv", index=False)
    print("  intercept_decomposition.csv, calibration_error_by_cell.csv, domain_ranking_slope_vs_ice.csv")
    print(rank.to_string(index=False))


def write_clustered_cell_se() -> None:
    """Event-clustered (CR1) standard errors for the 216 first-stage slopes.

    Re-queries trade-level data keeping ``event_ticker`` (lost in the main
    pre-aggregated pipeline), then for each domain x time x size cell fits the
    logistic slope and computes both the naive Fisher SE and an event-clustered
    sandwich SE. The clustered SE feeds the Bayesian measurement-error model.
    """
    import duckdb

    markets = str(KALSHI_MARKETS).replace("\\", "/")
    trades = str(KALSHI_TRADES).replace("\\", "/")
    tb = time_bin_sql()
    sb = size_bin_sql()

    conn = duckdb.connect()
    try:
        raw = conn.execute(
            f"""
            WITH resolved AS (
                SELECT ticker, event_ticker, result, close_time
                FROM '{markets}/*.parquet'
                WHERE status='finalized' AND result IN ('yes','no')
            ),
            trade_data AS (
                SELECT t.yes_price, t.count AS trade_count,
                       CASE WHEN m.result='yes' THEN 1 ELSE 0 END AS is_yes,
                       m.event_ticker,
                       regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) AS cat_prefix,
                       EXTRACT(EPOCH FROM (m.close_time - t.created_time))/3600.0 AS hours_to_close,
                       ({sb}) AS sbin, m.ticker
                FROM '{trades}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
                WHERE t.created_time <= TIMESTAMP '{DATE_CUTOFF}'
                  AND m.close_time > t.created_time
                  AND t.yes_price BETWEEN 5 AND 95
            ),
            market_counts AS (
                SELECT ticker FROM trade_data GROUP BY ticker HAVING COUNT(*) >= 10
            )
            SELECT td.cat_prefix, ({tb}) AS tbin, td.sbin, td.event_ticker,
                   td.yes_price, td.is_yes,
                   SUM(td.trade_count) AS total_contracts, COUNT(*) AS n_trades
            FROM trade_data td
            INNER JOIN market_counts mc ON td.ticker = mc.ticker
            WHERE ({tb}) >= 0 AND td.sbin >= 0
            GROUP BY td.cat_prefix, ({tb}), td.sbin, td.event_ticker, td.yes_price, td.is_yes
            """
        ).df()
    finally:
        conn.close()

    raw["domain"] = raw["cat_prefix"].apply(get_group)
    raw = raw[raw["domain"].isin(DOMAINS)].copy()

    rows = []
    for (domain, tbin, sbin), cell in raw.groupby(["domain", "tbin", "sbin"]):
        n_t = int(cell["n_trades"].sum())
        if n_t < CELL_MIN:
            continue
        prices = cell["yes_price"].to_numpy(float)
        outcomes = cell["is_yes"].to_numpy(float)
        weights = cell["total_contracts"].to_numpy(float)
        res = fit_logistic(prices, outcomes, weights)
        if res is None:
            continue
        b, a, naive_se = res
        clustered_se = cluster_robust_slope_se(
            prices, outcomes, weights, cell["event_ticker"].to_numpy(), a, b
        )
        rows.append(
            {
                "domain": domain,
                "time_bin": BIN_LABELS[int(tbin)],
                "time_bin_order": int(tbin) + 1,
                "size_bin": SIZE_LABELS[int(sbin)],
                "size_bin_order": int(sbin) + 1,
                "n_trades": n_t,
                "n_events": int(cell["event_ticker"].nunique()),
                "slope_b": b,
                "intercept_a": a,
                "naive_se": naive_se,
                "event_clustered_se": clustered_se,
                "se_inflation": clustered_se / naive_se if naive_se and naive_se > 0 else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out.round(6).to_csv(OUT / "cell_clustered_se.csv", index=False)
    med_infl = out["se_inflation"].median()
    print(
        f"  cell_clustered_se.csv: {len(out)} cells, median SE inflation "
        f"(clustered/naive) = {med_infl:.1f}x"
    )


def write_fee_and_size_diagnostics() -> None:
    """Reviewer 1 diagnostics: (a) Kalshi trade-size distribution by domain to
    back the claim that political bets are larger; (b) a walk-the-book metric
    quantifying how often 'Large' executions are multi-price sweeps of the same
    order; (c) a fee-burden proxy (Kalshi fees peak near 50c).
    """
    import duckdb

    markets = str(KALSHI_MARKETS).replace("\\", "/")
    trades = str(KALSHI_TRADES).replace("\\", "/")

    conn = duckdb.connect()
    try:
        # (a) trade-size frequency table by (prefix, count) -> exact weighted
        # quantiles per domain in pandas (medians cannot be averaged across
        # prefixes). Also fee-burden proxy weighted by trade count.
        size_freq = conn.execute(
            f"""
            WITH resolved AS (
                SELECT ticker, event_ticker FROM '{markets}/*.parquet'
                WHERE status='finalized' AND result IN ('yes','no')
            )
            SELECT regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) AS cat_prefix,
                   t.count AS trade_size, COUNT(*) AS freq,
                   AVG((t.yes_price/100.0)*(1-t.yes_price/100.0)) AS fee_proxy
            FROM '{trades}/*.parquet' t
            INNER JOIN resolved m ON t.ticker = m.ticker
            WHERE t.created_time <= TIMESTAMP '{DATE_CUTOFF}'
              AND t.yes_price BETWEEN 5 AND 95
            GROUP BY cat_prefix, t.count
            """
        ).df()

        # (b) walk-the-book: bursts = (ticker, created_time, taker_side);
        # of bursts summing to >100 contracts (Large), what fraction sweep >1 price
        walk = conn.execute(
            f"""
            WITH resolved AS (
                SELECT ticker, event_ticker FROM '{markets}/*.parquet'
                WHERE status='finalized' AND result IN ('yes','no')
            ),
            bursts AS (
                SELECT regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) AS cat_prefix,
                       t.ticker, t.created_time, COALESCE(t.taker_side,'') AS side,
                       SUM(t.count) AS burst_contracts,
                       COUNT(DISTINCT t.yes_price) AS n_prices
                FROM '{trades}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
                WHERE t.created_time <= TIMESTAMP '{DATE_CUTOFF}'
                  AND t.yes_price BETWEEN 5 AND 95
                GROUP BY cat_prefix, t.ticker, t.created_time, side
            )
            SELECT cat_prefix,
                   SUM(CASE WHEN burst_contracts > 100 THEN 1 ELSE 0 END) AS n_large_bursts,
                   SUM(CASE WHEN burst_contracts > 100 AND n_prices > 1 THEN 1 ELSE 0 END)::DOUBLE
                       / NULLIF(SUM(CASE WHEN burst_contracts > 100 THEN 1 ELSE 0 END),0) AS frac_large_bursts_multiprice
            FROM bursts
            GROUP BY cat_prefix
            """
        ).df()
    finally:
        conn.close()

    for df in (size_freq, walk):
        df["domain"] = df["cat_prefix"].apply(get_group)
    size_freq = size_freq[size_freq["domain"].isin(DOMAINS)]
    walk = walk[walk["domain"].isin(DOMAINS)]

    def wquantile(sizes, freqs, q):
        order = np.argsort(sizes)
        s, f = sizes[order], freqs[order]
        cum = np.cumsum(f) - 0.5 * f
        return float(np.interp(q * f.sum(), cum, s))

    size_rows = []
    for d, g in size_freq.groupby("domain"):
        sizes = g["trade_size"].to_numpy(float)
        freqs = g["freq"].to_numpy(float)
        N = freqs.sum()
        vol = (sizes * freqs).sum()
        large = freqs[sizes > 100].sum()
        vol_large = (sizes[sizes > 100] * freqs[sizes > 100]).sum()
        size_rows.append(dict(
            domain=d, n_trades=int(N),
            median_size=round(wquantile(sizes, freqs, 0.50), 1),
            p90_size=round(wquantile(sizes, freqs, 0.90), 1),
            p99_size=round(wquantile(sizes, freqs, 0.99), 1),
            mean_size=round(vol / N, 1),
            frac_large_trades=round(large / N, 4),
            frac_vol_from_large=round(vol_large / vol, 4),
            mean_fee_burden_proxy=round(float(np.average(g["fee_proxy"], weights=freqs)), 4),
        ))
    size_by_domain = pd.DataFrame(size_rows)
    size_by_domain.to_csv(OUT / "tradesize_distribution_kalshi.csv", index=False)

    walk["w"] = walk["frac_large_bursts_multiprice"] * walk["n_large_bursts"]
    walk_by_domain = (walk.groupby("domain")
                      .agg(n_large_bursts=("n_large_bursts", "sum"), w=("w", "sum"))
                      .reset_index())
    walk_by_domain["frac_large_bursts_multiprice"] = (
        walk_by_domain["w"] / walk_by_domain["n_large_bursts"]).round(4)
    walk_by_domain = walk_by_domain.drop(columns="w")
    walk_by_domain.to_csv(OUT / "burst_walkbook_flags.csv", index=False)
    print("  tradesize_distribution_kalshi.csv, burst_walkbook_flags.csv")
    print(size_by_domain[["domain", "median_size", "p99_size", "frac_vol_from_large", "mean_fee_burden_proxy"]].to_string(index=False))
    print(walk_by_domain.to_string(index=False))


def write_availability_restricted_slopes() -> None:
    import duckdb

    markets = str(KALSHI_MARKETS).replace("\\", "/")
    trades = str(KALSHI_TRADES).replace("\\", "/")
    tb = time_bin_sql()

    conn = duckdb.connect()
    try:
        raw = conn.execute(
            f"""
            WITH resolved AS (
                SELECT ticker, event_ticker, result, open_time, close_time
                FROM '{markets}/*.parquet'
                WHERE status='finalized'
                  AND result IN ('yes','no')
                  AND open_time IS NOT NULL
                  AND EXTRACT(EPOCH FROM (close_time - open_time))/3600.0 >= 24 * 30
            ),
            trade_data AS (
                SELECT
                    t.yes_price,
                    t.count AS trade_count,
                    CASE WHEN m.result='yes' THEN 1 ELSE 0 END AS is_yes,
                    regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) AS cat_prefix,
                    EXTRACT(EPOCH FROM (m.close_time - t.created_time))/3600.0 AS hours_to_close,
                    m.ticker
                FROM '{trades}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
                WHERE t.created_time <= TIMESTAMP '{DATE_CUTOFF}'
                  AND m.close_time > t.created_time
                  AND t.yes_price BETWEEN 5 AND 95
            ),
            market_counts AS (
                SELECT ticker FROM trade_data GROUP BY ticker HAVING COUNT(*) >= 10
            )
            SELECT
                cat_prefix,
                ({tb}) AS tbin,
                yes_price,
                is_yes,
                SUM(trade_count) AS total_contracts,
                COUNT(*) AS n_trades
            FROM trade_data td
            INNER JOIN market_counts mc ON td.ticker = mc.ticker
            WHERE ({tb}) >= 0
            GROUP BY cat_prefix, ({tb}), yes_price, is_yes
            """
        ).df()
    finally:
        conn.close()

    raw["domain"] = raw["cat_prefix"].apply(get_group)
    raw = raw[raw["domain"].isin(DOMAINS)].copy()

    rows = []
    for (domain, tbin), cell in raw.groupby(["domain", "tbin"]):
        n_t = int(cell["n_trades"].sum())
        if n_t < CELL_MIN:
            continue
        result = fit_logistic(
            cell["yes_price"].astype(float).to_numpy(),
            cell["is_yes"].astype(float).to_numpy(),
            cell["total_contracts"].astype(float).to_numpy(),
        )
        if result is None:
            continue
        b, a, se = result
        rows.append(
            {
                "domain": domain,
                "time_bin": BIN_LABELS[int(tbin)],
                "time_bin_order": int(tbin) + 1,
                "n_trades": n_t,
                "slope_b": b,
                "intercept_a": a,
                "slope_stderr": se,
            }
        )

    restricted = pd.DataFrame(rows)
    baseline = pd.read_csv(KALSHI_OUT / "calibration_slopes_by_domain_time.csv")
    comparison = restricted.merge(
        baseline[["domain", "time_bin", "slope_b", "n_trades"]],
        on=["domain", "time_bin"],
        suffixes=("_availability_restricted", "_baseline"),
    )
    comparison["slope_difference"] = comparison["slope_b_availability_restricted"] - comparison["slope_b_baseline"]
    comparison.round(6).to_csv(OUT / "availability_restricted_domain_time.csv", index=False)


def write_balanced_availability() -> None:
    """Balanced-panel availability check (Reviewer 1: show results when only
    contracts available across the horizon range are included).

    Stricter than the >=30-day-duration proxy: restricts to contracts that are
    actually *traded* in at least six of the nine horizon bins, so each
    contributing contract spans most of the horizon range, and re-estimates the
    domain-by-horizon slopes on this balanced subset.
    """
    import duckdb

    markets = str(KALSHI_MARKETS).replace("\\", "/")
    trades = str(KALSHI_TRADES).replace("\\", "/")
    tb = time_bin_sql()

    conn = duckdb.connect()
    try:
        raw = conn.execute(
            f"""
            WITH resolved AS (
                SELECT ticker, event_ticker, result, close_time
                FROM '{markets}/*.parquet'
                WHERE status='finalized' AND result IN ('yes','no')
            ),
            trade_data AS (
                SELECT t.yes_price, t.count AS trade_count, t.ticker,
                       CASE WHEN m.result='yes' THEN 1 ELSE 0 END AS is_yes,
                       regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) AS cat_prefix,
                       EXTRACT(EPOCH FROM (m.close_time - t.created_time))/3600.0 AS hours_to_close
                FROM '{trades}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
                WHERE t.created_time <= TIMESTAMP '{DATE_CUTOFF}'
                  AND m.close_time > t.created_time
                  AND t.yes_price BETWEEN 5 AND 95
            ),
            tagged AS (
                SELECT *, ({tb}) AS tbin FROM trade_data WHERE ({tb}) >= 0
            ),
            balanced AS (
                SELECT ticker FROM tagged
                GROUP BY ticker
                HAVING COUNT(*) >= 10 AND COUNT(DISTINCT tbin) >= 6
            )
            SELECT td.cat_prefix, td.tbin, td.yes_price, td.is_yes,
                   SUM(td.trade_count) AS total_contracts, COUNT(*) AS n_trades
            FROM tagged td INNER JOIN balanced b ON td.ticker = b.ticker
            GROUP BY td.cat_prefix, td.tbin, td.yes_price, td.is_yes
            """
        ).df()
    finally:
        conn.close()

    raw["domain"] = raw["cat_prefix"].apply(get_group)
    raw = raw[raw["domain"].isin(DOMAINS)].copy()

    rows = []
    for (domain, tbin), cell in raw.groupby(["domain", "tbin"]):
        n_t = int(cell["n_trades"].sum())
        if n_t < CELL_MIN:
            continue
        res = fit_logistic(cell["yes_price"].to_numpy(float),
                           cell["is_yes"].to_numpy(float),
                           cell["total_contracts"].to_numpy(float))
        if res is None:
            continue
        b, a, se = res
        rows.append(dict(domain=domain, time_bin=BIN_LABELS[int(tbin)],
                         time_bin_order=int(tbin) + 1, n_trades=n_t,
                         slope_b=b, intercept_a=a, slope_stderr=se))
    bal = pd.DataFrame(rows)
    baseline = pd.read_csv(KALSHI_OUT / "calibration_slopes_by_domain_time.csv")
    comp = bal.merge(baseline[["domain", "time_bin", "slope_b"]],
                     on=["domain", "time_bin"], suffixes=("_balanced", "_baseline"))
    comp["slope_difference"] = comp["slope_b_balanced"] - comp["slope_b_baseline"]
    comp.round(6).to_csv(OUT / "availability_balanced_panel.csv", index=False)
    # per-domain mean absolute change
    summ = comp.groupby("domain")["slope_difference"].agg(
        mean_diff="mean", mean_abs_diff=lambda x: float(np.mean(np.abs(x)))).round(4).reset_index()
    summ.to_csv(OUT / "availability_balanced_summary.csv", index=False)
    print("  availability_balanced_panel.csv + summary")
    print(summ.to_string(index=False))


def write_balanced_4bin_panel() -> None:
    """Exactly-balanced availability panel (Reviewer 1: contracts available for
    ALL time intervals). Requiring trades in all nine fine bins is infeasible, so
    the horizon is coarsened to four bins (<24h, 1d-1w, 1w-1mo, >1mo) and the
    panel is restricted to contracts traded in ALL FOUR bins; domain-by-coarse-bin
    slopes are then re-estimated on this fully balanced subset.
    """
    import duckdb

    markets = str(KALSHI_MARKETS).replace("\\", "/")
    trades = str(KALSHI_TRADES).replace("\\", "/")
    coarse = "CASE WHEN h < 24 THEN 0 WHEN h < 168 THEN 1 WHEN h < 720 THEN 2 ELSE 3 END"
    CB = {0: "<24h", 1: "1d-1w", 2: "1w-1mo", 3: ">1mo"}

    conn = duckdb.connect()
    try:
        raw = conn.execute(
            f"""
            WITH resolved AS (
                SELECT ticker, event_ticker, result, close_time FROM '{markets}/*.parquet'
                WHERE status='finalized' AND result IN ('yes','no')
            ),
            td AS (
                SELECT t.yes_price, t.count AS c,
                       CASE WHEN m.result='yes' THEN 1 ELSE 0 END AS is_yes,
                       m.ticker, regexp_extract(m.event_ticker,'^([A-Z0-9]+)',1) AS cat,
                       EXTRACT(EPOCH FROM (m.close_time - t.created_time))/3600.0 AS h
                FROM '{trades}/*.parquet' t INNER JOIN resolved m ON t.ticker=m.ticker
                WHERE t.created_time<=TIMESTAMP '{DATE_CUTOFF}' AND m.close_time>t.created_time
                      AND t.yes_price BETWEEN 5 AND 95
            ),
            tagged AS (SELECT *, ({coarse}) AS cb FROM td),
            mc AS (SELECT ticker FROM tagged GROUP BY ticker HAVING COUNT(*)>=10),
            bal AS (SELECT ticker FROM tagged t JOIN mc USING(ticker)
                    GROUP BY ticker HAVING COUNT(DISTINCT cb)=4)
            SELECT t.cat, t.cb, t.yes_price, t.is_yes, SUM(t.c) AS w, COUNT(*) AS n
            FROM tagged t JOIN bal USING(ticker)
            GROUP BY t.cat, t.cb, t.yes_price, t.is_yes
            """
        ).df()
        n_contracts = conn.execute(
            f"""
            WITH resolved AS (SELECT ticker,event_ticker,result,close_time FROM '{markets}/*.parquet'
                              WHERE status='finalized' AND result IN ('yes','no')),
            td AS (SELECT t.count,m.ticker,EXTRACT(EPOCH FROM (m.close_time-t.created_time))/3600.0 AS h
                   FROM '{trades}/*.parquet' t JOIN resolved m ON t.ticker=m.ticker
                   WHERE t.created_time<=TIMESTAMP '{DATE_CUTOFF}' AND m.close_time>t.created_time
                         AND t.yes_price BETWEEN 5 AND 95),
            tagged AS (SELECT ticker,({coarse}) AS cb FROM td),
            mc AS (SELECT ticker FROM tagged GROUP BY ticker HAVING COUNT(*)>=10)
            SELECT COUNT(*) FROM (SELECT ticker FROM tagged t JOIN mc USING(ticker)
                                  GROUP BY ticker HAVING COUNT(DISTINCT cb)=4)
            """
        ).fetchone()[0]
    finally:
        conn.close()

    raw["domain"] = raw["cat"].apply(get_group)
    raw = raw[raw["domain"].isin(DOMAINS)].copy()
    rows = []
    for (d, cb), g in raw.groupby(["domain", "cb"]):
        if int(g["n"].sum()) < CELL_MIN:
            continue
        res = fit_logistic(g["yes_price"].to_numpy(float), g["is_yes"].to_numpy(float),
                           g["w"].to_numpy(float))
        if res:
            rows.append(dict(domain=d, coarse_bin=CB[int(cb)], coarse_bin_order=int(cb),
                             slope_b=round(res[0], 3), n_trades=int(g["n"].sum())))
    out = pd.DataFrame(rows)
    out.attrs["n_contracts"] = n_contracts
    out.to_csv(OUT / "availability_balanced_4bin.csv", index=False)
    print(f"  availability_balanced_4bin.csv: {n_contracts} contracts traded in all 4 coarse bins")
    print(out.pivot_table(index="domain", columns="coarse_bin", values="slope_b")
          .reindex(columns=["<24h", "1d-1w", "1w-1mo", ">1mo"]).to_string())


def main() -> None:
    cal = pd.read_csv(KALSHI_OUT / "calibration_matrix_decomposed.csv")
    write_intercept_and_sample_summaries(cal)
    write_decomposition_with_size_main(cal)
    write_intercept_and_ice()
    write_clustered_cell_se()
    write_flexible_calibration_diagnostics()
    write_nonparametric_metrics()
    write_burst_aggregation_robustness()
    write_fee_and_size_diagnostics()
    write_availability_restricted_slopes()
    write_balanced_4bin_panel()
    write_balanced_availability()
    print(f"Revision diagnostics written to {OUT}")


if __name__ == "__main__":
    main()
