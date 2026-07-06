"""SQL helpers, DuckDB data loading, and aggregation functions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.calibration import fit_logistic
from src.classify import get_group
from src.config import (
    BIN_LABELS,
    CELL_MIN,
    DATE_CUTOFF,
    DOMAINS,
    KALSHI_MARKETS,
    KALSHI_TRADES,
    SIZE_BINS,
    SIZE_LABELS,
    TIME_BINS,
)


def time_bin_sql():
    """Generate SQL CASE expression for time-to-close bins."""
    parts = []
    for i, (lo, hi, _) in enumerate(TIME_BINS):
        if hi >= 1e9:
            parts.append(f"WHEN hours_to_close >= {lo} THEN {i}")
        else:
            parts.append(f"WHEN hours_to_close >= {lo} AND hours_to_close < {hi} THEN {i}")
    return "CASE " + " ".join(parts) + " ELSE -1 END"


def size_bin_sql():
    """Generate SQL CASE expression for trade-size bins."""
    parts = []
    for i, (lo, hi, _) in enumerate(SIZE_BINS):
        if hi >= int(1e9):
            parts.append(f"WHEN t.count >= {lo} THEN {i}")
        else:
            parts.append(f"WHEN t.count >= {lo} AND t.count <= {hi} THEN {i}")
    return "CASE " + " ".join(parts) + " ELSE -1 END"


def load_kalshi_trades(conn, price_lo=5, price_hi=95, min_trades=10):
    """Load and pre-aggregate Kalshi trade data via DuckDB.

    Returns a DataFrame with columns: cat_prefix, domain, tbin, sbin,
    yes_price, is_yes, total_contracts, n_trades.
    """
    markets = str(KALSHI_MARKETS).replace("\\", "/")
    trades = str(KALSHI_TRADES).replace("\\", "/")
    tb = time_bin_sql()
    sb = size_bin_sql()

    df = conn.execute(f"""
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
                   ({sb}) AS sbin,
                   m.ticker
            FROM '{trades}/*.parquet' t
            INNER JOIN resolved m ON t.ticker = m.ticker
            WHERE t.created_time <= TIMESTAMP '{DATE_CUTOFF}'
              AND m.close_time > t.created_time
        ),
        market_counts AS (
            SELECT ticker, COUNT(*) AS ntrades
            FROM trade_data
            GROUP BY ticker
            HAVING COUNT(*) >= {min_trades}
        )
        SELECT td.cat_prefix, ({tb}) AS tbin, td.sbin, td.yes_price, td.is_yes,
               SUM(td.trade_count) AS total_contracts, COUNT(*) AS n_trades
        FROM trade_data td
        INNER JOIN market_counts mc ON td.ticker = mc.ticker
        WHERE td.yes_price BETWEEN {price_lo} AND {price_hi} AND ({tb}) >= 0 AND td.sbin >= 0
        GROUP BY td.cat_prefix, ({tb}), td.sbin, td.yes_price, td.is_yes
    """).df()

    df["domain"] = df["cat_prefix"].apply(get_group)
    df = df[df["domain"].isin(DOMAINS)].copy()
    return df


def load_kalshi_market_stats(conn):
    """Load market-level summary statistics for Table 1.

    Returns (all_mkts, resolved) DataFrames.
    """
    markets = str(KALSHI_MARKETS).replace("\\", "/")
    trades = str(KALSHI_TRADES).replace("\\", "/")

    all_mkts = conn.execute(f"""
        SELECT
            regexp_extract(event_ticker, '^([A-Z0-9]+)', 1) AS cat_prefix,
            ticker, result, status, close_time
        FROM '{markets}/*.parquet'
        WHERE close_time <= TIMESTAMP '{DATE_CUTOFF}'
    """).df()
    all_mkts["domain"] = all_mkts["cat_prefix"].apply(get_group)

    resolved = conn.execute(f"""
        SELECT m.ticker, m.event_ticker, m.result,
               regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) AS cat_prefix,
               COUNT(*) AS n_trades,
               SUM(t.count) AS n_contracts,
               AVG(t.yes_price) AS mean_price
        FROM '{markets}/*.parquet' m
        INNER JOIN '{trades}/*.parquet' t ON t.ticker = m.ticker
        WHERE m.status = 'finalized' AND m.result IN ('yes', 'no')
              AND t.created_time <= TIMESTAMP '{DATE_CUTOFF}'
        GROUP BY m.ticker, m.event_ticker, m.result, cat_prefix
        HAVING COUNT(*) >= 10
    """).df()
    resolved["domain"] = resolved["cat_prefix"].apply(get_group)

    return all_mkts, resolved


def fit_calibration_matrix(df, cell_min=CELL_MIN):
    """Fit logistic calibration per domain x time x size cell.

    Returns DataFrame with calibration slopes per cell.
    """
    rows = []
    for (domain, tbin, sbin), cell in df.groupby(["domain", "tbin", "sbin"]):
        n_t = int(cell["n_trades"].sum())
        if n_t < cell_min:
            continue
        prices = cell["yes_price"].values.astype(float)
        outcomes = cell["is_yes"].values.astype(float)
        weights = cell["total_contracts"].values.astype(float)

        result = fit_logistic(prices, outcomes, weights)
        if result is None:
            continue
        b, a, se = result
        n_yes = int(cell.loc[cell["is_yes"] == 1, "n_trades"].sum())
        n_no = int(cell.loc[cell["is_yes"] == 0, "n_trades"].sum())

        rows.append(dict(
            domain=domain,
            time_bin=BIN_LABELS[int(tbin)],
            time_bin_order=int(tbin) + 1,
            size_bin=SIZE_LABELS[int(sbin)],
            size_bin_order=int(sbin) + 1,
            n_trades=n_t,
            n_yes=n_yes,
            n_no=n_no,
            slope_b=b,
            intercept_a=a,
            slope_stderr=se,
            mean_price=float(np.average(prices, weights=weights)),
            median_price=float(np.median(prices)),
        ))

    return pd.DataFrame(rows)


def fit_slopes_by_domain_time(df, cell_min=CELL_MIN):
    """Fit calibration slopes aggregated by domain x time (no size split)."""
    rows = []
    for (domain, tbin), cell in df.groupby(["domain", "tbin"]):
        n_t = int(cell["n_trades"].sum())
        if n_t < cell_min:
            continue
        result = fit_logistic(
            cell["yes_price"].values.astype(float),
            cell["is_yes"].values.astype(float),
            cell["total_contracts"].values.astype(float),
        )
        if result:
            b, a, se = result
            rows.append(dict(
                domain=domain, time_bin=BIN_LABELS[int(tbin)],
                time_bin_order=int(tbin) + 1, n_trades=n_t,
                slope_b=b, intercept_a=a, slope_stderr=se,
            ))
    return pd.DataFrame(rows)


def fit_slopes_by_domain_size(df, cell_min=CELL_MIN):
    """Fit calibration slopes aggregated by domain x size (no time split)."""
    rows = []
    for (domain, sbin), cell in df.groupby(["domain", "sbin"]):
        n_t = int(cell["n_trades"].sum())
        if n_t < cell_min:
            continue
        result = fit_logistic(
            cell["yes_price"].values.astype(float),
            cell["is_yes"].values.astype(float),
            cell["total_contracts"].values.astype(float),
        )
        if result:
            b, a, se = result
            rows.append(dict(
                domain=domain, size_bin=SIZE_LABELS[int(sbin)],
                size_bin_order=int(sbin) + 1, n_trades=n_t,
                slope_b=b, intercept_a=a, slope_stderr=se,
            ))
    return pd.DataFrame(rows)
