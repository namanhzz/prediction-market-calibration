"""Build market-level data for scripts/run_bayesian.py."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classify import classify_polymarket_domain, get_group
from src.config import DATE_CUTOFF, KALSHI_MARKETS, KALSHI_TRADES, OUTPUT_DIR, PM_MARKETS, PM_TRADES


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _logit(p: pd.Series) -> np.ndarray:
    p = p.clip(1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def collect(min_trades: int = 10, output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "calibration_model_data.csv"
    out_meta = output_dir / "calibration_model_meta.json"

    conn = duckdb.connect()
    kalshi_markets = _sql_path(KALSHI_MARKETS / "*.parquet")
    kalshi_trades = _sql_path(KALSHI_TRADES / "*.parquet")
    pm_markets = _sql_path(PM_MARKETS)
    pm_trades = _sql_path(PM_TRADES)

    print("Aggregating Kalshi markets...")
    kalshi = conn.execute(
        f"""
        SELECT
            m.ticker,
            regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) AS cat_prefix,
            CASE WHEN m.result = 'yes' THEN 1 ELSE 0 END AS outcome,
            SUM(t.yes_price * t.count) / SUM(t.count) / 100.0 AS implied_prob,
            COUNT(*) AS n_trades,
            SUM(t.count) AS n_contracts
        FROM '{kalshi_markets}' m
        INNER JOIN '{kalshi_trades}' t ON t.ticker = m.ticker
        WHERE m.status = 'finalized'
          AND m.result IN ('yes', 'no')
          AND t.created_time <= TIMESTAMP '{DATE_CUTOFF}'
          AND m.close_time > t.created_time
          AND t.yes_price BETWEEN 1 AND 99
        GROUP BY m.ticker, cat_prefix, m.result
        HAVING COUNT(*) >= {int(min_trades)}
        """
    ).df()
    kalshi["category"] = kalshi["cat_prefix"].map(get_group)
    kalshi = kalshi[kalshi["category"] != "Other"].copy()
    kalshi["platform"] = "kalshi"
    kalshi["platform_id"] = 0

    print("Aggregating Polymarket markets...")
    polymarket = conn.execute(
        f"""
        SELECT
            m.ticker,
            m.title,
            CASE WHEN m.result = 'yes' THEN 1 ELSE 0 END AS outcome,
            SUM(t.yes_price * t.count) / SUM(t.count) / 100.0 AS implied_prob,
            COUNT(*) AS n_trades,
            SUM(t.count) AS n_contracts
        FROM '{pm_markets}' m
        INNER JOIN '{pm_trades}' t ON t.ticker = m.ticker
        WHERE m.status = 'finalized'
          AND m.result IN ('yes', 'no')
          AND t.created_time <= TIMESTAMP '{DATE_CUTOFF}'
          AND m.close_time > t.created_time
          AND t.yes_price BETWEEN 1 AND 99
        GROUP BY m.ticker, m.title, m.result
        HAVING COUNT(*) >= {int(min_trades)}
        """
    ).df()
    conn.close()

    polymarket["category"] = polymarket["title"].map(classify_polymarket_domain)
    polymarket = polymarket[polymarket["category"] != "Other"].copy()
    polymarket["platform"] = "polymarket"
    polymarket["platform_id"] = 1

    cols = ["platform", "platform_id", "category", "ticker", "implied_prob", "outcome", "n_trades", "n_contracts"]
    df = pd.concat([kalshi[cols], polymarket[cols]], ignore_index=True)
    df = df[(df["implied_prob"] > 0) & (df["implied_prob"] < 1)].copy()
    df["logit_p"] = _logit(df["implied_prob"])
    df = df[["platform", "platform_id", "category", "ticker", "implied_prob", "logit_p", "outcome", "n_trades", "n_contracts"]]
    df.to_csv(out_csv, index=False)

    meta = {
        "description": "Market-level VWAP calibration data for Bayesian models.",
        "min_trades_per_market": min_trades,
        "date_cutoff": DATE_CUTOFF,
        "rows": int(len(df)),
        "platform_counts": {k: int(v) for k, v in df["platform"].value_counts().to_dict().items()},
        "category_counts": {k: int(v) for k, v in df["category"].value_counts().to_dict().items()},
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(df):,} rows to {out_csv}")
    print(f"Wrote metadata to {out_meta}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    collect(min_trades=args.min_trades, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
