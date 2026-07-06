"""Create Polymarket parquet files expected by the calibration repo.

The companion data repo stores raw Polymarket CTF trades and market metadata
under data/polymarket/. This script builds the normalized files consumed by
scripts/run_cross_platform.py:

  data/unified/markets/polymarket.parquet
  data/unified/trades/polymarket_ctf.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT.parent / "prediction-market-analysis" / "data"


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def build_unified(data_dir: Path, overwrite: bool = False, threads: int | None = None) -> None:
    raw_markets = data_dir / "polymarket" / "markets"
    raw_trades = data_dir / "polymarket" / "trades"
    raw_blocks = data_dir / "polymarket" / "blocks"
    out_markets = data_dir / "unified" / "markets" / "polymarket.parquet"
    out_trades = data_dir / "unified" / "trades" / "polymarket_ctf.parquet"

    for path in (raw_markets, raw_trades, raw_blocks):
        if not path.exists():
            raise FileNotFoundError(f"Missing raw Polymarket data directory: {path}")

    for path in (out_markets.parent, out_trades.parent):
        path.mkdir(parents=True, exist_ok=True)

    if not overwrite and out_markets.exists() and out_trades.exists():
        print("Unified Polymarket files already exist; use --overwrite to rebuild.")
        return

    conn = duckdb.connect()
    if threads:
        conn.execute(f"SET threads TO {int(threads)}")

    markets_glob = _sql_path(raw_markets / "*.parquet")
    trades_glob = _sql_path(raw_trades / "*.parquet")
    blocks_glob = _sql_path(raw_blocks / "*.parquet")
    out_markets_sql = _sql_path(out_markets)
    out_trades_sql = _sql_path(out_trades)

    resolved_markets = f"""
        SELECT
            id AS ticker,
            slug AS event_ticker,
            CASE
                WHEN try_cast(json_extract_string(outcome_prices, '$[0]') AS DOUBLE) > 0.99
                 AND try_cast(json_extract_string(outcome_prices, '$[1]') AS DOUBLE) < 0.01
                THEN 'yes'
                ELSE 'no'
            END AS result,
            end_date AS close_time,
            question AS title,
            NULL::VARCHAR AS _domain,
            'finalized' AS status,
            json_extract_string(clob_token_ids, '$[0]') AS yes_token,
            json_extract_string(clob_token_ids, '$[1]') AS no_token
        FROM '{markets_glob}'
        WHERE closed = true
          AND json_array_length(outcome_prices) = 2
          AND json_array_length(clob_token_ids) = 2
          AND (
                (
                    try_cast(json_extract_string(outcome_prices, '$[0]') AS DOUBLE) > 0.99
                    AND try_cast(json_extract_string(outcome_prices, '$[1]') AS DOUBLE) < 0.01
                )
             OR (
                    try_cast(json_extract_string(outcome_prices, '$[0]') AS DOUBLE) < 0.01
                    AND try_cast(json_extract_string(outcome_prices, '$[1]') AS DOUBLE) > 0.99
                )
          )
    """

    if overwrite:
        for path in (out_markets, out_trades):
            if path.exists():
                path.unlink()

    print(f"Writing {out_markets}")
    conn.execute(
        f"""
        COPY (
            SELECT ticker, event_ticker, result, close_time, title, _domain, status
            FROM ({resolved_markets})
        )
        TO '{out_markets_sql}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    print(f"Writing {out_trades}")
    conn.execute(
        f"""
        COPY (
            WITH resolved AS ({resolved_markets}),
            token_map AS (
                SELECT ticker, yes_token AS token_id, true AS is_yes_token FROM resolved
                UNION ALL
                SELECT ticker, no_token AS token_id, false AS is_yes_token FROM resolved
            ),
            normalized_trades AS (
                SELECT
                    block_number,
                    CASE WHEN maker_asset_id = '0' THEN taker_asset_id ELSE maker_asset_id END AS token_id,
                    CASE WHEN maker_asset_id = '0' THEN maker_amount ELSE taker_amount END AS usdc_amount,
                    CASE WHEN maker_asset_id = '0' THEN taker_amount ELSE maker_amount END AS token_amount
                FROM '{trades_glob}'
                WHERE maker_amount > 0
                  AND taker_amount > 0
                  AND (maker_asset_id = '0' OR taker_asset_id = '0')
            )
            SELECT
                tm.ticker,
                CAST(
                    CASE
                        WHEN tm.is_yes_token
                        THEN round(100.0 * t.usdc_amount / t.token_amount)
                        ELSE round(100.0 - 100.0 * t.usdc_amount / t.token_amount)
                    END
                    AS INTEGER
                ) AS yes_price,
                t.token_amount / 1000000.0 AS count,
                CAST(b.timestamp AS TIMESTAMPTZ) AS created_time
            FROM normalized_trades t
            INNER JOIN token_map tm ON t.token_id = tm.token_id
            INNER JOIN '{blocks_glob}' b ON t.block_number = b.block_number
            WHERE t.token_amount > 0
              AND t.usdc_amount > 0
        )
        TO '{out_trades_sql}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000);
        """
    )

    market_count = conn.execute(f"SELECT count(*) FROM '{out_markets_sql}'").fetchone()[0]
    trade_count = conn.execute(f"SELECT count(*) FROM '{out_trades_sql}'").fetchone()[0]
    conn.close()
    print(f"Unified markets: {market_count:,}")
    print(f"Unified CTF trades: {trade_count:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    build_unified(args.data_dir.resolve(), overwrite=args.overwrite, threads=args.threads)


if __name__ == "__main__":
    main()
