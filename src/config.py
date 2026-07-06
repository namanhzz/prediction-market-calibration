"""All constants, paths, and bin definitions."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ── Paths (configurable via environment) ───────────────────────────
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / ".." / "prediction-market-analysis" / "data")))
KALSHI_MARKETS = DATA_DIR / "kalshi" / "markets"
KALSHI_TRADES = DATA_DIR / "kalshi" / "trades"
PM_MARKETS = DATA_DIR / "unified" / "markets" / "polymarket.parquet"
PM_TRADES = DATA_DIR / "unified" / "trades" / "polymarket_ctf.parquet"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(Path(__file__).resolve().parent.parent / "output")))

# ── Constants ──────────────────────────────────────────────────────
DOMAINS = ["Sports", "Crypto", "Politics", "Finance", "Weather", "Entertainment"]
TIME_BINS = [
    (0, 1, "0-1h"), (1, 3, "1-3h"), (3, 6, "3-6h"), (6, 12, "6-12h"),
    (12, 24, "12-24h"), (24, 48, "24-48h"), (48, 168, "2d-1w"),
    (168, 720, "1w-1mo"), (720, 1e9, "1mo+"),
]
SIZE_BINS = [(1, 1, "Single"), (2, 10, "Small"), (11, 100, "Medium"), (101, int(1e9), "Large")]
COLORS = {
    "Politics": "#D62728", "Sports": "#1F77B4", "Weather": "#2CA02C",
    "Crypto": "#FF7F0E", "Finance": "#9467BD", "Entertainment": "#7F7F7F",
}
BIN_LABELS = [label for _, _, label in TIME_BINS]
SIZE_LABELS = [label for _, _, label in SIZE_BINS]
SIZE_LOG_MEDIANS = {
    "Single": np.log(1.0), "Small": np.log(4.0),
    "Medium": np.log(30.0), "Large": np.log(300.0),
}
DATE_CUTOFF = "2025-12-31T23:59:59Z"
C_REG = 10.0
CELL_MIN = 200

# Cross-platform analysis uses a smaller set of shared domains
CROSS_PLATFORM_DOMAINS = ["Sports", "Crypto", "Politics", "Finance"]
