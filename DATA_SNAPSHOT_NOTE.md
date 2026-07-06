# Data Snapshot Note for IJOF R1 Revision

Created: 2026-06-23. Status: **LOCKED** to the snapshot below as of the R1 revision.

## Locked snapshot

All Kalshi numbers reproduce exactly from the local companion data. All
Polymarket and cross-platform numbers in the revised manuscript are regenerated
from, and locked to, the following unified Polymarket snapshot:

| File | Size (bytes) | SHA256 |
| --- | ---: | --- |
| `../prediction-market-analysis/data/unified/markets/polymarket.parquet` | 6,157,856 | `890090261543fa03f4c09012c50fb31c8e436f4335144cde8df3f0ec7c8ac645` |
| `../prediction-market-analysis/data/unified/trades/polymarket_ctf.parquet` | 2,173,978,796 | `cb1dbf9713f521e87a3bea39d50338e1d3b9e2b084d1c3edffc8a96ec653df75` |

## Decision taken for R1

Per author decision (2026-06-23), the revision **regenerates all Polymarket and
cross-platform numbers from the snapshot above** rather than recovering the
original submitted snapshot, and discloses the snapshot checksums in the Data
Availability statement. This guarantees 100% reproducibility from the deposited
snapshot.

## Material consequences of regeneration (vs the originally submitted numbers)

The current snapshot differs from the one underlying the original submission. The
**three analyzed domains are essentially unchanged** (e.g. Politics: 45.7M resolved
trades and 24.6B contracts in both; Sports 49.1M; Crypto ~125M), but the current
snapshot carries a **larger long-tail of bespoke "Other" markets**, so full-dataset
Polymarket totals are larger (288.7M resolved trades / 218k resolved contracts now,
vs 227.6M / 116k originally). After the 5--95c price filter and the four-domain
cross-platform restriction, 135.6M trades enter the calibration analysis (this is a
filtered subset, not directly comparable to the original 3-domain total). Regenerating
changes several Polymarket numbers, but the substantive cross-platform conclusions are
unchanged or strengthened:

- **Political underconfidence replicates on Polymarket** and is, if anything,
  stronger: Politics mean slope over reliable bins = 1.45 (was 1.31).
- **Whale (scale) effect remains Kalshi-specific under proper clustering.**
  Polymarket Politics Delta = +0.281 [0.026, 0.542] under the cell-level
  bootstrap (significant) but +0.207 [-0.315, 1.100] under the
  market-clustered bootstrap (NOT significant). Kalshi Politics is significant
  under cell, market- AND event-clustered bootstraps. So the requested
  clustering reinforces the platform-specific reading.
- **Trade sizes are comparable across platforms** (Politics median 43.5 on
  Polymarket vs 45 on Kalshi), so the stronger Kalshi whale effect is NOT
  explained by larger Kalshi political bets.
- Polymarket bins 0-1h and 1-3h remain unreliable (block-number timestamp noise)
  and are excluded from reliable-bin means.

## Reproduction

`make reproduce` regenerates every number from the deposited snapshot:
kalshi -> revision-diagnostics -> bayesian-decomp -> bayesian -> cross-platform
-> robustness -> figures.
