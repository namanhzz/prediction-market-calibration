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

## Why the snapshot changed (technical provenance)

The revised snapshot was regenerated from the public Polymarket source using the
released ingestion pipeline (`scripts/create_unified_polymarket.py`). It contains more
resolved markets than the snapshot underlying the original submission — chiefly a
larger tail of bespoke "Other" markets, plus additional Politics/Crypto markets that
resolved between the two data pulls. The difference could **not** be attributed to a
specific pipeline bug or a changed filter; it is consistent with a fuller/later pull of
the same source rather than a correction of an error. The three analyzed domains
(Sports, Crypto, Politics) are stable in trade and contract counts; the visible changes
are in full-dataset totals and, to a lesser degree, in the Polymarket political slope
and scale gap.

### Reconciliation (original submission vs revised snapshot)

| Quantity | Original submission | Revised snapshot |
| --- | --- | --- |
| Full-dataset PM resolved trades | 227.6M | 288.7M |
| Full-dataset PM resolved contracts | 116k | 218k |
| PM markets (3 comparable domains) | 113,483 | 124,881 |
| PM trades (3 comparable domains) | 220.1M | 219.4M |
| Politics markets | 14,225 | 14,389 |
| Politics trades / contracts | 45.7M / 24.6B | 45.7M / 24.6B |
| Politics mean slope (reliable bins) | 1.31 | 1.45 |
| Politics scale gap Δ (cell bootstrap) | +0.11 [-0.15, 0.39] | +0.28 [0.03, 0.54] |
| Politics scale gap Δ (market-clustered) | not reported | +0.21 [-0.31, 1.12] (n.s.) |

The substantive cross-platform conclusions are unchanged or strengthened:

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
