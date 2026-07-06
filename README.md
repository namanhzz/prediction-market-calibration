# Prediction Market Calibration: A Microstructure Analysis

Replication code for the paper *Prediction Market Calibration: A Microstructure Analysis*.

## Overview

This repository contains code to reproduce the empirical analysis of prediction market calibration across Kalshi and Polymarket. The analysis fits logistic recalibration models, performs ANOVA-style variance decomposition, estimates Bayesian hierarchical models, and runs robustness checks.

## Repository Structure

```
src/
  config.py           Constants, paths, bin definitions
  classify.py          Domain classification (Kalshi taxonomy + Polymarket regex)
  calibration.py       Logistic recalibration, decomposition, bootstrap
  bayesian.py          NumPyro hierarchical models (M0, M1, M2)
  pipeline.py          SQL helpers, DuckDB data loading
  plotting.py          matplotlib figure generation

scripts/
  run_kalshi.py            Main Kalshi analysis pipeline
  run_bayesian.py          Bayesian hierarchical models + LOO-CV
  run_cross_platform.py    Polymarket replication
  run_robustness.py        All robustness checks
  run_revision_diagnostics.py  R1 revision diagnostics from generated outputs
  generate_figures.py      Publication figures

tests/
  test_imports.py          Smoke tests
```

## Data Requirements

This repository does not include data. The raw Parquet data (~36 GiB) is available from the companion data repository:

```bash
# Clone the data repo and download data
git clone https://github.com/Jon-Becker/prediction-market-analysis.git
cd prediction-market-analysis
make setup
```

By default, scripts look for data at `../prediction-market-analysis/data/`. Override with:

```bash
export DATA_DIR=/path/to/data
```

### Polymarket snapshot note

Kalshi outputs reproduce from the downloaded companion data. Polymarket and cross-platform outputs require the exact normalized unified files used for the manuscript:

```
data/unified/markets/polymarket.parquet
data/unified/trades/polymarket_ctf.parquet
```

If those files are regenerated from a newer or differently normalized raw Polymarket dump, aggregate Politics size slopes remain close to the manuscript, but some cross-platform time-bin slopes and bootstrap intervals can change. For an archival release, deposit the exact unified Polymarket snapshot with checksums, or regenerate the manuscript tables from the current snapshot and report that version explicitly.

## Reproduction

### Quick Start

```bash
# Install dependencies
pip install -e .

# Run all analyses
make reproduce
```

### Step by Step

```bash
# 1. Kalshi calibration (requires data)
make kalshi

# 2. Bayesian models (requires calibration data CSV)
make bayesian

# 3. Cross-platform comparison (requires Polymarket unified data)
make cross-platform

# 4. Robustness checks (requires Kalshi data + step 1 outputs)
make robustness

# 5. R1 revision diagnostics (requires step 1 outputs)
make revision-diagnostics

# 6. Generate all publication figures
make figures
```

### Tests

```bash
make test
```

## Key Results

- **87.3%** of cell-level slope variance explained by the five-term decomposition (descriptive, slopes treated as exact)
- Components: horizon 30.2% | domain 14.6% | common size 3.2% | domain x horizon 26.0% | domain x size 13.3%
- Overfitting checks: adjusted R2 = 0.810; permutation-null mean R2 = 0.329 (observed far above, p < 0.001); leave-one-cell-out R2 = 0.715; 10-fold CV R2 = 0.704
- Bayesian **measurement-error** model (event-clustered SE in the likelihood): once first-stage uncertainty is propagated, ~57% of raw slope variance is estimation noise and structural components explain ~46% (95% CrI [31%, 62%]); residual structural sigma ~ 0
- Politics domain intercept: alpha = +0.107 [0.062, 0.152] (Bayesian, shrunk by clustered SE), +0.156 (descriptive)
- Bayesian diagnostics: PPC coverage 99.5% (215/216 cells), max R-hat = 1.000, min ESS = 3,794, 0 divergences
- Clustering: event-clustered first-stage SE ~ 50x naive Fisher SE
- Politics whale effect: Delta = +0.53 [0.29, 0.75] on Kalshi, robust to market- and event-clustered bootstraps; on Polymarket +0.28 [0.03, 0.54] at the cell level but NOT robust to market clustering (+0.21 [-0.31, 1.12])
- Intercept treated as first-class: parallel decomposition (domain x horizon explains 62.8% of intercept variance) and integrated calibration error (volume-weighted ICE ranks Politics worst, Sports best)
- Nonparametric: Politics ECE = 0.117 vs 0.007-0.022 elsewhere (model-free confirmation)
- Cross-platform: political underconfidence replicates on Polymarket (mean slope 1.45); the scale effect is specific to Kalshi under clustering. All Polymarket numbers locked to a checksummed snapshot (see DATA_SNAPSHOT_NOTE.md)

## License

MIT License. See [LICENSE](LICENSE).
