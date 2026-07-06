"""Smoke tests: verify all modules import and key objects exist."""
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_config_imports():
    from src import config
    assert config.DOMAINS
    assert len(config.TIME_BINS) == 9
    assert len(config.SIZE_BINS) == 4
    assert config.C_REG == 10.0


def test_classify_imports():
    from src import classify
    assert callable(classify.get_group)
    assert callable(classify.get_hierarchy)
    assert callable(classify.classify_polymarket_domain)
    assert classify.get_group("NFLGAME") == "Sports"
    assert classify.get_group("PRES") == "Politics"
    assert classify.get_group("BTC") == "Crypto"
    assert classify.classify_polymarket_domain("Will the Lakers win?") == "Sports"
    assert classify.classify_polymarket_domain("Trump approval rating") == "Politics"


def test_calibration_imports():
    from src import calibration
    assert callable(calibration.fit_logistic)
    assert callable(calibration.decompose)
    assert callable(calibration.bootstrap_whale_effect)
    assert callable(calibration.fit_slope)
    assert callable(calibration.compute_weighted_decomposition)


def test_pipeline_imports():
    from src import pipeline
    assert callable(pipeline.time_bin_sql)
    assert callable(pipeline.size_bin_sql)
    assert callable(pipeline.fit_calibration_matrix)
    assert callable(pipeline.fit_slopes_by_domain_time)


def test_plotting_imports():
    from src import plotting
    assert callable(plotting.setup_matplotlib)
    assert callable(plotting.fig_slope_trajectories)
    assert callable(plotting.fig_hero_decomposition)
    assert callable(plotting.fig_observed_vs_fitted)
    assert callable(plotting.fig_whale_effect)


def test_all_imports():
    """Combined smoke test."""
    from src import config, classify, calibration, pipeline, plotting
    assert config.DOMAINS
    assert callable(classify.get_group)
    assert callable(calibration.fit_logistic)
