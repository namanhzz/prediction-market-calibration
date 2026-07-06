"""Bayesian hierarchical model: M0/M1/M2, LOO-CV, posterior predictive check.

Requires calibration data CSV produced by scripts/collect_calibration_data.py
from the data repo, or a pre-computed file at output/calibration_model_data.csv.

Outputs to output/bayesian/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUTPUT_DIR

OUT = OUTPUT_DIR / "bayesian"
OUT.mkdir(parents=True, exist_ok=True)


def load_data(sample_size=None, seed=42):
    """Load calibration data CSV."""
    # Try multiple locations for the calibration data
    candidates = [
        OUTPUT_DIR / "calibration_model_data.csv",
        Path(__file__).resolve().parent.parent.parent / "prediction-market-analysis" / "output" / "calibration_model_data.csv",
    ]
    csv_path = None
    for p in candidates:
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        raise FileNotFoundError(
            "Calibration data not found. Expected at one of:\n"
            + "\n".join(f"  {p}" for p in candidates)
        )

    df = pd.read_csv(csv_path)
    meta_path = csv_path.parent / "calibration_model_meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}

    print(f"Loaded {len(df):,} markets from {csv_path.name}")
    print(f"  Platforms: {df['platform'].value_counts().to_dict()}")
    print(f"  Categories: {df['category'].nunique()}")

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)
        print(f"  Subsampled to {sample_size:,} markets")

    return df, meta


def main():
    parser = argparse.ArgumentParser(description="Fit hierarchical Bayesian calibration models")
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-loo", action="store_true")
    args = parser.parse_args()

    # Late import to avoid slow JAX init when just checking --help
    from src.bayesian import (
        NUM_CHAINS,
        compute_log_lik,
        fit_mcmc,
        model_m0,
        model_m1,
        model_m2,
        prepare_arrays,
        print_diagnostics,
        run_loo_comparison,
    )
    from src.plotting import fig_forest_plot, fig_calibration_by_category, fig_platform_effect
    import arviz as az
    import jax

    chains = args.chains or NUM_CHAINS
    sample_size = None if args.full else args.sample_size

    print(f"Hardware: {jax.device_count()} JAX devices, {chains} MCMC chains")

    df, meta = load_data(sample_size=sample_size, seed=args.seed)
    data_full = prepare_arrays(df)
    data_kalshi = prepare_arrays(df, platform_filter="kalshi")

    print(f"\nFull: N={data_full['N']:,}, J={data_full['J']}")
    print(f"Kalshi: N={data_kalshi['N']:,}, J={data_kalshi['J']}")

    summaries = {}
    traces = {}

    # M0 Pooled (Kalshi)
    mcmc_m0k, trace_m0k = fit_mcmc(
        model_m0, (data_kalshi["logit_p"],), {"y": data_kalshi["y"]},
        n_obs=data_kalshi["N"], chains=chains, draws=args.draws, warmup=args.warmup,
        seed=args.seed + 1, label="M0 (Pooled, Kalshi)",
    )
    summaries["M0 (Kalshi)"] = print_diagnostics(trace_m0k, ["alpha", "beta"], "M0 Kalshi")
    traces["M0_kalshi"] = trace_m0k

    # M1 Category RE (Kalshi)
    mcmc_m1, trace_m1 = fit_mcmc(
        model_m1,
        (data_kalshi["logit_p"], data_kalshi["cat_idx"], data_kalshi["J"]),
        {"y": data_kalshi["y"]},
        n_obs=data_kalshi["N"], chains=chains, draws=args.draws, warmup=args.warmup,
        seed=args.seed + 2, label="M1 (Category RE, Kalshi)",
    )
    m1_vars = ["mu_alpha", "mu_beta", "sigma_alpha", "sigma_beta", "alpha_j", "beta_j"]
    summaries["M1 (Kalshi)"] = print_diagnostics(trace_m1, m1_vars, "M1 Kalshi")
    traces["M1"] = trace_m1

    # M2 Category RE + Platform (Full)
    mcmc_m2, trace_m2 = fit_mcmc(
        model_m2,
        (data_full["logit_p"], data_full["cat_idx"], data_full["platform"], data_full["J"]),
        {"y": data_full["y"]},
        n_obs=data_full["N"], chains=chains, draws=args.draws, warmup=args.warmup,
        seed=args.seed + 3, label="M2 (Category RE + Platform, Full)",
    )
    m2_vars = ["mu_alpha", "mu_beta", "gamma", "sigma_alpha", "sigma_beta", "alpha_j", "beta_j"]
    summaries["M2 (Full)"] = print_diagnostics(trace_m2, m2_vars, "M2 Full")
    traces["M2"] = trace_m2

    # LOO-CV
    loo_results = {}
    if not args.skip_loo:
        print(f"\n{'=' * 60}")
        print("Computing LOO-CV")
        print(f"{'=' * 60}")

        total_draws = chains * args.draws
        thin = max(1, total_draws // 1000)

        def _make_loo_trace(model_fn, mcmc_obj, model_args, model_kwargs, label):
            print(f"  Computing log-lik for {label}...")
            ll_arr = compute_log_lik(model_fn, mcmc_obj, model_args, model_kwargs, thin=thin)
            chain_post = mcmc_obj.get_samples(group_by_chain=True)
            thinned_post = {k: np.asarray(v[:, ::thin, ...]) for k, v in chain_post.items()}
            return az.from_dict(posterior=thinned_post, log_likelihood={"y_obs": ll_arr})

        loo_m0k = _make_loo_trace(model_m0, mcmc_m0k, (data_kalshi["logit_p"],),
                                  {"y": data_kalshi["y"]}, "M0 (Kalshi)")
        loo_m1 = _make_loo_trace(model_m1, mcmc_m1,
                                 (data_kalshi["logit_p"], data_kalshi["cat_idx"], data_kalshi["J"]),
                                 {"y": data_kalshi["y"]}, "M1 (Kalshi RE)")

        comp1 = run_loo_comparison({"M0 (Kalshi)": loo_m0k, "M1 (Kalshi RE)": loo_m1},
                                   "M0 (Kalshi)", "M1 (Kalshi RE)")
        if comp1 is not None:
            loo_results["M0 vs M1 (Kalshi)"] = comp1

    # Figures
    print(f"\n{'=' * 60}")
    print("Generating figures")
    print(f"{'=' * 60}")

    fig_forest_plot(trace_m1, data_kalshi["cat_names"], str(OUT / "forest_plot"), label="M1")
    print("  saved forest_plot.{png,pdf}")

    fig_calibration_by_category(trace_m1, data_kalshi, data_kalshi["cat_names"],
                                str(OUT / "calibration_by_category"), label="M1")
    print("  saved calibration_by_category.{png,pdf}")

    fig_platform_effect(trace_m2, str(OUT / "platform_effect"))
    print("  saved platform_effect.{png,pdf}")

    # Save summaries
    for name, summary in summaries.items():
        csv_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        summary.to_csv(OUT / f"summary_{csv_name}.csv")

    # Save traces
    for name, trace in traces.items():
        nc_path = OUT / f"trace_{name}.nc"
        try:
            trace.to_netcdf(str(nc_path))
            print(f"  saved {nc_path.name}")
        except (ImportError, Exception) as e:
            print(f"  skipping {nc_path.name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"DONE — outputs in {OUT}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
