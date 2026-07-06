"""Bayesian hierarchical measurement-error decomposition (Section 6).

Fits the cell-level slope decomposition with first-stage (event-clustered)
standard errors carried into the likelihood, so first-stage estimation
uncertainty propagates into the decomposition. This is the uncertainty-aware
complement to the descriptive Type I decomposition and is the model behind the
Section 6 / Appendix B tables.

Inputs : output/revision/cell_clustered_se.csv  (216 cells; slope_b,
         event_clustered_se, domain, time_bin, size_bin)
         output/kalshi/calibration_matrix.csv    (frequentist comparison)
Outputs: output/bayesian/*.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BIN_LABELS, DOMAINS, OUTPUT_DIR, SIZE_LABELS, SIZE_LOG_MEDIANS

OUT = OUTPUT_DIR / "bayesian"
OUT.mkdir(parents=True, exist_ok=True)
KALSHI_OUT = OUTPUT_DIR / "kalshi"
REVISION_OUT = OUTPUT_DIR / "revision"


def load_cells():
    """Load the 216-cell matrix with event-clustered SE; fall back to naive SE."""
    path = REVISION_OUT / "cell_clustered_se.csv"
    if path.exists():
        cal = pd.read_csv(path)
        se_col = "event_clustered_se"
        print(f"Loaded {len(cal)} cells with event-clustered SE from {path.name}")
    else:
        cal = pd.read_csv(KALSHI_OUT / "calibration_matrix.csv")
        se_col = "slope_stderr"
        print(f"WARNING: clustered SE not found; using naive Fisher SE from calibration_matrix.csv")
    cal = cal.dropna(subset=["slope_b", se_col]).copy()
    cal = cal[cal[se_col] > 0].copy()

    # centered log trade size
    mean_logsize = np.mean([SIZE_LOG_MEDIANS[s] for s in SIZE_LABELS])
    cal["s_tilde"] = cal["size_bin"].map(lambda s: SIZE_LOG_MEDIANS[s] - mean_logsize)
    cal["domain_idx"] = cal["domain"].map({d: i for i, d in enumerate(DOMAINS)})
    cal["time_idx"] = cal["time_bin"].map({t: i for i, t in enumerate(BIN_LABELS)})
    cal["se_used"] = cal[se_col]
    return cal


def frequentist_alpha(cal):
    """Frequentist domain intercept alpha_d from the Type I decomposition,
    for side-by-side comparison with the Bayesian posterior."""
    from src.calibration import decompose
    base = pd.read_csv(KALSHI_OUT / "calibration_matrix.csv")
    dec = decompose(base)
    return dec.groupby("domain")["alpha"].first().to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import jax.random as random
    import numpyro
    from numpyro.infer import MCMC, NUTS
    import arviz as az

    from src.bayesian import model_decomp

    cal = load_cells()
    J_d, J_t = len(DOMAINS), len(BIN_LABELS)

    domain_idx = jnp.array(cal["domain_idx"].to_numpy(), dtype=jnp.int32)
    time_idx = jnp.array(cal["time_idx"].to_numpy(), dtype=jnp.int32)
    s_tilde = jnp.array(cal["s_tilde"].to_numpy(), dtype=jnp.float64)
    se = jnp.array(cal["se_used"].to_numpy(), dtype=jnp.float64)
    theta_obs = jnp.array(cal["slope_b"].to_numpy(), dtype=jnp.float64)

    kernel = NUTS(model_decomp, target_accept_prob=0.99, max_tree_depth=12)
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.draws,
                num_chains=args.chains, chain_method="parallel", progress_bar=True)
    mcmc.run(random.PRNGKey(args.seed), domain_idx, time_idx, s_tilde, se, J_d, J_t,
             theta_obs=theta_obs)

    trace = az.from_numpyro(mcmc)
    n_div = int(np.sum(np.asarray(trace.sample_stats["diverging"].values)))
    summ = az.summary(trace, var_names=["mu", "alpha", "beta", "delta",
                                        "sigma_alpha", "sigma_beta", "sigma_delta", "sigma"])
    max_rhat = float(summ["r_hat"].max())
    min_ess = float(summ["ess_bulk"].min())
    print(f"\nmax R-hat={max_rhat:.4f}  min ESS_bulk={min_ess:.0f}  divergences={n_div}")

    post = mcmc.get_samples()
    alpha = np.asarray(post["alpha"])         # (S, 6)
    beta = np.asarray(post["beta"])           # (S, 6, 9)
    delta = np.asarray(post["delta"])         # (S, 6)
    mu = np.asarray(post["mu"])               # (S, 9)
    sigma = np.asarray(post["sigma"])         # (S,)
    S = alpha.shape[0]

    def ci(x, axis=0):
        return np.percentile(x, [2.5, 97.5], axis=axis)

    # ── Domain intercepts vs frequentist ──────────────────────────────
    freq_alpha = frequentist_alpha(cal)
    lo, hi = ci(alpha)
    rows = []
    for i, d in enumerate(DOMAINS):
        rows.append(dict(domain=d, post_mean=round(float(alpha[:, i].mean()), 4),
                         sd=round(float(alpha[:, i].std()), 4),
                         ci_lo=round(float(lo[i]), 4), ci_hi=round(float(hi[i]), 4),
                         frequentist=round(float(freq_alpha.get(d, np.nan)), 4),
                         discrepancy=round(abs(float(alpha[:, i].mean()) - float(freq_alpha.get(d, np.nan))), 4)))
    df_alpha = pd.DataFrame(rows)
    df_alpha.to_csv(OUT / "bayesian_domain_intercepts.csv", index=False)
    print("\nDomain intercepts (Bayesian vs frequentist):")
    print(df_alpha.to_string(index=False))

    # ── delta_d (scale sensitivity) ───────────────────────────────────
    lo_d, hi_d = ci(delta)
    df_delta = pd.DataFrame([
        dict(domain=d, post_mean=round(float(delta[:, i].mean()), 4),
             sd=round(float(delta[:, i].std()), 4),
             ci_lo=round(float(lo_d[i]), 4), ci_hi=round(float(hi_d[i]), 4))
        for i, d in enumerate(DOMAINS)
    ])
    df_delta.to_csv(OUT / "bayesian_delta.csv", index=False)

    # ── beta matrix (posterior mean) ──────────────────────────────────
    beta_mean = beta.mean(axis=0)  # (6,9)
    df_beta = pd.DataFrame(beta_mean, index=DOMAINS, columns=BIN_LABELS).round(3)
    df_beta.to_csv(OUT / "bayesian_beta_matrix.csv")

    # ── hyperparameters ───────────────────────────────────────────────
    hyper = []
    for name in ["sigma_alpha", "sigma_beta", "sigma_delta", "sigma"]:
        v = np.asarray(post[name])
        l, h = np.percentile(v, [2.5, 97.5])
        hyper.append(dict(parameter=name, mean=round(float(v.mean()), 4),
                          sd=round(float(v.std()), 4),
                          ci_lo=round(float(l), 4), ci_hi=round(float(h), 4)))
    pd.DataFrame(hyper).to_csv(OUT / "bayesian_hyperparams.csv", index=False)

    # ── posterior predictive coverage ─────────────────────────────────
    d_idx = cal["domain_idx"].to_numpy()
    t_idx = cal["time_idx"].to_numpy()
    st = cal["s_tilde"].to_numpy()
    se_np = cal["se_used"].to_numpy()
    theta_np = cal["slope_b"].to_numpy()

    mean_pred = (mu[:, t_idx] + alpha[:, d_idx]
                 + beta[:, d_idx, t_idx] + delta[:, d_idx] * st[None, :])  # (S, N)
    total_sd = np.sqrt(sigma[:, None] ** 2 + se_np[None, :] ** 2)
    rng = np.random.default_rng(args.seed)
    ppc = mean_pred + rng.standard_normal(mean_pred.shape) * total_sd       # (S, N)
    lo_p = np.percentile(ppc, 2.5, axis=0)
    hi_p = np.percentile(ppc, 97.5, axis=0)
    within = (theta_np >= lo_p) & (theta_np <= hi_p)
    cal["_within"] = within
    cov_rows = [dict(domain="ALL", n_cells=len(within), within=int(within.sum()),
                     coverage=round(100 * within.mean(), 1))]
    for d in DOMAINS:
        m = cal["domain"] == d
        cov_rows.append(dict(domain=d, n_cells=int(m.sum()), within=int(cal.loc[m, "_within"].sum()),
                             coverage=round(100 * cal.loc[m, "_within"].mean(), 1)))
    pd.DataFrame(cov_rows).to_csv(OUT / "bayesian_ppc_coverage.csv", index=False)
    print(f"\nPPC coverage (overall): {100*within.mean():.1f}% ({within.sum()}/{len(within)})")

    # ── posterior % variance per component (R² with credible interval) ─
    # On the balanced grid the component vectors are mutually orthogonal, so
    # each component's centered sum of squares is its variance contribution.
    ss_obs = float(np.sum((theta_np - theta_np.mean()) ** 2))
    comp_shares = {"mu": [], "alpha": [], "beta": [], "delta_size": []}
    for s in range(S):
        comps = {
            "mu": mu[s, t_idx],
            "alpha": alpha[s, d_idx],
            "beta": beta[s, d_idx, t_idx],
            "delta_size": delta[s, d_idx] * st,
        }
        for k, v in comps.items():
            comp_shares[k].append(np.sum((v - v.mean()) ** 2) / ss_obs)
    var_rows = []
    total_share = np.zeros(S)
    for k, vals in comp_shares.items():
        vals = np.array(vals)
        total_share = total_share + vals
        l, h = np.percentile(vals, [2.5, 97.5])
        var_rows.append(dict(component=k, mean_share=round(float(vals.mean()), 4),
                             ci_lo=round(float(l), 4), ci_hi=round(float(h), 4)))
    l, h = np.percentile(total_share, [2.5, 97.5])
    var_rows.append(dict(component="total_explained", mean_share=round(float(total_share.mean()), 4),
                         ci_lo=round(float(l), 4), ci_hi=round(float(h), 4)))
    pd.DataFrame(var_rows).to_csv(OUT / "bayesian_variance_components.csv", index=False)
    print("\nPosterior variance shares (with 95% credible intervals):")
    print(pd.DataFrame(var_rows).to_string(index=False))

    # ── diagnostics summary ───────────────────────────────────────────
    pd.DataFrame([dict(max_rhat=round(max_rhat, 4), min_ess_bulk=round(min_ess, 0),
                       divergences=n_div, n_draws=S, n_cells=len(cal),
                       se_source="event_clustered")]).to_csv(
        OUT / "bayesian_diagnostics.csv", index=False)

    try:
        trace.to_netcdf(str(OUT / "trace_decomp.nc"))
    except Exception as e:
        print(f"  (could not save trace: {e})")

    print(f"\nDONE — outputs in {OUT}/")


if __name__ == "__main__":
    main()
