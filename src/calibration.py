"""Core statistical methods: logistic recalibration, decomposition, bootstrap."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.config import BIN_LABELS, C_REG, DOMAINS, SIZE_LABELS


def fit_logistic(prices, outcomes, weights, C=C_REG):
    """Fit logistic recalibration: logit(P(y=1)) = a + b*logit(p/100).

    Returns (slope_b, intercept_a, slope_se) or None if insufficient data.
    """
    from sklearn.linear_model import LogisticRegression

    X = logit(np.clip(prices / 100.0, 0.01, 0.99))
    y = outcomes.astype(int)
    w = weights.astype(float)

    if len(np.unique(y)) < 2:
        return None

    clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=2000,
                             fit_intercept=True, warm_start=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X.reshape(-1, 1), y, sample_weight=w)

    b = float(clf.coef_[0][0])
    a = float(clf.intercept_[0])

    # SE via Fisher information + L2 penalty Hessian
    eta = np.clip(a + b * X, -30, 30)
    p = expit(eta)
    W = w * p * (1 - p)
    X_aug = np.column_stack([np.ones(len(X)), X])
    fisher = (X_aug.T * W) @ X_aug
    fisher[1, 1] += 1.0 / C  # L2 penalty on slope only
    try:
        cov = np.linalg.inv(fisher)
        se = float(np.sqrt(max(cov[1, 1], 0)))
    except np.linalg.LinAlgError:
        se = np.nan

    return b, a, se


def fit_slope(prices, outcomes, weights):
    """Weighted logistic regression slope via MLE (no sklearn dependency).

    Returns slope float or NaN on failure.
    """
    from scipy.optimize import minimize

    if len(prices) < 5:
        return np.nan
    lp = logit(np.clip(prices / 100.0, 0.01, 0.99))
    y = outcomes.astype(float)
    w = weights.astype(float)
    if len(np.unique(y[w > 0])) < 2:
        return np.nan

    def neg_ll(params):
        a, b = params
        z = np.clip(a + b * lp, -30, 30)
        return -np.sum(w * (y * z - np.log1p(np.exp(z))))

    result = minimize(neg_ll, [0.0, 1.0], method="L-BFGS-B", options={"maxiter": 200})
    return result.x[1] if result.success else np.nan


def fit_slope_with_se(prices, outcomes, weights):
    """Return (slope, intercept, stderr) via MLE, or None on failure."""
    from scipy.optimize import minimize

    if len(prices) < 10:
        return None
    lp = logit(np.clip(prices / 100.0, 0.01, 0.99))
    y = outcomes.astype(float)
    w = weights.astype(float)
    if len(np.unique(y[w > 0])) < 2:
        return None

    def neg_ll(params):
        a, b = params
        z = np.clip(a + b * lp, -30, 30)
        return -np.sum(w * (y * z - np.log1p(np.exp(z))))

    result = minimize(neg_ll, [0.0, 1.0], method="L-BFGS-B", options={"maxiter": 200})
    if not result.success:
        return None
    a, b = result.x
    # Approximate SE from Hessian
    z = np.clip(a + b * lp, -30, 30)
    p_hat = 1 / (1 + np.exp(-z))
    v = w * p_hat * (1 - p_hat)
    X = np.column_stack([np.ones_like(lp), lp])
    H = X.T @ (v[:, None] * X)
    try:
        cov = np.linalg.inv(H)
        se = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        se = np.nan
    return b, a, se


def cluster_robust_slope_se(prices, outcomes, weights, cluster_ids, a, b, C=C_REG):
    """Cluster-robust (CR1) sandwich SE for the logistic recalibration slope,
    clustering by ``cluster_ids`` (event_ticker).

    Accounts for the fact that trades are not independent: many trades occur
    within the same contract/event, and yes/no contracts in the same event are
    correlated. The naive Fisher-information SE in ``fit_logistic`` treats every
    (weighted) trade as independent and badly understates uncertainty; this
    estimator instead sums the score contributions within each event cluster.

    Args mirror fit_logistic; (a, b) are the already-fitted intercept/slope.
    Returns the clustered slope SE (float) or NaN on failure.
    """
    x = logit(np.clip(prices / 100.0, 0.01, 0.99))
    y = outcomes.astype(float)
    w = weights.astype(float)
    eta = np.clip(a + b * x, -30, 30)
    p = expit(eta)

    X = np.column_stack([np.ones_like(x), x])           # (N, 2)
    # Bread: penalized Hessian of the weighted log-likelihood
    W = w * p * (1 - p)
    bread_inv = (X.T * W) @ X
    bread_inv[1, 1] += 1.0 / C                            # L2 penalty on slope
    try:
        bread = np.linalg.inv(bread_inv)
    except np.linalg.LinAlgError:
        return np.nan

    # Meat: sum of squared per-cluster score sums. Score_i = w_i (y_i - p_i) x_i
    s = (w * (y - p))[:, None] * X                        # (N, 2)
    codes, _ = pd.factorize(cluster_ids)
    G = codes.max() + 1
    if G < 2:
        return np.nan
    meat = np.zeros((2, 2))
    Ug = np.zeros((G, 2))
    np.add.at(Ug, codes, s)
    meat = Ug.T @ Ug

    V = bread @ meat @ bread
    # CR1 finite-sample correction
    N, k = len(x), 2
    corr = (G / (G - 1.0)) * ((N - 1.0) / (N - k)) if N > k else 1.0
    var_b = corr * V[1, 1]
    return float(np.sqrt(var_b)) if var_b > 0 else np.nan


def decompose(mat, domains=None, bin_labels=None, size_labels=None):
    """Sequential (Type I) decomposition on the calibration slope matrix.

    Five additive terms in the manuscript's Type I order: universal horizon
    mu(tau), domain intercept alpha_d, common size main effect kappa(s),
    domain-by-horizon beta_d(tau) and domain-by-size gamma_d(s). On the
    balanced 6x9x4 grid this sequential mean-removal is identical to an OLS
    Type I sums-of-squares fit, so squared component values reproduce the
    regression-based variance decomposition (Table 4 / Table B.5).

    Returns mat with added columns: mu, alpha, kappa, beta, gamma,
    fitted, residual.
    """
    if domains is None:
        domains = DOMAINS
    if bin_labels is None:
        bin_labels = BIN_LABELS
    if size_labels is None:
        size_labels = SIZE_LABELS

    theta = mat["slope_b"].values.copy()

    # 1. mu(tau): mean slope at each time bin (over all domain x size cells)
    mu_map = {}
    for tl in bin_labels:
        m = mat["time_bin"] == tl
        if m.any():
            mu_map[tl] = theta[m.values].mean()
    mat["mu"] = mat["time_bin"].map(mu_map).fillna(0.0)

    # 2. alpha_d: domain intercept (cross-horizon, cross-size mean deviation)
    r1 = theta - mat["mu"].values
    alpha_map = {}
    for d in domains:
        m = mat["domain"] == d
        if m.any():
            alpha_map[d] = r1[m.values].mean()
    mat["alpha"] = mat["domain"].map(alpha_map).fillna(0.0)

    # 3. kappa(s): common trade-size main effect (shared across domains)
    r2 = r1 - mat["alpha"].values
    kappa_map = {}
    for sl in size_labels:
        m = mat["size_bin"] == sl
        if m.any():
            kappa_map[sl] = r2[m.values].mean()
    mat["kappa"] = mat["size_bin"].map(kappa_map).fillna(0.0)

    # 4. beta_d(tau): domain-by-horizon interaction
    r3 = r2 - mat["kappa"].values
    beta_map = {}
    for d in domains:
        for tl in bin_labels:
            m = (mat["domain"] == d) & (mat["time_bin"] == tl)
            if m.any():
                beta_map[(d, tl)] = r3[m.values].mean()
    mat["beta"] = mat.apply(lambda r: beta_map.get((r["domain"], r["time_bin"]), 0.0), axis=1)

    # 5. gamma_d(s): domain-by-size interaction (after common size effect)
    r4 = r3 - mat["beta"].values
    gamma_map = {}
    for d in domains:
        for sl in size_labels:
            m = (mat["domain"] == d) & (mat["size_bin"] == sl)
            if m.any():
                gamma_map[(d, sl)] = r4[m.values].mean()
    mat["gamma"] = mat.apply(lambda r: gamma_map.get((r["domain"], r["size_bin"]), 0.0), axis=1)

    mat["fitted"] = mat["mu"] + mat["alpha"] + mat["kappa"] + mat["beta"] + mat["gamma"]
    mat["residual"] = theta - mat["fitted"].values

    return mat


def bootstrap_whale_effect(cal, domain, n_iter=10000, seed=42):
    """Bootstrap the Δ(Large - Single) whale effect for a given domain.

    Returns (obs_diff, ci_lo, ci_hi).
    """
    rng = np.random.default_rng(seed)
    large = cal[(cal["domain"] == domain) & (cal["size_bin"] == "Large")]["slope_b"].values
    single = cal[(cal["domain"] == domain) & (cal["size_bin"] == "Single")]["slope_b"].values
    if len(large) == 0 or len(single) == 0:
        return np.nan, np.nan, np.nan
    obs_diff = large.mean() - single.mean()
    boot_diffs = []
    for _ in range(n_iter):
        bl = rng.choice(large, size=len(large), replace=True)
        bs = rng.choice(single, size=len(single), replace=True)
        boot_diffs.append(bl.mean() - bs.mean())
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    return obs_diff, ci_lo, ci_hi


def compute_weighted_decomposition(slopes, time_bins, domains_arr, size_bins, w):
    """Sequential (Type I) variance decomposition with given weights.

    Five-term order matching decompose(): mu(tau), alpha_d, kappa(s),
    beta_d(tau), gamma_d(s). Returns dict with marginal R² for each
    component plus the combined size block (kappa+gamma) and total.
    """
    gm = np.average(slopes, weights=w)
    ss_total = np.sum(w * (slopes - gm) ** 2)

    # mu(tau)
    mu = {}
    for tb in BIN_LABELS:
        mask = time_bins == tb
        if mask.any():
            mu[tb] = np.average(slopes[mask], weights=w[mask]) - gm
    fitted = gm + np.array([mu.get(t, 0) for t in time_bins])
    ss_after_mu = np.sum(w * (slopes - fitted) ** 2)

    # alpha_d
    resid = slopes - fitted
    alpha = {}
    for d in DOMAINS:
        mask = domains_arr == d
        if mask.any():
            alpha[d] = np.average(resid[mask], weights=w[mask])
    fitted = fitted + np.array([alpha.get(d, 0) for d in domains_arr])
    ss_after_alpha = np.sum(w * (slopes - fitted) ** 2)

    # kappa(s): common size main effect
    resid = slopes - fitted
    kappa = {}
    for sb in SIZE_LABELS:
        mask = size_bins == sb
        if mask.any():
            kappa[sb] = np.average(resid[mask], weights=w[mask])
    fitted = fitted + np.array([kappa.get(s, 0) for s in size_bins])
    ss_after_kappa = np.sum(w * (slopes - fitted) ** 2)

    # beta_d(tau)
    resid = slopes - fitted
    beta = {}
    for d in DOMAINS:
        for tb in BIN_LABELS:
            mask = (domains_arr == d) & (time_bins == tb)
            if mask.any():
                beta[(d, tb)] = np.average(resid[mask], weights=w[mask])
    fitted = fitted + np.array([beta.get((d, t), 0) for d, t in zip(domains_arr, time_bins)])
    ss_after_beta = np.sum(w * (slopes - fitted) ** 2)

    # gamma_d(s)
    resid = slopes - fitted
    gamma = {}
    for d in DOMAINS:
        for sb in SIZE_LABELS:
            mask = (domains_arr == d) & (size_bins == sb)
            if mask.any():
                gamma[(d, sb)] = np.average(resid[mask], weights=w[mask])
    fitted = fitted + np.array([gamma.get((d, s), 0) for d, s in zip(domains_arr, size_bins)])
    ss_after_gamma = np.sum(w * (slopes - fitted) ** 2)

    r2_kappa = (ss_after_alpha - ss_after_kappa) / ss_total
    r2_gamma = (ss_after_beta - ss_after_gamma) / ss_total
    return {
        "mu": (ss_total - ss_after_mu) / ss_total,
        "alpha": (ss_after_mu - ss_after_alpha) / ss_total,
        "kappa": r2_kappa,
        "beta": (ss_after_kappa - ss_after_beta) / ss_total,
        "gamma": r2_gamma,
        "size_block": r2_kappa + r2_gamma,
        "total": (ss_total - ss_after_gamma) / ss_total,
    }
