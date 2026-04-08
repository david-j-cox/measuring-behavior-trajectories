"""Foraging models based on optimal foraging theory.

Three models derived from the marginal value theorem (Charnov, 1976):
  1. MVT threshold — switch when current patch return drops below average
  2. Patch-leaving threshold — switch when return drops below fixed threshold
  3. Softmax MVT — probabilistic version of MVT with inverse temperature
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def fit_foraging_models(events_df: pd.DataFrame,
                        config: dict) -> pd.DataFrame:
    """Fit all three foraging models per session."""
    rows = []

    for sid, sdf in events_df.groupby("session_id"):
        sdf = sdf.sort_values("timestamp_ms").reset_index(drop=True)
        if len(sdf) < 50:
            continue

        choices = sdf["choice_a"].values.astype(float)
        rewards = sdf["reward_outcome"].values.astype(float)
        n = len(choices)

        # --- Model 1: MVT threshold ---
        # Switch when current patch return < average return * threshold_ratio
        # One free parameter: window (how many recent trials to estimate rates)
        best_nll_mvt = np.inf
        best_params_mvt = {}

        for window in [5, 10, 15, 20, 30]:
            nll = _mvt_nll(choices, rewards, window)
            if nll < best_nll_mvt:
                best_nll_mvt = nll
                best_params_mvt = {"window": window}

        rows.append({
            "session_id": sid,
            "model": "mvt_threshold",
            "n_params": 1,
            "nll": best_nll_mvt,
            "aic": 2 * 1 + 2 * best_nll_mvt,
            "bic": 1 * np.log(n) + 2 * best_nll_mvt,
            "n_obs": n,
            **best_params_mvt,
        })

        # --- Model 2: Patch-leaving threshold ---
        # Switch when current patch return < fixed threshold
        # One free parameter: threshold
        result = minimize(
            lambda p: _patch_leaving_nll(choices, rewards, p[0]),
            x0=[0.5], bounds=[(0.01, 0.99)], method="L-BFGS-B"
        )
        nll_pl = result.fun
        rows.append({
            "session_id": sid,
            "model": "patch_leaving",
            "n_params": 1,
            "nll": nll_pl,
            "aic": 2 * 1 + 2 * nll_pl,
            "bic": 1 * np.log(n) + 2 * nll_pl,
            "n_obs": n,
            "threshold": result.x[0],
        })

        # --- Model 3: Softmax MVT ---
        # P(stay on current) = sigmoid(beta * (current_rate - avg_rate))
        # Two free parameters: window, beta
        best_nll_smvt = np.inf
        best_params_smvt = {}

        for window in [5, 10, 15, 20, 30]:
            result = minimize(
                lambda p, w=window: _softmax_mvt_nll(
                    choices, rewards, w, p[0]),
                x0=[2.0], bounds=[(0.01, 50.0)], method="L-BFGS-B"
            )
            if result.fun < best_nll_smvt:
                best_nll_smvt = result.fun
                best_params_smvt = {"window": window, "beta": result.x[0]}

        rows.append({
            "session_id": sid,
            "model": "softmax_mvt",
            "n_params": 2,
            "nll": best_nll_smvt,
            "aic": 2 * 2 + 2 * best_nll_smvt,
            "bic": 2 * np.log(n) + 2 * best_nll_smvt,
            "n_obs": n,
            **best_params_smvt,
        })

    return pd.DataFrame(rows)


def _compute_patch_rates(choices, rewards, window):
    """Compute rolling reward rate for current patch and average rate.

    'Current patch' = the option most recently chosen.
    Returns (current_rate, avg_rate) arrays of length n.
    """
    n = len(choices)
    current_rate = np.full(n, 0.5)
    avg_rate = np.full(n, 0.5)

    for t in range(1, n):
        start = max(0, t - window)
        recent_c = choices[start:t]
        recent_r = rewards[start:t]

        # Current patch = whatever was chosen on t-1
        current_patch = choices[t - 1]
        patch_mask = recent_c == current_patch
        other_mask = ~patch_mask

        if patch_mask.sum() > 0:
            current_rate[t] = recent_r[patch_mask].mean()
        if len(recent_r) > 0:
            avg_rate[t] = recent_r.mean()

    return current_rate, avg_rate


def _mvt_nll(choices, rewards, window):
    """MVT: predict stay on current patch if current_rate >= avg_rate."""
    n = len(choices)
    current_rate, avg_rate = _compute_patch_rates(choices, rewards, window)
    eps = 1e-6

    ll = 0.0
    for t in range(1, n):
        # Predict: stay on current patch if its rate >= average
        current_patch = choices[t - 1]
        stay = current_rate[t] >= avg_rate[t]

        if stay:
            p_same = 0.8  # high prob of staying
        else:
            p_same = 0.2  # high prob of switching

        # Did they actually stay?
        actual_stay = (choices[t] == current_patch)
        p = p_same if actual_stay else (1 - p_same)
        ll += np.log(max(p, eps))

    return -ll


def _patch_leaving_nll(choices, rewards, threshold, window=15):
    """Patch-leaving: switch when current patch rate < threshold."""
    n = len(choices)
    current_rate, _ = _compute_patch_rates(choices, rewards, window)
    eps = 1e-6

    ll = 0.0
    for t in range(1, n):
        current_patch = choices[t - 1]
        stay = current_rate[t] >= threshold

        if stay:
            p_same = 0.8
        else:
            p_same = 0.2

        actual_stay = (choices[t] == current_patch)
        p = p_same if actual_stay else (1 - p_same)
        ll += np.log(max(p, eps))

    return -ll


def _softmax_mvt_nll(choices, rewards, window, beta):
    """Softmax MVT: P(stay) = sigmoid(beta * (current_rate - avg_rate))."""
    n = len(choices)
    current_rate, avg_rate = _compute_patch_rates(choices, rewards, window)
    eps = 1e-6

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    ll = 0.0
    for t in range(1, n):
        current_patch = choices[t - 1]
        p_stay = sigmoid(beta * (current_rate[t] - avg_rate[t]))

        actual_stay = (choices[t] == current_patch)
        p = p_stay if actual_stay else (1 - p_stay)
        p = max(p, eps)
        ll += np.log(p)

    return -ll


def predict_foraging_pa(model_name, params, sdf):
    """Reconstruct trial-level P(A) for classification metrics."""
    choices = sdf["choice_a"].values.astype(float)
    rewards = sdf["reward_outcome"].values.astype(float)
    n = len(choices)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    if model_name == "mvt_threshold":
        window = int(params.get("window", 15))
        current_rate, avg_rate = _compute_patch_rates(
            choices, rewards, window)
        pa = np.full(n, 0.5)
        for t in range(1, n):
            current_patch = choices[t - 1]
            stay = current_rate[t] >= avg_rate[t]
            if current_patch == 1:  # currently on A
                pa[t] = 0.8 if stay else 0.2
            else:  # currently on B
                pa[t] = 0.2 if stay else 0.8
        return pa

    elif model_name == "patch_leaving":
        threshold = params.get("threshold", 0.5)
        current_rate, _ = _compute_patch_rates(choices, rewards, 15)
        pa = np.full(n, 0.5)
        for t in range(1, n):
            current_patch = choices[t - 1]
            stay = current_rate[t] >= threshold
            if current_patch == 1:
                pa[t] = 0.8 if stay else 0.2
            else:
                pa[t] = 0.2 if stay else 0.8
        return pa

    elif model_name == "softmax_mvt":
        window = int(params.get("window", 15))
        beta = params.get("beta", 2.0)
        current_rate, avg_rate = _compute_patch_rates(
            choices, rewards, window)
        pa = np.full(n, 0.5)
        for t in range(1, n):
            current_patch = choices[t - 1]
            p_stay = sigmoid(beta * (current_rate[t] - avg_rate[t]))
            if current_patch == 1:
                pa[t] = p_stay
            else:
                pa[t] = 1 - p_stay
        return pa

    return np.full(n, 0.5)
