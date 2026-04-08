"""Generalized matching law (Baum, 1974) model fitting.

Fits the GML via OLS on binned log-ratios, then derives trial-level
choice probabilities from the fitted parameters so the model can be
compared with process models via AIC.
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress


def fit_matching_models(events_df: pd.DataFrame, metrics_df: pd.DataFrame,
                        config: dict) -> pd.DataFrame:
    """Fit the generalized matching law per session.

    log(B_A/B_B) = s * log(R_A/R_B) + log(b)

    where s is sensitivity to reinforcement and b is bias.

    Trial-level AIC is computed by using the fitted s and b to predict
    P(A) on every trial from local reward rates, then evaluating the
    binary log-likelihood.
    """
    window = config.get("rolling_window_clicks", 20)
    rows = []

    for sid, sdf in events_df.groupby("session_id"):
        sdf = sdf.sort_values("timestamp_ms").reset_index(drop=True)

        if len(sdf) < 30:
            continue

        result = _fit_generalized_matching_law(sdf, window)
        if result is not None:
            # Compute trial-level log-likelihood and AIC
            ll = _trial_level_loglik(sdf, result["sensitivity"],
                                     result["bias"])
            n_params = 2
            n_obs = len(sdf)
            nll = -ll
            aic = 2 * n_params + 2 * nll
            bic = n_params * np.log(n_obs) + 2 * nll

            rows.append({
                "session_id": sid,
                "model": "generalized_matching_law",
                "n_params": n_params,
                "nll": nll,
                "aic": aic,
                "bic": bic,
                "n_obs": n_obs,
                "sensitivity": result["sensitivity"],
                "bias": result["bias"],
                "r_squared": result["r_squared"],
                "n_bins": result["n_bins"],
            })

    return pd.DataFrame(rows)


def _trial_level_loglik(sdf, sensitivity, bias):
    """Compute trial-level log-likelihood from GML parameters.

    On each trial, local reward rates R_A and R_B are used to predict:
        log(B_A/B_B) = s * log(R_A/R_B) + log(b)
    which is converted to P(A) = predicted_ratio / (1 + predicted_ratio).
    The log-likelihood is the sum of log P(choice_i | predicted P(A)_i).
    """
    if ("local_reward_rate_a" not in sdf.columns or
            "local_reward_rate_b" not in sdf.columns):
        return np.nan

    r_a = sdf["local_reward_rate_a"].values.astype(float)
    r_b = sdf["local_reward_rate_b"].values.astype(float)
    choices = sdf["choice_a"].values.astype(float)

    eps = 1e-6
    # Replace NaN and clip reward rates away from zero for log-ratio
    r_a = np.clip(np.nan_to_num(r_a, nan=eps), eps, None)
    r_b = np.clip(np.nan_to_num(r_b, nan=eps), eps, None)

    # Predicted log(B_A/B_B)
    log_ratio = sensitivity * np.log(r_a / r_b) + np.log(bias)

    # Convert to P(A) via ratio: B_A/B_B = exp(log_ratio),
    # P(A) = ratio / (1 + ratio) = sigmoid(log_ratio)
    p_a = 1.0 / (1.0 + np.exp(-log_ratio))
    p_a = np.clip(p_a, eps, 1 - eps)

    # Binary log-likelihood
    ll = np.sum(choices * np.log(p_a) + (1 - choices) * np.log(1 - p_a))
    return ll


def _fit_generalized_matching_law(sdf: pd.DataFrame, window: int) -> dict:
    """
    Generalized matching law (Baum, 1974):
        log(B_A/B_B) = s * log(R_A/R_B) + log(b)

    Fitted via linear regression on non-overlapping bins of local
    reinforcement and response ratios.
    """
    if "local_reward_rate_a" not in sdf.columns:
        return None

    # Use non-overlapping windows
    n = len(sdf)
    n_bins = n // window
    if n_bins < 3:
        return None

    log_choice_ratios = []
    log_reward_ratios = []

    for i in range(n_bins):
        start = i * window
        end = start + window
        chunk = sdf.iloc[start:end]

        count_a = chunk["choice_a"].sum()
        count_b = (1 - chunk["choice_a"]).sum()
        rew_a = (chunk["choice_a"] * chunk["reward_outcome"]).sum()
        rew_b = ((1 - chunk["choice_a"]) * chunk["reward_outcome"]).sum()

        # Need at least 1 of each to take log ratio
        if count_a < 1 or count_b < 1 or rew_a < 1 or rew_b < 1:
            continue

        log_choice_ratios.append(np.log(count_a / count_b))
        log_reward_ratios.append(np.log(rew_a / rew_b))

    if len(log_choice_ratios) < 3:
        return None

    x = np.array(log_reward_ratios)
    y = np.array(log_choice_ratios)

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    return {
        "sensitivity": slope,
        "bias": np.exp(intercept),
        "r_squared": r_value ** 2,
        "n_bins": len(x),
    }
