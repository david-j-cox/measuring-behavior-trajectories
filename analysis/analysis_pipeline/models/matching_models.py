"""Generalized matching law models."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import linregress


def fit_matching_models(events_df: pd.DataFrame, metrics_df: pd.DataFrame,
                        config: dict) -> pd.DataFrame:
    """Fit matching law models per session."""
    window = config.get("rolling_window_clicks", 20)
    rows = []

    for sid, sdf in events_df.groupby("session_id"):
        sdf = sdf.sort_values("timestamp_ms").reset_index(drop=True)

        if len(sdf) < 30:
            continue

        # Strict matching (log ratio regression)
        strict_result = _fit_strict_matching(sdf, window)
        if strict_result is not None:
            rows.append({
                "session_id": sid,
                "model": "strict_matching",
                "n_params": 2,
                "sensitivity": strict_result["sensitivity"],
                "bias": strict_result["bias"],
                "r_squared": strict_result["r_squared"],
                "n_bins": strict_result["n_bins"],
            })

        # Generalized matching (power function)
        gen_result = _fit_generalized_matching(sdf, window)
        if gen_result is not None:
            rows.append({
                "session_id": sid,
                "model": "generalized_matching",
                "n_params": 2,
                "sensitivity": gen_result["sensitivity"],
                "bias": gen_result["bias"],
                "r_squared": gen_result["r_squared"],
                "n_bins": gen_result["n_bins"],
            })

    return pd.DataFrame(rows)


def _fit_strict_matching(sdf: pd.DataFrame, window: int) -> dict:
    """
    Strict matching law: log(B_A/B_B) = s * log(R_A/R_B) + log(b)
    Using local estimates in rolling windows.
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


def _fit_generalized_matching(sdf: pd.DataFrame, window: int) -> dict:
    """
    Generalized matching: B_A/(B_A+B_B) = (R_A^s) / (R_A^s + b*R_B^s)
    Fit via log-odds regression.
    """
    n = len(sdf)
    n_bins = n // window
    if n_bins < 3:
        return None

    prop_a_vals = []
    rew_rate_a_vals = []
    rew_rate_b_vals = []

    for i in range(n_bins):
        start = i * window
        end = start + window
        chunk = sdf.iloc[start:end]

        count_a = chunk["choice_a"].sum()
        count_b = (1 - chunk["choice_a"]).sum()
        rew_a = (chunk["choice_a"] * chunk["reward_outcome"]).sum()
        rew_b = ((1 - chunk["choice_a"]) * chunk["reward_outcome"]).sum()

        prop_a = count_a / window
        rate_a = rew_a / max(count_a, 1)
        rate_b = rew_b / max(count_b, 1)

        if rate_a > 0 and rate_b > 0 and 0 < prop_a < 1:
            prop_a_vals.append(prop_a)
            rew_rate_a_vals.append(rate_a)
            rew_rate_b_vals.append(rate_b)

    if len(prop_a_vals) < 3:
        return None

    # Log-odds of choice vs log-odds of reward rate
    prop_a = np.array(prop_a_vals)
    rate_a = np.array(rew_rate_a_vals)
    rate_b = np.array(rew_rate_b_vals)

    y = np.log(prop_a / (1 - prop_a))
    x = np.log(rate_a / rate_b)

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    return {
        "sensitivity": slope,
        "bias": np.exp(intercept),
        "r_squared": r_value ** 2,
        "n_bins": len(x),
    }
