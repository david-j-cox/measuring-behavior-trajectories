"""Generalized matching law (Baum, 1974) model fitting."""

import numpy as np
import pandas as pd
from scipy.stats import linregress


def fit_matching_models(events_df: pd.DataFrame, metrics_df: pd.DataFrame,
                        config: dict) -> pd.DataFrame:
    """Fit the generalized matching law per session.

    log(B_A/B_B) = s * log(R_A/R_B) + log(b)

    where s is sensitivity to reinforcement and b is bias.
    """
    window = config.get("rolling_window_clicks", 20)
    rows = []

    for sid, sdf in events_df.groupby("session_id"):
        sdf = sdf.sort_values("timestamp_ms").reset_index(drop=True)

        if len(sdf) < 30:
            continue

        result = _fit_generalized_matching_law(sdf, window)
        if result is not None:
            rows.append({
                "session_id": sid,
                "model": "generalized_matching_law",
                "n_params": 2,
                "sensitivity": result["sensitivity"],
                "bias": result["bias"],
                "r_squared": result["r_squared"],
                "n_bins": result["n_bins"],
            })

    return pd.DataFrame(rows)


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
