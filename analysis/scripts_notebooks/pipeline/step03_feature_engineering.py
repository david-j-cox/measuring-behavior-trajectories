"""Derived variable engineering at the click level."""

import pandas as pd
import numpy as np


def compute_derived_variables(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add all derived columns to the event-level DataFrame."""
    df = df.sort_values(["session_id", "timestamp_ms"]).copy()

    # Basic time variables
    df["elapsed_time_s"] = df["timestamp_ms"] / 1000
    df["ici_s"] = df["time_since_prev_click_ms"] / 1000

    # Choice indicators
    df["choice_a"] = (df["chosen_option"] == "A").astype(int)
    df["choice_b"] = (df["chosen_option"] == "B").astype(int)

    # Verified switch flag (vectorized)
    df["switch_flag_verified"] = 0
    shifted = df.groupby("session_id")["chosen_option"].shift(1)
    mask = shifted.notna()
    df.loc[mask, "switch_flag_verified"] = (
        df.loc[mask, "chosen_option"] != shifted[mask]
    ).astype(int)

    # Run lengths (vectorized)
    df["run_length_current"] = _compute_run_lengths(df)

    # Previous run length
    df["run_length_previous"] = _compute_previous_run_lengths(df)

    # Time since last reward
    df["time_since_last_reward_s"] = _compute_time_since_event(
        df, event_col="reward_outcome", event_val=1
    )

    # Time since last switch
    df["time_since_last_switch_s"] = _compute_time_since_event(
        df, event_col="switch_flag_verified", event_val=1
    )

    # Rolling windows
    win_clicks = config.get("rolling_window_clicks", 20)
    win_time_s = config.get("rolling_window_time_s", 15)

    df["rolling_choice_prop_a_clicks"] = _rolling_by_clicks(df, "choice_a", win_clicks)
    df["rolling_reward_rate_clicks"] = _rolling_by_clicks(df, "reward_outcome", win_clicks)
    df["rolling_switch_rate_clicks"] = _rolling_by_clicks(df, "switch_flag_verified", win_clicks)
    df["rolling_mean_ici_clicks"] = _rolling_by_clicks(df, "ici_s", win_clicks, func="mean")

    df["rolling_choice_prop_a_time"] = _rolling_by_time(df, "choice_a", win_time_s)
    df["rolling_reward_rate_time"] = _rolling_by_time(df, "reward_outcome", win_time_s)

    # Local reward estimates per option (recent window)
    df = _compute_local_reward_estimates(df, win_clicks)

    # Win-stay / lose-shift indicators
    df = _compute_wslf(df)

    # Latent advantage
    if "latent_value_a_pre" in df.columns and "latent_value_b_pre" in df.columns:
        df["latent_advantage_pre"] = df["latent_value_a_pre"] - df["latent_value_b_pre"]
        df["optimal_choice"] = np.where(
            df["latent_value_a_pre"] >= df["latent_value_b_pre"], "A", "B"
        )
        df["choice_optimal"] = (df["chosen_option"] == df["optimal_choice"]).astype(int)
        df["regret_proxy"] = np.abs(
            np.maximum(df["latent_value_a_pre"], df["latent_value_b_pre"]) -
            np.where(df["chosen_option"] == "A",
                     df["latent_value_a_pre"], df["latent_value_b_pre"])
        )
        df["cumulative_regret"] = df.groupby("session_id")["regret_proxy"].cumsum()

    # Phase time
    phase_boundaries = {
        p["id"]: p for p in config.get("phase_boundaries", [])
    }
    df["phase_time_elapsed_s"] = 0.0
    df["time_to_phase_end_s"] = 0.0
    for pid, pinfo in phase_boundaries.items():
        mask = df["phase_id"] == pid
        df.loc[mask, "phase_time_elapsed_s"] = (
            df.loc[mask, "timestamp_ms"] - pinfo["start_ms"]
        ) / 1000
        df.loc[mask, "time_to_phase_end_s"] = (
            pinfo["end_ms"] - df.loc[mask, "timestamp_ms"]
        ) / 1000

    return df


def _compute_run_lengths(df: pd.DataFrame) -> pd.Series:
    """Compute consecutive same-option run length at each click."""
    result = pd.Series(1, index=df.index)
    for sid, grp in df.groupby("session_id"):
        idx = grp.index
        choices = grp["chosen_option"].values
        runs = np.ones(len(choices), dtype=int)
        for i in range(1, len(choices)):
            if choices[i] == choices[i - 1]:
                runs[i] = runs[i - 1] + 1
            else:
                runs[i] = 1
        result.loc[idx] = runs
    return result


def _compute_previous_run_lengths(df: pd.DataFrame) -> pd.Series:
    """Length of the run that ended at the most recent switch."""
    result = pd.Series(0, index=df.index)
    for sid, grp in df.groupby("session_id"):
        idx = grp.index
        choices = grp["chosen_option"].values
        prev_run = 0
        current_run = 1
        vals = np.zeros(len(choices), dtype=int)
        for i in range(1, len(choices)):
            if choices[i] != choices[i - 1]:
                prev_run = current_run
                current_run = 1
            else:
                current_run += 1
            vals[i] = prev_run
        result.loc[idx] = vals
    return result


def _compute_time_since_event(df: pd.DataFrame, event_col: str, event_val) -> pd.Series:
    """Time (s) since the last occurrence of event_val in event_col."""
    result = pd.Series(np.nan, index=df.index)
    for sid, grp in df.groupby("session_id"):
        idx = grp.index
        times = grp["elapsed_time_s"].values
        events = grp[event_col].values
        last_time = np.nan
        vals = np.full(len(times), np.nan)
        for i in range(len(times)):
            if events[i] == event_val:
                last_time = times[i]
            if not np.isnan(last_time):
                vals[i] = times[i] - last_time
        result.loc[idx] = vals
    return result


def _rolling_by_clicks(df: pd.DataFrame, col: str, window: int,
                       func: str = "mean") -> pd.Series:
    """Rolling statistic over a click-based window, per session."""
    grouped = df.groupby("session_id")[col]
    if func == "mean":
        return grouped.transform(lambda x: x.rolling(window, min_periods=1).mean())
    elif func == "sum":
        return grouped.transform(lambda x: x.rolling(window, min_periods=1).sum())
    return pd.Series(np.nan, index=df.index)


def _rolling_by_time(df: pd.DataFrame, col: str, window_s: float) -> pd.Series:
    """Rolling mean over a time-based window, per session.

    Uses pandas time-based rolling with a TimedeltaIndex for vectorized
    computation instead of a per-row Python loop.
    """
    result = pd.Series(np.nan, index=df.index)
    for sid, grp in df.groupby("session_id"):
        idx = grp.index
        # Create a temporary series indexed by timedelta for time-based rolling
        td_index = pd.to_timedelta(grp["elapsed_time_s"].values, unit="s")
        ts = pd.Series(grp[col].values, index=td_index)
        rolled = ts.rolling(f"{window_s}s", min_periods=1).mean()
        result.loc[idx] = rolled.values
    return result


def _compute_local_reward_estimates(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute local reward rate per option in recent window.

    Uses vectorized pandas rolling sums instead of per-row Python loops.
    """
    grouped = df.groupby("session_id")

    # Rolling counts and reward sums per option via vectorized rolling
    df["local_count_a"] = grouped["choice_a"].transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )
    df["local_count_b"] = grouped["choice_b"].transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )

    reward_a = df["choice_a"] * df["reward_outcome"]
    reward_b = df["choice_b"] * df["reward_outcome"]

    df["local_rewards_a"] = reward_a.groupby(df["session_id"]).transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )
    df["local_rewards_b"] = reward_b.groupby(df["session_id"]).transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )

    # Compute rates (NaN where count is 0)
    df["local_reward_rate_a"] = np.where(
        df["local_count_a"] > 0,
        df["local_rewards_a"] / df["local_count_a"],
        np.nan
    )
    df["local_reward_rate_b"] = np.where(
        df["local_count_b"] > 0,
        df["local_rewards_b"] / df["local_count_b"],
        np.nan
    )

    total = df["local_count_a"] + df["local_count_b"]
    df["choice_bias_recent"] = np.where(
        total > 0, (df["local_count_a"] - df["local_count_b"]) / total, 0
    )

    return df


def _compute_wslf(df: pd.DataFrame) -> pd.DataFrame:
    """Compute win-stay, lose-shift, stay-after-reward, stay-after-no-reward."""
    df["stay"] = 0
    df["win_stay"] = np.nan
    df["lose_shift"] = np.nan
    df["stay_after_reward"] = np.nan
    df["stay_after_no_reward"] = np.nan

    for sid, grp in df.groupby("session_id"):
        idx = grp.index
        choices = grp["chosen_option"].values
        rewards = grp["reward_outcome"].values
        n = len(choices)

        stay = np.zeros(n, dtype=int)
        ws = np.full(n, np.nan)
        ls = np.full(n, np.nan)
        sar = np.full(n, np.nan)
        sanr = np.full(n, np.nan)

        for i in range(1, n):
            stayed = int(choices[i] == choices[i - 1])
            stay[i] = stayed
            if rewards[i - 1] == 1:
                ws[i] = stayed
                sar[i] = stayed
            else:
                ls[i] = 1 - stayed
                sanr[i] = stayed

        df.loc[idx, "stay"] = stay
        df.loc[idx, "win_stay"] = ws
        df.loc[idx, "lose_shift"] = ls
        df.loc[idx, "stay_after_reward"] = sar
        df.loc[idx, "stay_after_no_reward"] = sanr

    return df
