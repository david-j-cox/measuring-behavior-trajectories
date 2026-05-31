"""Session-level summary metrics."""

import pandas as pd
import numpy as np
from scipy.stats import entropy


def compute_session_metrics(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute summary metrics per session."""
    rows = []
    phase_boundaries = {p["id"]: p for p in config.get("phase_boundaries", [])}

    for sid, sdf in df.groupby("session_id"):
        sdf = sdf.sort_values("timestamp_ms")
        pid = sdf["participant_id"].iloc[0]
        n = len(sdf)

        row = {
            "session_id": sid,
            "participant_id": pid,
            "total_clicks": n,
            "session_duration_s": sdf["elapsed_time_s"].max(),
            "final_score": sdf["cumulative_score"].iloc[-1] if "cumulative_score" in sdf else 0,
            "total_rewards": sdf["reward_outcome"].sum(),
            "overall_reward_rate": sdf["reward_outcome"].mean(),
            "overall_switch_rate": sdf["switch_flag_verified"].mean() if "switch_flag_verified" in sdf else 0,
            "total_switches": sdf["switch_flag_verified"].sum() if "switch_flag_verified" in sdf else 0,
            "mean_run_length": _mean_completed_run_length(sdf) if "run_length_current" in sdf else 0,
            "median_run_length": _median_completed_run_length(sdf) if "run_length_current" in sdf else 0,
            "mean_ici_s": sdf["ici_s"].mean() if "ici_s" in sdf else 0,
            "median_ici_s": sdf["ici_s"].median() if "ici_s" in sdf else 0,
            "choice_prop_a": sdf["choice_a"].mean() if "choice_a" in sdf else 0.5,
        }

        # Choice entropy
        prop_a = row["choice_prop_a"]
        if 0 < prop_a < 1:
            row["choice_entropy"] = entropy([prop_a, 1 - prop_a], base=2)
        else:
            row["choice_entropy"] = 0.0

        # Optimality
        if "choice_optimal" in sdf.columns:
            row["proportion_optimal"] = sdf["choice_optimal"].mean()
            row["mean_regret"] = sdf["regret_proxy"].mean() if "regret_proxy" in sdf else np.nan
            row["cumulative_regret"] = sdf["regret_proxy"].sum() if "regret_proxy" in sdf else np.nan

        # Win-stay / lose-shift rates
        if "win_stay" in sdf.columns:
            row["win_stay_rate"] = sdf["win_stay"].dropna().mean()
            row["lose_shift_rate"] = sdf["lose_shift"].dropna().mean()

        # Per-phase metrics
        for pid_phase, pinfo in phase_boundaries.items():
            phase_df = sdf[sdf["phase_id"] == pid_phase]
            prefix = f"phase{pid_phase}"
            if len(phase_df) > 0:
                row[f"{prefix}_clicks"] = len(phase_df)
                row[f"{prefix}_choice_prop_a"] = phase_df["choice_a"].mean()
                row[f"{prefix}_reward_rate"] = phase_df["reward_outcome"].mean()
                row[f"{prefix}_switch_rate"] = phase_df["switch_flag_verified"].mean() if "switch_flag_verified" in phase_df else 0
                row[f"{prefix}_mean_run_length"] = _mean_completed_run_length(phase_df) if "run_length_current" in phase_df else 0
                row[f"{prefix}_mean_ici_s"] = phase_df["ici_s"].mean() if "ici_s" in phase_df else 0
                if "choice_optimal" in phase_df.columns:
                    row[f"{prefix}_proportion_optimal"] = phase_df["choice_optimal"].mean()
            else:
                row[f"{prefix}_clicks"] = 0

        # Adaptation lag at phase transitions
        for transition_from, transition_to in [(1, 2), (2, 3), (3, 4)]:
            lag = _compute_adaptation_lag(
                sdf, phase_boundaries, transition_to, config
            )
            row[f"adaptation_lag_phase{transition_to}_s"] = lag

        rows.append(row)

    return pd.DataFrame(rows)


def _completed_run_lengths(sdf: pd.DataFrame) -> np.ndarray:
    """Extract lengths of completed runs from the cumulative run counter.

    A run is "completed" at the click just before a switch, or at the last
    click of the session. For example, run_length_current = [1,2,3,1,2,1]
    yields completed runs [3, 2, 1].
    """
    rl = sdf["run_length_current"].values
    if len(rl) == 0:
        return np.array([])
    # A run ends wherever the next value resets to 1, or at the end
    boundaries = np.where(np.diff(rl) < 0)[0]  # indices just before reset
    lengths = list(rl[boundaries])
    lengths.append(rl[-1])  # final run
    return np.array(lengths)


def _mean_completed_run_length(sdf: pd.DataFrame) -> float:
    lengths = _completed_run_lengths(sdf)
    return float(np.mean(lengths)) if len(lengths) > 0 else 0.0


def _median_completed_run_length(sdf: pd.DataFrame) -> float:
    lengths = _completed_run_lengths(sdf)
    return float(np.median(lengths)) if len(lengths) > 0 else 0.0


def _compute_adaptation_lag(sdf: pd.DataFrame, phase_boundaries: dict,
                            target_phase_id: int, config: dict) -> float:
    """
    Compute adaptation lag: time from phase onset until rolling choice prop
    crosses the adaptation threshold toward the advantaged option.
    """
    if target_phase_id not in phase_boundaries:
        return np.nan

    pinfo = phase_boundaries[target_phase_id]
    phase_df = sdf[sdf["phase_id"] == target_phase_id].sort_values("timestamp_ms")

    if len(phase_df) < 5:
        return np.nan

    threshold = config.get("adaptation_threshold", 0.6)

    # Determine which option is advantaged
    if "rolling_choice_prop_a_clicks" not in phase_df.columns:
        return np.nan

    # Phase 2: A is better -> look for prop_a > threshold
    # Phase 3: B is better -> look for prop_a < (1-threshold)
    # Phase 4: symmetric scarcity -> return to neutral band [1-threshold, threshold]
    target_col = "rolling_choice_prop_a_clicks"
    if target_phase_id == 2:
        adapted = phase_df[target_col] > threshold
    elif target_phase_id == 3:
        adapted = phase_df[target_col] < (1 - threshold)
    elif target_phase_id == 4:
        # Adaptation = returned to symmetric range (0.4 <= prop_a <= 0.6 when threshold=0.6)
        adapted = ((phase_df[target_col] >= 1 - threshold) &
                   (phase_df[target_col] <= threshold))
    else:
        return np.nan

    if adapted.any():
        first_adapted_time = phase_df.loc[adapted.idxmax(), "timestamp_ms"]
        lag_ms = first_adapted_time - pinfo["start_ms"]
        return lag_ms / 1000
    else:
        return np.nan  # Never adapted
