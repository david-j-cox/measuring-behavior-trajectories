"""Data validation and quality checks."""

import pandas as pd
import numpy as np
from typing import List, Tuple


def validate_events(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate event-level data. Returns (clean_df, report_df).
    report_df has columns: session_id, check, severity, message.
    """
    required_cols = [
        "participant_id", "session_id", "timestamp_ms", "click_index",
        "chosen_option", "reward_outcome", "cumulative_score", "phase_id"
    ]

    reports = []

    # Check required columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        reports.append({
            "session_id": "ALL", "check": "required_columns",
            "severity": "ERROR", "message": f"Missing columns: {missing}"
        })
        return df, pd.DataFrame(reports)

    sessions = df.groupby("session_id")
    drop_sessions = set()

    for sid, sdf in sessions:
        sdf_sorted = sdf.sort_values("timestamp_ms")

        # Timestamps monotonic
        if not sdf_sorted["timestamp_ms"].is_monotonic_increasing:
            reports.append({
                "session_id": sid, "check": "timestamps_monotonic",
                "severity": "WARNING", "message": "Timestamps not strictly monotonic"
            })

        # Click indices monotonic
        if "click_index" in sdf_sorted.columns:
            if not sdf_sorted["click_index"].is_monotonic_increasing:
                reports.append({
                    "session_id": sid, "check": "click_index_monotonic",
                    "severity": "WARNING", "message": "Click indices not strictly monotonic"
                })

        # Valid chosen_option
        invalid_choices = sdf_sorted[~sdf_sorted["chosen_option"].isin(["A", "B"])]
        if len(invalid_choices) > 0:
            reports.append({
                "session_id": sid, "check": "valid_choices",
                "severity": "ERROR",
                "message": f"{len(invalid_choices)} rows with invalid chosen_option"
            })

        # Reward outcomes binary
        if "reward_outcome" in sdf_sorted.columns:
            non_binary = sdf_sorted[~sdf_sorted["reward_outcome"].isin([0, 1])]
            if len(non_binary) > 0:
                reports.append({
                    "session_id": sid, "check": "reward_binary",
                    "severity": "ERROR",
                    "message": f"{len(non_binary)} rows with non-binary reward"
                })

        # Valid phases
        valid_phases = {p["id"] for p in config.get("phase_boundaries", [])}
        if valid_phases and "phase_id" in sdf_sorted.columns:
            invalid_phases = sdf_sorted[~sdf_sorted["phase_id"].isin(valid_phases)]
            if len(invalid_phases) > 0:
                reports.append({
                    "session_id": sid, "check": "valid_phases",
                    "severity": "WARNING",
                    "message": f"{len(invalid_phases)} rows with unexpected phase_id"
                })

        # No negative timestamps
        neg_times = sdf_sorted[sdf_sorted["timestamp_ms"] < 0]
        if len(neg_times) > 0:
            reports.append({
                "session_id": sid, "check": "no_negative_times",
                "severity": "ERROR",
                "message": f"{len(neg_times)} rows with negative timestamps"
            })

        # Cumulative score consistency
        if "points_earned" in sdf_sorted.columns:
            expected_cumsum = sdf_sorted["points_earned"].cumsum()
            actual = sdf_sorted["cumulative_score"].values
            if not np.allclose(expected_cumsum.values, actual, atol=1):
                reports.append({
                    "session_id": sid, "check": "cumulative_score_consistency",
                    "severity": "WARNING",
                    "message": "Cumulative score doesn't match sum of points earned"
                })

        # Switch flag consistency
        if "switch_flag" in sdf_sorted.columns and len(sdf_sorted) > 1:
            actual_switches = (sdf_sorted["chosen_option"].values[1:] !=
                               sdf_sorted["chosen_option"].values[:-1]).astype(int)
            logged_switches = sdf_sorted["switch_flag"].values[1:]
            mismatches = np.sum(actual_switches != logged_switches)
            if mismatches > 0:
                reports.append({
                    "session_id": sid, "check": "switch_flag_consistency",
                    "severity": "WARNING",
                    "message": f"{mismatches} switch flag mismatches"
                })

        # Minimum clicks
        min_clicks = config.get("min_clicks_per_session", 50)
        if len(sdf_sorted) < min_clicks:
            reports.append({
                "session_id": sid, "check": "min_clicks",
                "severity": "EXCLUDE",
                "message": f"Only {len(sdf_sorted)} clicks (min: {min_clicks})"
            })
            drop_sessions.add(sid)

        # Minimum duration
        min_dur_s = config.get("min_session_duration_s", 300)
        dur_s = (sdf_sorted["timestamp_ms"].max() - sdf_sorted["timestamp_ms"].min()) / 1000
        if dur_s < min_dur_s:
            reports.append({
                "session_id": sid, "check": "min_duration",
                "severity": "EXCLUDE",
                "message": f"Duration {dur_s:.0f}s (min: {min_dur_s}s)"
            })
            drop_sessions.add(sid)

    report_df = pd.DataFrame(reports)

    # Drop excluded sessions
    clean_df = df[~df["session_id"].isin(drop_sessions)].copy()

    n_excluded = len(drop_sessions)
    n_warnings = len(report_df[report_df["severity"] == "WARNING"]) if len(report_df) > 0 else 0
    n_errors = len(report_df[report_df["severity"] == "ERROR"]) if len(report_df) > 0 else 0
    print(f"Validation: {n_excluded} sessions excluded, {n_warnings} warnings, {n_errors} errors")

    return clean_df, report_df
