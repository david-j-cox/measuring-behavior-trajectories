"""Data loading from Firebase Firestore or local CSV/JSON files."""

import json
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np


# ---- Column name mapping (Firestore camelCase -> snake_case) ----

COLUMN_MAP = {
    "participantId": "participant_id",
    "sessionId": "session_id",
    "prolificId": "prolific_id",
    "timestampMsFromTaskStart": "timestamp_ms",
    "absoluteTimestamp": "absolute_timestamp",
    "clickIndex": "click_index",
    "chosenOption": "chosen_option",
    "rewardOutcome": "reward_outcome",
    "pointsEarnedThisClick": "points_earned",
    "cumulativeScore": "cumulative_score",
    "timeSincePrevClickMs": "time_since_prev_click_ms",
    "phaseId": "phase_id",
    "phaseLabel": "phase_label",
    "latentValueAPreClick": "latent_value_a_pre",
    "latentValueBPreClick": "latent_value_b_pre",
    "latentValueAPostClick": "latent_value_a_post",
    "latentValueBPostClick": "latent_value_b_post",
    "rewardProbabilityUsed": "reward_probability_used",
    "activeBonusPulse": "active_bonus_pulse",
    "bonusTarget": "bonus_target",
    "runLengthCurrentOption": "run_length",
    "timeSinceLastSwitchMs": "time_since_last_switch_ms",
    "switchFlag": "switch_flag",
    "totalClicksSoFar": "total_clicks_so_far",
    "totalRewardsSoFar": "total_rewards_so_far",
}

META_FIELDS = [
    "participant_id", "prolific_id", "session_id", "taskVersion",
    "browser", "viewportWidth", "viewportHeight", "startTime", "endTime",
    "consentGiven", "practiceCompleted", "totalDurationCompletedMs",
    "finalScore", "totalClicks", "totalRewards", "totalSwitches",
    "overallRewardRate", "totalInactivityMs", "longestGapMs",
    "inactivityGapCount", "uniqueOptionsChosen", "zeroSwitches",
    "validityFlags", "passedValidityCheck", "isTestSession",
]


def load_from_firebase(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load all sessions from Firestore and return (events_df, meta_df)."""
    import firebase_admin
    from firebase_admin import credentials, firestore

    key_path = config.get("firebase_service_account_key", "service-account-key.json")
    if not os.path.exists(key_path):
        raise FileNotFoundError(
            f"Firebase service account key not found: {key_path}\n"
            "Download from: Firebase Console > Project Settings > Service Accounts"
        )

    if not firebase_admin._apps:
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    collection = config.get("firebase_collection", "sessions")
    docs = db.collection(collection).stream()

    all_events = []
    all_meta = []

    for doc in docs:
        data = doc.to_dict()
        meta = data.get("metadata", {})
        events = data.get("events", [])

        session_id = meta.get("sessionId", doc.id)
        participant_id = meta.get("participantId", "unknown")

        # Skip test sessions if configured
        if config.get("exclude_test_sessions", True) and meta.get("isTestSession"):
            continue

        # Build metadata row
        meta_row = {
            "session_id": session_id,
            "participant_id": participant_id,
            "prolific_id": meta.get("prolificId", ""),
        }
        for field in META_FIELDS:
            camel = field
            if field not in meta:
                # Try camelCase variants
                pass
            meta_row[field] = meta.get(field, meta.get(camel))
        all_meta.append(meta_row)

        # Build event rows
        for event in events:
            row = {"participant_id": participant_id, "session_id": session_id}
            for src_col, dst_col in COLUMN_MAP.items():
                if src_col in event:
                    row[dst_col] = event[src_col]
                elif dst_col in event:
                    row[dst_col] = event[dst_col]
            all_events.append(row)

    events_df = pd.DataFrame(all_events)
    meta_df = pd.DataFrame(all_meta)

    if len(events_df) > 0:
        events_df = _standardize_types(events_df)

    print(f"Loaded {len(meta_df)} sessions, {len(events_df)} total events from Firestore.")
    return events_df, meta_df


def load_from_local(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load from a directory of CSV or JSON files."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_events = []
    all_meta = []

    for filepath in sorted(data_path.glob("*")):
        if filepath.suffix == ".json":
            events_df, meta_row = _load_json_file(filepath)
        elif filepath.suffix == ".csv":
            events_df, meta_row = _load_csv_file(filepath)
        else:
            continue

        if events_df is not None and len(events_df) > 0:
            all_events.append(events_df)
        if meta_row is not None:
            all_meta.append(meta_row)

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    meta_df = pd.DataFrame(all_meta) if all_meta else pd.DataFrame()

    if len(events_df) > 0:
        events_df = _standardize_columns(events_df)
        events_df = _standardize_types(events_df)

    print(f"Loaded {len(meta_df)} sessions, {len(events_df)} total events from {data_dir}.")
    return events_df, meta_df


def _load_json_file(filepath: Path) -> Tuple[pd.DataFrame, dict]:
    """Load a single JSON file (full data object with metadata + events)."""
    with open(filepath) as f:
        data = json.load(f)

    if "events" in data and "metadata" in data:
        events_df = pd.DataFrame(data["events"])
        meta = data["metadata"]
        meta["session_id"] = meta.get("sessionId", filepath.stem)
        meta["participant_id"] = meta.get("participantId", "unknown")
        return events_df, meta
    elif isinstance(data, list):
        return pd.DataFrame(data), None
    else:
        return pd.DataFrame(), None


def _load_csv_file(filepath: Path) -> Tuple[pd.DataFrame, dict]:
    """Load a single CSV file."""
    df = pd.read_csv(filepath)
    return df, None


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename camelCase columns to snake_case using COLUMN_MAP."""
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    return df.rename(columns=rename)


def _standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct dtypes for key columns."""
    int_cols = [
        "timestamp_ms", "click_index", "reward_outcome", "points_earned",
        "cumulative_score", "phase_id",
        "active_bonus_pulse", "run_length",
        "total_clicks_so_far", "total_rewards_so_far"
    ]
    # These columns legitimately have NaN on the first click of each session;
    # keep them as float so NaN is preserved rather than coerced to 0.
    nullable_int_cols = [
        "time_since_prev_click_ms", "time_since_last_switch_ms", "switch_flag"
    ]
    float_cols = [
        "latent_value_a_pre", "latent_value_b_pre",
        "latent_value_a_post", "latent_value_b_post",
        "reward_probability_used"
    ]
    str_cols = ["participant_id", "session_id", "chosen_option", "phase_label", "bonus_target"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in nullable_int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    return df


def load_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch to the correct loader based on config."""
    source = config.get("data_source", "local")
    if source == "firebase":
        return load_from_firebase(config)
    else:
        return load_from_local(config.get("local_data_dir", "./data"))
