#!/usr/bin/env python3
"""Compute classification metrics (accuracy, F1, MCC) for ratio_invariance
and append to classification_metrics.csv.

Uses fitted omega/beta parameters from model_comparison.csv to generate
per-trial P(A) predictions, thresholds at 0.5, and computes the same
metrics already present for the other historical models.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                              precision_score, recall_score)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from pipeline.utils import load_config
from pipeline.step01_load_data import load_data
from pipeline.step02_validate_data import validate_events
from pipeline.step03_feature_engineering import compute_derived_variables
from pipeline.models.historical_dynamics import _ratio_invariance_predict

OUT = os.path.join(ANALYSIS_DIR, "data", "03_analytic_outputs")


def _metrics(y_true, y_pred):
    """Compute classification metrics, handling degenerate cases."""
    if len(y_true) == 0:
        return dict(accuracy=np.nan, precision=np.nan, recall=np.nan,
                    f1=np.nan, mcc=np.nan)
    try:
        prec = precision_score(y_true, y_pred, zero_division=0)
    except Exception:
        prec = np.nan
    try:
        rec = recall_score(y_true, y_pred, zero_division=0)
    except Exception:
        rec = np.nan
    try:
        f1 = f1_score(y_true, y_pred, zero_division=0)
    except Exception:
        f1 = np.nan
    try:
        mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 and len(set(y_pred)) > 1 else 0.0
    except Exception:
        mcc = 0.0
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        precision=prec,
        recall=rec,
        f1=f1,
        mcc=mcc,
    )


def main():
    config = load_config(os.path.join(SCRIPT_DIR, "config.yaml"))
    config["data_source"] = "local"
    config["local_data_dir"] = os.path.join(ANALYSIS_DIR, "data", "01_raw_data")

    print("Loading data...")
    events_df, _ = load_data(config)
    events_df, _ = validate_events(events_df, config)
    events_df = compute_derived_variables(events_df, config)

    mc = pd.read_csv(os.path.join(OUT, "model_comparison.csv"))
    ri_fits = mc[mc["model"] == "ratio_invariance"].copy()
    print(f"Found {len(ri_fits)} ratio_invariance fits")

    rows = []
    for _, fit in ri_fits.iterrows():
        sid = fit["session_id"]
        omega = fit.get("omega", 0.0)
        beta = fit.get("beta", 5.0)

        sdf = events_df[events_df["session_id"] == sid].sort_values(
            "timestamp_ms").reset_index(drop=True)
        if len(sdf) < 10:
            continue

        choices = sdf["choice_a"].values
        rewards = sdf["reward_outcome"].values
        phases = sdf["phase_id"].values

        p_a_seq = _ratio_invariance_predict(choices, rewards, omega, beta)
        pred = (p_a_seq > 0.5).astype(int)

        # Full session metrics
        m = _metrics(choices, pred)
        rows.append({"session_id": sid, "model": "ratio_invariance",
                     "phase": "Full", **m})

        # Per-phase metrics (labeled "Ph 1", "Ph 2", etc. to match existing file)
        for pid in [1, 2, 3, 4]:
            mask = phases == pid
            if mask.sum() < 5:
                continue
            m = _metrics(choices[mask], pred[mask])
            rows.append({"session_id": sid, "model": "ratio_invariance",
                         "phase": f"Ph {pid}", **m})

    new_df = pd.DataFrame(rows)
    print(f"Generated {len(new_df)} rows for ratio_invariance")

    # Append to existing classification_metrics.csv
    class_path = os.path.join(OUT, "classification_metrics.csv")
    existing = pd.read_csv(class_path)
    # Drop any pre-existing ratio_invariance rows (if rerunning)
    existing = existing[existing["model"] != "ratio_invariance"]

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(class_path, index=False)
    print(f"Saved {len(combined)} total rows to {class_path}")

    # Report ratio_invariance median metrics (Full session)
    full = new_df[new_df["phase"] == "Full"]
    print("\nRatio invariance median / std (Full session):")
    for metric in ["accuracy", "f1", "mcc"]:
        vals = full[metric].dropna()
        print(f"  {metric}: median={vals.median():.3f}, std={vals.std():.3f}")


if __name__ == "__main__":
    main()
