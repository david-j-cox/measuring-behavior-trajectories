#!/usr/bin/env python3
"""Run BOCPD change-point detection standalone."""
import pandas as pd
from analysis_pipeline.utils import load_config
from analysis_pipeline.changepoint_analysis import run_changepoint_analysis
import os, time

config = load_config("config.yaml")
events_df = pd.read_csv("outputs/tables/events_processed.csv")
output_dir = config.get("output_dir", "./outputs")

print("=== Running Change-Point Detection / BOCPD (section 10) ===")
t0 = time.time()
cp_results = run_changepoint_analysis(events_df, config, output_dir)
elapsed = time.time() - t0
print(f"  BOCPD completed in {elapsed:.1f}s")

if "changepoints" in cp_results:
    cp_df = cp_results["changepoints"]
    cps_only = cp_df[cp_df["click_index"] >= 0]
    print(f"  Detected {len(cps_only)} change points across {cps_only['session_id'].nunique()} sessions")
    cp_df.to_csv(os.path.join(output_dir, "tables", "changepoints.csv"), index=False)
    print(f"  Saved changepoints.csv")

if "boundary_alignment" in cp_results:
    print(f"\n  Boundary alignment:")
    for boundary, stats in cp_results["boundary_alignment"].items():
        print(f"    {boundary}: {stats['n_sessions_with_nearby_cp']} sessions, "
              f"mean lag={stats['mean_lag_s']:.1f}s, median={stats['median_lag_s']:.1f}s")
