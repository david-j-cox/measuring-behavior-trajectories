#!/usr/bin/env python3
"""Run CCM causal coupling analysis standalone."""
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import pandas as pd
from analysis_pipeline.utils import load_config
from analysis_pipeline.dynamical_analysis import _run_ccm
import os, time

if __name__ == '__main__':
    config = load_config("config.yaml")
    events_df = pd.read_csv("outputs/tables/events_processed.csv")
    output_dir = config.get("output_dir", "./outputs")
    fmt = config.get("plot_format", "png")
    fig_dir = os.path.join(output_dir, "figures", "dynamical")

    print("=== Running CCM (section 8) ===")
    t0 = time.time()
    ccm_results = _run_ccm(events_df, config, fig_dir, fmt)
    elapsed = time.time() - t0
    print(f"  CCM completed in {elapsed:.1f}s")

    if len(ccm_results) > 0:
        ccm_results.to_csv(os.path.join(output_dir, "tables", "ccm_results.csv"), index=False)
        print(f"  Saved ccm_results.csv ({len(ccm_results)} rows, {ccm_results['session_id'].nunique()} sessions)")

        max_lib = ccm_results.groupby("session_id").apply(
            lambda g: g.loc[g["lib_size"].idxmax()]
        ).reset_index(drop=True)
        print(f"\n  CCM summary (at max library size):")
        print(f"    Mean rho (reward->choice): {max_lib['rho_reward_causes_choice'].mean():.3f}")
        print(f"    Mean rho (choice->reward): {max_lib['rho_choice_causes_reward'].mean():.3f}")
    else:
        print("  No CCM results produced.")
