#!/usr/bin/env python3
"""Regenerate empirical paper figures from cached analysis outputs.

Usage:
    python regen_figures.py              # all figures
    python regen_figures.py --figure 5   # just figure 5
    python regen_figures.py --figure 2 5 # figures 2 and 5
"""

import argparse
import os
import sys
import pandas as pd
import yaml

# Resolve paths relative to analysis/ (one level up from scripts_notebooks/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from pipeline.step07_generate_figures import generate_all_figures


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate empirical paper figures.")
    parser.add_argument("--figure", "-f", type=int, nargs="+", default=None,
                        help="Figure number(s) to regenerate (default: all)")
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, "config.yaml")) as f:
        config = yaml.safe_load(f)

    # Resolve config paths relative to analysis/ directory
    for key in ("output_dir", "tables_dir", "transformed_data_dir",
                "figures_dir", "local_data_dir"):
        if key in config and not os.path.isabs(config[key]):
            config[key] = os.path.join(ANALYSIS_DIR, config[key])

    tables_dir = config.get("tables_dir",
                            os.path.join(ANALYSIS_DIR, "data", "03_analytic_outputs"))
    transformed_dir = config.get("transformed_data_dir",
                                  os.path.join(ANALYSIS_DIR, "data", "02_transformed_data"))
    figures_dir = config.get("figures_dir",
                              os.path.join(ANALYSIS_DIR, "figures"))
    output_dir = config.get("output_dir", tables_dir)

    print("Loading cached data...")
    events_df = pd.read_csv(os.path.join(transformed_dir, "events_processed.csv"))
    metrics_df = pd.read_csv(os.path.join(transformed_dir, "session_metrics.csv"))
    print(f"  {events_df['session_id'].nunique()} sessions, "
          f"{len(events_df)} events")

    # Build analysis_results dict from cached tables
    analysis_results = {}

    # Model comparison
    mc_path = os.path.join(tables_dir, "model_comparison.csv")
    if os.path.exists(mc_path):
        analysis_results["model_comparison"] = pd.read_csv(mc_path)
        print(f"  Model comparison: {len(analysis_results['model_comparison'])} rows")

    # CCM
    ccm_path = os.path.join(tables_dir, "ccm_results.csv")
    if os.path.exists(ccm_path):
        analysis_results["ccm"] = pd.read_csv(ccm_path)
        print(f"  CCM: {len(analysis_results['ccm'])} rows")

    # RQA
    rqa_path = os.path.join(tables_dir, "rqa_metrics.csv")
    if os.path.exists(rqa_path):
        analysis_results["rqa"] = pd.read_csv(rqa_path)
        print(f"  RQA: {len(analysis_results['rqa'])} rows")

    # HMM state summary
    hmm_path = os.path.join(tables_dir, "hmm_state_summary.csv")
    if os.path.exists(hmm_path):
        analysis_results["hmm_state_summary"] = pd.read_csv(hmm_path)
        print(f"  HMM summary: {len(analysis_results['hmm_state_summary'])} rows")

    # HMM state sequences (per-click assignments)
    hmm_seq_path = os.path.join(tables_dir, "hmm_state_sequences.json")
    if os.path.exists(hmm_seq_path):
        from pipeline.models.hmm_models import load_state_sequences
        analysis_results["hmm_state_sequences"] = load_state_sequences(
            hmm_seq_path)
        print(f"  HMM sequences: {len(analysis_results['hmm_state_sequences'])} sessions")

    print("\nGenerating figures...")
    generate_all_figures(events_df, metrics_df, analysis_results, config,
                         output_dir, only=args.figure,
                         figures_dir=figures_dir)
    print("Done.")


if __name__ == "__main__":
    main()
