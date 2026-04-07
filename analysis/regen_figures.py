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

sys.path.insert(0, os.path.dirname(__file__))

from analysis_pipeline.empirical_paper.figure_generation import generate_all_figures


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate empirical paper figures.")
    parser.add_argument("--figure", "-f", type=int, nargs="+", default=None,
                        help="Figure number(s) to regenerate (default: all)")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    output_dir = config.get("output_dir", "./outputs")
    tables_dir = os.path.join(output_dir, "tables")

    print("Loading cached data...")
    events_df = pd.read_csv(os.path.join(tables_dir, "events_processed.csv"))
    metrics_df = pd.read_csv(os.path.join(tables_dir, "session_metrics.csv"))
    print(f"  {events_df['session_id'].nunique()} sessions, "
          f"{len(events_df)} events")

    # Build analysis_results dict from cached tables
    analysis_results = {}

    # Model comparison
    mc_path = os.path.join(output_dir, "empirical_paper", "tables",
                           "model_comparison.csv")
    if not os.path.exists(mc_path):
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
        from analysis_pipeline.models.hmm_models import load_state_sequences
        analysis_results["hmm_state_sequences"] = load_state_sequences(
            hmm_seq_path)
        print(f"  HMM sequences: {len(analysis_results['hmm_state_sequences'])} sessions")

    print("\nGenerating figures...")
    generate_all_figures(events_df, metrics_df, analysis_results, config,
                         output_dir, only=args.figure)
    print("Done.")


if __name__ == "__main__":
    main()
