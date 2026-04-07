#!/usr/bin/env python3
"""Regenerate manuscript outputs (tables, figures, captions, etc.) from cached CSVs.

Skips all model fitting and dynamical analysis — just reloads previous results
and re-runs the manuscript layer.

Usage:
    python regen_manuscript.py                  # everything
    python regen_manuscript.py --tables-only    # just tables
    python regen_manuscript.py --figures-only   # just figures
"""

import argparse
import os
import sys
import time

import pandas as pd

from analysis_pipeline.utils import load_config, ensure_dirs


def _load_cached(tables_dir, filename):
    path = os.path.join(tables_dir, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Regenerate manuscript outputs from cached results.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config.get("output_dir", "./outputs")
    tables_dir = os.path.join(output_dir, "tables")

    for subdir in ["figures", "tables", "captions", "results_summary", "supplementary"]:
        os.makedirs(os.path.join(output_dir, "manuscript", subdir), exist_ok=True)

    t0 = time.time()

    # ── Load cached data ───────────────────────────────────────────────────
    print("Loading cached results...")
    events_df = _load_cached(tables_dir, "events_processed.csv")
    metrics_df = _load_cached(tables_dir, "session_metrics.csv")

    if len(events_df) == 0:
        print("ERROR: No events_processed.csv found. Run the full pipeline first.")
        sys.exit(1)

    analysis_results = {}
    analysis_results["model_comparison"] = _load_cached(tables_dir, "model_comparison.csv")
    analysis_results["rqa"] = _load_cached(tables_dir, "rqa_metrics.csv")
    analysis_results["edm_simplex"] = _load_cached(tables_dir, "edm_simplex.csv")
    analysis_results["ccm"] = _load_cached(tables_dir, "ccm_results.csv")
    analysis_results["smap"] = _load_cached(tables_dir, "smap_results.csv")
    analysis_results["dfa"] = _load_cached(tables_dir, "dfa_results.csv")
    analysis_results["sample_entropy"] = _load_cached(tables_dir, "sample_entropy.csv")
    analysis_results["changepoints"] = _load_cached(tables_dir, "changepoints.csv")
    analysis_results["phase_descriptives"] = _load_cached(tables_dir, "phase_descriptives.csv")
    analysis_results["null_comparison"] = _load_cached(tables_dir, "null_comparison.csv")
    analysis_results["robustness"] = _load_cached(tables_dir, "robustness_summary.csv")

    # HMM state summary
    hmm_summary = _load_cached(tables_dir, "hmm_state_summary.csv")
    if len(hmm_summary) > 0:
        analysis_results["hmm_state_summary"] = hmm_summary

    # Individual differences
    id_features = _load_cached(tables_dir, "individual_differences_features.csv")
    clusters = _load_cached(tables_dir, "cluster_assignments.csv")
    profiles = _load_cached(tables_dir, "cluster_profiles.csv")
    diagnostics = _load_cached(tables_dir, "cluster_diagnostics.csv")
    analysis_results["individual_differences"] = {
        "features": id_features,
        "clusters": clusters,
        "profiles": profiles,
        "diagnostics": diagnostics,
    }

    loaded = {k: len(v) if hasattr(v, '__len__') else 'dict' for k, v in analysis_results.items()}
    print(f"  Loaded: {loaded}")

    # ── Generate manuscript outputs ────────────────────────────────────────
    if not args.tables_only:
        print("\n--- Generating manuscript figures ---")
        from analysis_pipeline.manuscript_tutorial_archive.figure_generation import generate_all_figures
        generate_all_figures(events_df, metrics_df, analysis_results, config, output_dir)

    if not args.figures_only:
        print("\n--- Generating manuscript tables ---")
        from analysis_pipeline.manuscript_tutorial_archive.table_generation import generate_all_tables
        generate_all_tables(events_df, metrics_df, analysis_results, config, output_dir)

    if not args.figures_only and not args.tables_only:
        print("\n--- Generating captions ---")
        from analysis_pipeline.manuscript_tutorial_archive.caption_generator import generate_all_captions
        generate_all_captions(metrics_df, analysis_results, config, output_dir)

        print("\n--- Generating results summary ---")
        from analysis_pipeline.manuscript_tutorial_archive.results_summary import generate_results_summary
        generate_results_summary(events_df, metrics_df, analysis_results, config, output_dir)

        print("\n--- Generating manuscript outline ---")
        from analysis_pipeline.manuscript_tutorial_archive.manuscript_outline import generate_manuscript_outline
        generate_manuscript_outline(metrics_df, analysis_results, config, output_dir)

        print("\n--- Assembling export package ---")
        from analysis_pipeline.manuscript_tutorial_archive.export_package import export_manuscript_package
        export_manuscript_package(events_df, metrics_df, analysis_results, config, output_dir)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"MANUSCRIPT REGEN COMPLETE  ({elapsed:.1f}s)")
    print(f"Outputs: {os.path.join(output_dir, 'manuscript')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
