#!/usr/bin/env python3
"""
CLI entry point for the manuscript-oriented analysis layer.

Runs the base pipeline (if needed), then generates all manuscript outputs:
figures, tables, captions, results summary, outline, and export package.

Usage:
    python run_manuscript_analysis.py --config config.yaml
    python run_manuscript_analysis.py --figures-only
    python run_manuscript_analysis.py --tables-only
    python run_manuscript_analysis.py --no-models
"""

import argparse
import os
import sys
import time

import pandas as pd

from analysis_pipeline.utils import load_config, ensure_dirs
from analysis_pipeline.io import load_data
from analysis_pipeline.validate import validate_events
from analysis_pipeline.transform import compute_derived_variables
from analysis_pipeline.metrics import compute_session_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Generate manuscript-ready outputs from the behavioral analysis pipeline."
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config (default: config.yaml)")
    parser.add_argument("--input", default=None,
                        help="Override local data directory")
    parser.add_argument("--output", default=None,
                        help="Override output directory")
    parser.add_argument("--figures-only", action="store_true",
                        help="Generate figures only, skip tables/summaries")
    parser.add_argument("--tables-only", action="store_true",
                        help="Generate tables only, skip figures/summaries")
    parser.add_argument("--no-models", action="store_true",
                        help="Skip model fitting")
    parser.add_argument("--participant-id", default=None,
                        help="Restrict to a single participant")
    parser.add_argument("--session-id", default=None,
                        help="Restrict to a single session")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.input:
        config["data_source"] = "local"
        config["local_data_dir"] = args.input
    if args.output:
        config["output_dir"] = args.output

    output_dir = config.get("output_dir", "./outputs")
    ensure_dirs(output_dir)

    # Create manuscript output directories
    for subdir in ["figures", "tables", "captions", "results_summary", "supplementary"]:
        os.makedirs(os.path.join(output_dir, "manuscript", subdir), exist_ok=True)

    t0 = time.time()

    # ════════════════════════════════════════════════════════════════════════
    # Step 1: Run base pipeline
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("MANUSCRIPT ANALYSIS PIPELINE")
    print("=" * 60)

    print("\n--- Loading data ---")
    events_df, meta_df = load_data(config)
    if len(events_df) == 0:
        print("ERROR: No events loaded. Check data source configuration.")
        sys.exit(1)

    # Filter by participant/session if requested
    if args.participant_id:
        events_df = events_df[events_df["participant_id"] == args.participant_id]
        print(f"  Filtered to participant: {args.participant_id}")
    if args.session_id:
        events_df = events_df[events_df["session_id"] == args.session_id]
        print(f"  Filtered to session: {args.session_id}")

    print("\n--- Validating ---")
    events_df, validation_report = validate_events(events_df, config)

    print("\n--- Computing derived variables ---")
    events_df = compute_derived_variables(events_df, config)

    print("\n--- Computing session metrics ---")
    metrics_df = compute_session_metrics(events_df, config)
    print(f"  {len(metrics_df)} sessions.")

    # ════════════════════════════════════════════════════════════════════════
    # Step 2: Model fitting (unless --no-models)
    # ════════════════════════════════════════════════════════════════════════
    analysis_results = {}

    if not args.no_models and not args.tables_only and config.get("run_models", True):
        print("\n--- Fitting models ---")
        model_dfs = []

        from analysis_pipeline.models.baseline_models import fit_all_baselines
        print("  Baseline models...")
        model_dfs.append(fit_all_baselines(events_df))

        from analysis_pipeline.models.rl_models import fit_rl_models
        print("  RL models...")
        model_dfs.append(fit_rl_models(events_df, config))

        from analysis_pipeline.models.matching_models import fit_matching_models
        print("  Matching models...")
        model_dfs.append(fit_matching_models(events_df, metrics_df, config))

        model_comparison = pd.concat(model_dfs, ignore_index=True)
        model_comparison.to_csv(
            os.path.join(output_dir, "tables", "model_comparison.csv"), index=False
        )
        analysis_results["model_comparison"] = model_comparison
        print(f"  {len(model_comparison)} model fits.")

    # Phase analysis results
    from analysis_pipeline.phase_analysis import run_phase_analysis
    print("\n--- Phase analysis ---")
    phase_results = run_phase_analysis(events_df, metrics_df, config, output_dir)
    analysis_results.update(phase_results)

    # ════════════════════════════════════════════════════════════════════════
    # Step 3: Manuscript outputs
    # ════════════════════════════════════════════════════════════════════════

    if not args.tables_only:
        print("\n--- Generating manuscript figures ---")
        from analysis_pipeline.manuscript.figure_generation import generate_all_figures
        generate_all_figures(events_df, metrics_df, analysis_results, config, output_dir)

    if not args.figures_only:
        print("\n--- Generating manuscript tables ---")
        from analysis_pipeline.manuscript.table_generation import generate_all_tables
        generate_all_tables(events_df, metrics_df, analysis_results, config, output_dir)

    if not args.figures_only and not args.tables_only:
        print("\n--- Generating captions ---")
        from analysis_pipeline.manuscript.caption_generator import generate_all_captions
        generate_all_captions(metrics_df, analysis_results, config, output_dir)

        print("\n--- Generating results summary ---")
        from analysis_pipeline.manuscript.results_summary import generate_results_summary
        generate_results_summary(events_df, metrics_df, analysis_results, config, output_dir)

        print("\n--- Generating manuscript outline ---")
        from analysis_pipeline.manuscript.manuscript_outline import generate_manuscript_outline
        generate_manuscript_outline(metrics_df, analysis_results, config, output_dir)

        print("\n--- Assembling export package ---")
        from analysis_pipeline.manuscript.export_package import export_manuscript_package
        export_manuscript_package(events_df, metrics_df, analysis_results, config, output_dir)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"MANUSCRIPT PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"Outputs: {os.path.join(output_dir, 'manuscript')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
