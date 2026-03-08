#!/usr/bin/env python3
"""CLI entry point for the behavioral trajectory analysis pipeline."""

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
from analysis_pipeline.plots import (
    plot_individual_session, plot_group_summary, plot_group_trajectories,
)
from analysis_pipeline.phase_analysis import run_phase_analysis
from analysis_pipeline.perturbation_analysis import run_perturbation_analysis
from analysis_pipeline.dynamical_analysis import run_dynamical_analysis
from analysis_pipeline.changepoint_analysis import run_changepoint_analysis
from analysis_pipeline.fractal_analysis import run_fractal_analysis
from analysis_pipeline.individual_differences import run_individual_differences
from analysis_pipeline.null_comparison import run_null_comparison
from analysis_pipeline.robustness_checks import run_robustness_checks
from analysis_pipeline.reports import generate_report


def main():
    parser = argparse.ArgumentParser(
        description="Run the behavioral trajectory analysis pipeline."
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)"
    )
    parser.add_argument(
        "--steps", nargs="*", default=None,
        help="Run specific steps: load, validate, transform, metrics, "
             "plots, phase, pulse, dynamical, changepoint, fractal, "
             "models, individual_differences, null_comparison, "
             "robustness, report. Default: run all steps."
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation."
    )
    parser.add_argument(
        "--null-comparison", action="store_true",
        help="Run null comparison analysis (generate null data and compare)."
    )
    parser.add_argument(
        "--robustness", action="store_true",
        help="Run sensitivity/robustness checks on key analysis parameters."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config.get("output_dir", "./outputs")
    ensure_dirs(output_dir)

    all_steps = args.steps is None
    steps = set(args.steps) if args.steps else set()

    t0 = time.time()
    analysis_results = {}

    # ---- Step 1: Load data ----
    print("\n=== Step 1: Loading data ===")
    events_df, meta_df = load_data(config)
    if len(events_df) == 0:
        print("ERROR: No events loaded. Check data source configuration.")
        sys.exit(1)

    # ---- Step 2: Validate ----
    if all_steps or "validate" in steps:
        print("\n=== Step 2: Validating data ===")
        events_df, validation_report = validate_events(events_df, config)
        validation_report.to_csv(
            os.path.join(output_dir, "reports", "validation_report.csv"), index=False
        )
    else:
        validation_report = pd.DataFrame()

    # ---- Step 3: Transform ----
    if all_steps or "transform" in steps:
        print("\n=== Step 3: Computing derived variables ===")
        events_df = compute_derived_variables(events_df, config)

    # ---- Step 4: Session metrics ----
    if all_steps or "metrics" in steps:
        print("\n=== Step 4: Computing session metrics ===")
        metrics_df = compute_session_metrics(events_df, config)
        metrics_df.to_csv(
            os.path.join(output_dir, "tables", "session_metrics.csv"), index=False
        )
        print(f"  {len(metrics_df)} sessions processed.")
    else:
        metrics_df = pd.DataFrame()

    # ---- Step 5: Plots ----
    if (all_steps or "plots" in steps) and not args.no_plots:
        print("\n=== Step 5: Generating plots ===")

        # Individual session plots
        for sid in events_df["session_id"].unique():
            sdf = events_df[events_df["session_id"] == sid]
            plot_individual_session(sdf, sid, config, output_dir)
        print(f"  {events_df['session_id'].nunique()} individual session plots.")

        # Group plots
        if len(metrics_df) > 0:
            plot_group_summary(metrics_df, config, output_dir)
            print("  Group summary plots done.")

        plot_group_trajectories(events_df, config, output_dir)
        print("  Group trajectory plots done.")

    # ---- Step 6: Phase analysis ----
    if all_steps or "phase" in steps:
        print("\n=== Step 6: Phase transition analysis ===")
        phase_results = run_phase_analysis(events_df, metrics_df, config, output_dir)
        analysis_results.update(phase_results)

        if "phase_descriptives" in phase_results:
            phase_results["phase_descriptives"].to_csv(
                os.path.join(output_dir, "tables", "phase_descriptives.csv"), index=False
            )

    # ---- Step 7: Pulse analysis ----
    if all_steps or "pulse" in steps:
        print("\n=== Step 7: Perturbation (pulse) analysis ===")
        pulse_results = run_perturbation_analysis(events_df, config, output_dir)
        analysis_results.update(pulse_results)

    # ---- Step 8: Dynamical analysis ----
    if all_steps or "dynamical" in steps:
        print("\n=== Step 8: Dynamical systems analysis ===")
        dyn_results = run_dynamical_analysis(events_df, config, output_dir)
        analysis_results.update(dyn_results)

    # ---- Step 9: Change-point detection ----
    if all_steps or "changepoint" in steps:
        print("\n=== Step 9: Change-point detection (BOCPD) ===")
        cp_results = run_changepoint_analysis(events_df, config, output_dir)
        analysis_results.update(cp_results)

    # ---- Step 10: Fractal analysis ----
    if all_steps or "fractal" in steps:
        print("\n=== Step 10: Fractal analysis (DFA, sample entropy) ===")
        fractal_results = run_fractal_analysis(events_df, config, output_dir)
        analysis_results.update(fractal_results)

    # ---- Step 11: Model fitting ----
    if (all_steps or "models" in steps) and config.get("run_models", True):
        print("\n=== Step 11: Fitting behavioral models ===")
        model_families = config.get("model_families", [])
        all_model_results = []

        if "baseline" in model_families:
            print("  Fitting baseline models...")
            from analysis_pipeline.models.baseline_models import fit_all_baselines
            baseline_results = fit_all_baselines(events_df)
            all_model_results.append(baseline_results)

        if "rl" in model_families:
            print("  Fitting RL models (including phase-aware and dynamic)...")
            from analysis_pipeline.models.rl_models import fit_rl_models
            rl_results = fit_rl_models(events_df, config)
            all_model_results.append(rl_results)

        if "matching" in model_families:
            print("  Fitting matching law models...")
            from analysis_pipeline.models.matching_models import fit_matching_models
            matching_results = fit_matching_models(events_df, metrics_df, config)
            all_model_results.append(matching_results)

        if "hmm" in model_families:
            print("  Fitting Hidden Markov Models...")
            from analysis_pipeline.models.hmm_models import (
                fit_hmm_models, summarize_hmm_states,
            )
            hmm_results, hmm_state_sequences = fit_hmm_models(
                events_df, config, output_dir
            )
            all_model_results.append(hmm_results)
            hmm_summary = summarize_hmm_states(hmm_state_sequences, output_dir)
            if len(hmm_summary) > 0:
                analysis_results["hmm_state_summary"] = hmm_summary

        if all_model_results:
            model_comparison = pd.concat(all_model_results, ignore_index=True)
            model_comparison.to_csv(
                os.path.join(output_dir, "tables", "model_comparison.csv"), index=False
            )
            analysis_results["model_comparison"] = model_comparison
            print(f"  {len(model_comparison)} model fits completed.")

    # ---- Step 12: Individual differences (PCA + GMM clustering) ----
    if all_steps or "individual_differences" in steps:
        print("\n=== Step 12: Individual differences (PCA + GMM clustering) ===")
        id_results = run_individual_differences(
            metrics_df, analysis_results, config, output_dir
        )
        analysis_results.update({"individual_differences": id_results})

    # ---- Step 13: Null comparison ----
    if args.null_comparison or "null_comparison" in steps:
        print("\n=== Step 13: Null comparison analysis ===")
        null_results = run_null_comparison(events_df, config, output_dir)
        analysis_results.update(null_results)

    # ---- Step 14: Robustness checks ----
    if args.robustness or "robustness" in steps:
        print("\n=== Step 14: Sensitivity/robustness checks ===")
        robustness_results = run_robustness_checks(events_df, config, output_dir)
        analysis_results.update({"robustness": robustness_results})

    # ---- Step 15: Save processed data ----
    print("\n=== Saving processed data ===")
    events_df.to_csv(
        os.path.join(output_dir, "tables", "events_processed.csv"), index=False
    )

    # ---- Step 16: Report ----
    if (all_steps or "report" in steps) and config.get("generate_report", True):
        print("\n=== Generating report ===")
        generate_report(metrics_df, validation_report, analysis_results,
                        config, output_dir)

    elapsed = time.time() - t0
    print(f"\n=== Pipeline complete in {elapsed:.1f}s ===")


if __name__ == "__main__":
    main()
