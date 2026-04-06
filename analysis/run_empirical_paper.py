#!/usr/bin/env python3
"""
Entry point for the empirical paper analysis:
"Historical models of behavior dynamics vs. modern approaches"

Runs:
  1. Data loading and preprocessing
  2. Historical dynamics models (melioration, kinetic, momentum, hill-climbing)
  3. Modern models (baselines, RL variants, HMM)
  4. Matching law (benchmark)
  5. Trajectory-level evaluation
  6. NDS diagnostics (RQA, DFA, EDM, changepoint, PCA+clustering)

Usage:
    python run_empirical_paper.py                    # full pipeline
    python run_empirical_paper.py --models-only      # just model fitting
    python run_empirical_paper.py --skip-models      # skip models, run NDS
    python run_empirical_paper.py --n-sessions 5     # subset for testing
"""

import argparse
import gc
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_pipeline.utils import load_config, ensure_dirs
from analysis_pipeline.io import load_data
from analysis_pipeline.validate import validate_events
from analysis_pipeline.transform import compute_derived_variables
from analysis_pipeline.metrics import compute_session_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run empirical paper analysis pipeline."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--models-only", action="store_true",
                        help="Run only model fitting, skip NDS analyses")
    parser.add_argument("--skip-models", action="store_true",
                        help="Skip model fitting, run NDS analyses only")
    parser.add_argument("--n-sessions", type=int, default=None,
                        help="Limit to N sessions (for testing)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.input:
        config["data_source"] = "local"
        config["local_data_dir"] = args.input
    if args.output:
        config["output_dir"] = args.output

    output_dir = config.get("output_dir", "./outputs")
    ensure_dirs(output_dir)
    paper_dir = os.path.join(output_dir, "empirical_paper")
    for subdir in ["tables", "figures", "supplementary"]:
        os.makedirs(os.path.join(paper_dir, subdir), exist_ok=True)

    t0 = time.time()
    print("\n" + "=" * 60)
    print("EMPIRICAL PAPER PIPELINE")
    print("Historical vs. Modern Models of Behavior Dynamics")
    print("=" * 60)

    # ── Step 1: Data ─────────────────────────────────────────────
    print("\n--- Loading data ---")
    events_df, meta_df = load_data(config)
    if len(events_df) == 0:
        print("ERROR: No events loaded.")
        sys.exit(1)

    print("\n--- Validating ---")
    events_df, validation_report = validate_events(events_df, config)

    print("\n--- Computing derived variables ---")
    events_df = compute_derived_variables(events_df, config)

    print("\n--- Computing session metrics ---")
    metrics_df = compute_session_metrics(events_df, config)

    if args.n_sessions:
        sessions = events_df["session_id"].unique()[:args.n_sessions]
        events_df = events_df[events_df["session_id"].isin(sessions)]
        metrics_df = metrics_df[metrics_df["session_id"].isin(sessions)]
        print(f"  Limited to {args.n_sessions} sessions for testing.")

    n_sessions = events_df["session_id"].nunique()
    n_events = len(events_df)
    print(f"  {n_sessions} sessions, {n_events} events.")

    analysis_results = {}

    # ── Step 2: Model fitting ────────────────────────────────────
    if not args.skip_models:
        print("\n" + "-" * 40)
        print("MODEL FITTING")
        print("-" * 40)

        model_dfs = []

        # Historical dynamics models
        print("\n--- Historical dynamics models ---")
        from analysis_pipeline.models.historical_dynamics import fit_historical_models
        print("  Melioration, Kinetic, Momentum, Hill-climbing...")
        t1 = time.time()
        hist_results = fit_historical_models(events_df, config)
        print(f"  {len(hist_results)} fits in {time.time() - t1:.1f}s")
        model_dfs.append(hist_results)

        # Matching law (benchmark)
        print("\n--- Matching law (benchmark) ---")
        from analysis_pipeline.models.matching_models import fit_matching_models
        matching_results = fit_matching_models(events_df, metrics_df, config)
        print(f"  {len(matching_results)} fits")
        model_dfs.append(matching_results)

        # Baseline models
        print("\n--- Baseline models ---")
        from analysis_pipeline.models.baseline_models import fit_all_baselines
        baseline_results = fit_all_baselines(events_df)
        # Drop logistic (not a process model)
        baseline_results = baseline_results[
            baseline_results["model"] != "logistic"
        ]
        print(f"  {len(baseline_results)} fits")
        model_dfs.append(baseline_results)

        # RL models
        print("\n--- RL models ---")
        from analysis_pipeline.models.rl_models import fit_rl_models
        rl_results = fit_rl_models(events_df, config)
        # Keep only: q_learning, q_dual_alpha, q_forgetting
        rl_results = rl_results[rl_results["model"].isin([
            "q_learning", "q_dual_alpha", "q_forgetting"
        ])]
        print(f"  {len(rl_results)} fits")
        model_dfs.append(rl_results)

        # HMM
        print("\n--- HMM ---")
        try:
            from analysis_pipeline.models.hmm_models import (
                fit_hmm_models, summarize_hmm_states,
            )
            hmm_results, hmm_state_sequences = fit_hmm_models(
                events_df, config, output_dir
            )
            if hmm_state_sequences:
                analysis_results["hmm_state_sequences"] = hmm_state_sequences
            hmm_summary = summarize_hmm_states(hmm_state_sequences, output_dir)
            if len(hmm_summary) > 0:
                analysis_results["hmm_state_summary"] = hmm_summary
            if len(hmm_results) > 0:
                analysis_results["hmm"] = hmm_results
                model_dfs.append(hmm_results)
            print(f"  {len(hmm_results)} fits")
        except Exception as e:
            print(f"  HMM fitting failed: {e}")

        # Combine all model results
        model_comparison = pd.concat(model_dfs, ignore_index=True)
        model_comparison.to_csv(
            os.path.join(paper_dir, "tables", "model_comparison.csv"),
            index=False
        )
        analysis_results["model_comparison"] = model_comparison

        # Print summary
        print(f"\n  Total model fits: {len(model_comparison)}")
        print("\n  Mean BIC by model (lower = better):")
        bic_summary = (model_comparison.dropna(subset=["bic"])
                       .groupby("model")["bic"].mean()
                       .sort_values())
        for model, bic in bic_summary.items():
            print(f"    {model:30s} {bic:10.1f}")

        plt.close("all")
        gc.collect()

    # ── Step 3: NDS diagnostics ──────────────────────────────────
    if not args.models_only:
        print("\n" + "-" * 40)
        print("NDS DIAGNOSTICS")
        print("-" * 40)

        # Phase analysis
        print("\n--- Phase analysis ---")
        from analysis_pipeline.phase_analysis import run_phase_analysis
        phase_results = run_phase_analysis(events_df, metrics_df, config, output_dir)
        analysis_results.update(phase_results)

        # Dynamical analysis (RQA, state space, EDM)
        print("\n--- Dynamical analysis ---")
        from analysis_pipeline.dynamical_analysis import run_dynamical_analysis
        dyn_results = run_dynamical_analysis(events_df, config, output_dir)
        analysis_results.update(dyn_results)
        plt.close("all")
        gc.collect()

        # Fractal analysis (DFA, sample entropy)
        print("\n--- Fractal analysis ---")
        from analysis_pipeline.fractal_analysis import run_fractal_analysis
        fractal_results = run_fractal_analysis(events_df, config, output_dir)
        analysis_results.update(fractal_results)
        plt.close("all")
        gc.collect()

        # Individual differences (PCA + GMM clustering)
        print("\n--- Individual differences ---")
        from analysis_pipeline.individual_differences import run_individual_differences
        id_results = run_individual_differences(
            metrics_df, analysis_results, config, output_dir
        )
        analysis_results["individual_differences"] = id_results

    # ── Step 4: Generate figures ────────────────────────────────
    print("\n" + "-" * 40)
    print("FIGURE GENERATION")
    print("-" * 40)
    from analysis_pipeline.empirical_paper.figure_generation import (
        generate_all_figures,
    )
    generate_all_figures(events_df, metrics_df, analysis_results, config,
                         output_dir)
    plt.close("all")
    gc.collect()

    # ── Done ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"EMPIRICAL PAPER PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"Outputs: {paper_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
