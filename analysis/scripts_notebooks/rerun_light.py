#!/usr/bin/env python3
"""
Lightweight pipeline rerun: re-processes data (steps 01-04), loads existing
heavy outputs (dynamical analysis, HMM, model comparison) from disk, reruns
lighter analyses (phase, fractal, individual differences), and regenerates
all figures and tables.

Skips:
  - step06_dynamical_analysis (EDM, CCM, S-Map — too memory-intensive for laptop)
  - Full model re-fitting (loads existing model_comparison.csv)

Usage:
    python rerun_light.py
"""

import gc
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.utils import load_config, ensure_dirs
from pipeline.step01_load_data import load_data
from pipeline.step02_validate_data import validate_events
from pipeline.step03_feature_engineering import compute_derived_variables
from pipeline.step04_session_metrics import compute_session_metrics


def main():
    config = load_config(os.path.join(SCRIPT_DIR, "config.yaml"))

    # Force local data source (no Firebase dependency needed)
    config["data_source"] = "local"

    # Resolve config paths relative to analysis/ directory
    for key in ("output_dir", "tables_dir", "transformed_data_dir",
                "figures_dir", "local_data_dir"):
        if key in config and not os.path.isabs(config[key]):
            config[key] = os.path.join(ANALYSIS_DIR, config[key])

    output_dir = config.get("output_dir",
                            os.path.join(ANALYSIS_DIR, "data", "03_analytic_outputs"))
    tables_dir = config.get("tables_dir", output_dir)
    transformed_dir = config.get("transformed_data_dir",
                                  os.path.join(ANALYSIS_DIR, "data", "02_transformed_data"))
    figures_dir = config.get("figures_dir",
                              os.path.join(ANALYSIS_DIR, "figures"))

    ensure_dirs(output_dir)
    for d in [tables_dir, transformed_dir, figures_dir]:
        os.makedirs(d, exist_ok=True)
    for subdir in ["tables", "supplementary"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    t0 = time.time()
    print("\n" + "=" * 60)
    print("LIGHTWEIGHT PIPELINE RERUN")
    print("Reprocessing data + regenerating figures/tables")
    print("(skipping dynamical analysis & model re-fitting)")
    print("=" * 60)

    # ── Step 1-4: Data processing ────────────────────────────────
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

    n_sessions = events_df["session_id"].nunique()
    n_events = len(events_df)
    print(f"  {n_sessions} sessions, {n_events} events.")

    analysis_results = {}

    # ── Load existing model comparison from disk ─────────────────
    print("\n" + "-" * 40)
    print("LOADING EXISTING RESULTS FROM DISK")
    print("-" * 40)

    model_comp_path = os.path.join(tables_dir, "model_comparison.csv")
    if os.path.exists(model_comp_path):
        model_comparison = pd.read_csv(model_comp_path)
        analysis_results["model_comparison"] = model_comparison
        print(f"  Loaded model_comparison.csv ({len(model_comparison)} rows)")
    else:
        print("  WARNING: model_comparison.csv not found!")

    # Load HMM state sequences
    hmm_cache_path = os.path.join(output_dir, "tables", "hmm_state_sequences.json")
    # Also check the root output dir (older location)
    hmm_alt_path = os.path.join(output_dir, "hmm_state_sequences.json")
    for hpath in [hmm_cache_path, hmm_alt_path]:
        if os.path.exists(hpath):
            from pipeline.models.hmm_models import load_state_sequences, summarize_hmm_states
            hmm_state_sequences = load_state_sequences(hpath)
            analysis_results["hmm_state_sequences"] = hmm_state_sequences
            print(f"  Loaded HMM state sequences ({len(hmm_state_sequences)} sessions)")
            hmm_summary = summarize_hmm_states(hmm_state_sequences, output_dir)
            if len(hmm_summary) > 0:
                analysis_results["hmm_state_summary"] = hmm_summary
            break
    else:
        print("  WARNING: hmm_state_sequences.json not found!")

    # Load existing RQA metrics
    rqa_path = os.path.join(output_dir, "rqa_metrics.csv")
    if os.path.exists(rqa_path):
        rqa_df = pd.read_csv(rqa_path)
        analysis_results["rqa"] = rqa_df
        print(f"  Loaded rqa_metrics.csv ({len(rqa_df)} rows)")

    # Load existing EDM results
    for fname, key in [("edm_simplex.csv", "edm_simplex"),
                       ("ccm_results.csv", "ccm"),
                       ("smap_results.csv", "smap"),
                       ("dfa_results.csv", "dfa"),
                       ("sample_entropy.csv", "sample_entropy")]:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            analysis_results[key] = pd.read_csv(fpath)
            print(f"  Loaded {fname}")

    # ── Run lighter step06 modules ───────────────────────────────
    print("\n" + "-" * 40)
    print("RUNNING LIGHT ANALYSES")
    print("-" * 40)

    # Phase analysis
    print("\n--- Phase analysis ---")
    from pipeline.step06_phase_analysis import run_phase_analysis
    phase_results = run_phase_analysis(events_df, metrics_df, config, output_dir)
    analysis_results.update(phase_results)

    # Save updated phase descriptives
    if "phase_descriptives" in phase_results:
        phase_results["phase_descriptives"].to_csv(
            os.path.join(tables_dir, "phase_descriptives.csv"), index=False)
        print("  Saved phase_descriptives.csv")

    # Fractal analysis
    print("\n--- Fractal analysis ---")
    from pipeline.step06_fractal_analysis import run_fractal_analysis
    fractal_results = run_fractal_analysis(events_df, config, output_dir)
    analysis_results.update(fractal_results)
    plt.close("all"); gc.collect()

    # Individual differences
    print("\n--- Individual differences ---")
    from pipeline.step06_individual_differences import run_individual_differences
    id_results = run_individual_differences(
        metrics_df, analysis_results, config, output_dir
    )
    analysis_results["individual_differences"] = id_results

    # ── Regenerate table3 (model comparison summary) ─────────────
    print("\n" + "-" * 40)
    print("REGENERATING SUMMARY TABLES")
    print("-" * 40)

    if "model_comparison" in analysis_results:
        _regenerate_table3(analysis_results, tables_dir, output_dir)

    # ── Generate all figures ─────────────────────────────────────
    print("\n" + "-" * 40)
    print("FIGURE GENERATION")
    print("-" * 40)

    from pipeline.step07_generate_figures import generate_all_figures
    generate_all_figures(events_df, metrics_df, analysis_results, config,
                         output_dir, figures_dir=figures_dir)
    plt.close("all"); gc.collect()

    # ── Done ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"LIGHTWEIGHT RERUN COMPLETE  ({elapsed:.1f}s)")
    print(f"Tables:  {tables_dir}")
    print(f"Figures: {figures_dir}")
    print(f"{'=' * 60}")


def _regenerate_table3(analysis_results, tables_dir, output_dir):
    """Rebuild the formatted table3_model_comparison_full.csv."""
    mc = analysis_results["model_comparison"]

    from pipeline.step07_generate_figures import _MODEL_FAMILIES, _MODEL_LABELS

    # Load classification metrics (accuracy, F1, MCC) — separate file
    class_path = os.path.join(tables_dir, "classification_metrics.csv")
    class_df = pd.read_csv(class_path) if os.path.exists(class_path) else pd.DataFrame()
    # Keep only full-session metrics
    if "phase" in class_df.columns:
        class_df = class_df[class_df["phase"] == "Full"]

    # Also load foraging classification metrics
    forage_class_path = os.path.join(tables_dir, "foraging_classification_metrics.csv")
    if os.path.exists(forage_class_path):
        forage_class = pd.read_csv(forage_class_path)
        if "phase" in forage_class.columns:
            forage_class = forage_class[forage_class["phase"] == "Full"]
        class_df = pd.concat([class_df, forage_class], ignore_index=True)

    # Load EDM forecast metrics (simplex)
    edm_path = os.path.join(tables_dir, "edm_forecast_metrics.csv")
    if os.path.exists(edm_path):
        edm_df = pd.read_csv(edm_path)
        # Map approach names to model names
        approach_map = {
            "phase1_train": "simplex_phase1_rest",
            "expanding_window": "simplex_expanding",
        }
        for approach, model_name in approach_map.items():
            adf = edm_df[edm_df["approach"] == approach]
            if len(adf) > 0:
                adf = adf.copy()
                adf["model"] = model_name
                class_df = pd.concat([class_df, adf], ignore_index=True)

    # Load S-Map forecast metrics
    smap_path = os.path.join(tables_dir, "smap_forecast_metrics.csv")
    if os.path.exists(smap_path):
        smap_df = pd.read_csv(smap_path)
        approach_map = {
            "smap_linear": "smap_linear",
            "smap_best": "smap_best_theta",
        }
        for approach, model_name in approach_map.items():
            adf = smap_df[smap_df["approach"] == approach]
            if len(adf) > 0:
                adf = adf.copy()
                adf["model"] = model_name
                class_df = pd.concat([class_df, adf], ignore_index=True)

    # Build per-model classification lookup: model -> {accuracy, f1, mcc}
    class_lookup = {}
    if len(class_df) > 0:
        for model_name, mdf in class_df.groupby("model"):
            class_lookup[model_name] = {
                "accuracy": mdf["accuracy"].dropna(),
                "f1": mdf["f1"].dropna(),
                "mcc": mdf["mcc"].dropna(),
            }

    # Aggregate per model
    rows = []
    all_models = set(mc["model"].dropna().unique())
    if len(class_df) > 0:
        all_models |= set(class_df["model"].dropna().unique())

    for model_name in sorted(all_models):
        family = _MODEL_FAMILIES.get(model_name, "EDM")
        display = _MODEL_LABELS.get(model_name, model_name.replace("_", " ").title())

        row = {"Model": display, "Family": family}

        # Parametric info from model_comparison
        mdf = mc[mc["model"] == model_name] if model_name in mc["model"].values else pd.DataFrame()

        if len(mdf) > 0 and "n_params" in mdf.columns:
            k = mdf["n_params"].iloc[0]
            row["k"] = int(k) if pd.notna(k) else "---"
        else:
            row["k"] = "---"

        for col, label in [("aic", "AIC"), ("nll", "NLL")]:
            if len(mdf) > 0 and col in mdf.columns:
                vals = mdf[col].dropna()
                if len(vals) > 0:
                    row[label] = f"{vals.median():.1f} ({vals.std():.1f})"
                else:
                    row[label] = "--- (---)"
            else:
                row[label] = "--- (---)"

        # Classification metrics (median + std, matching manuscript convention)
        cm = class_lookup.get(model_name, {})
        for metric, col_name in [("accuracy", "Accuracy"), ("f1", "F1"), ("mcc", "MCC")]:
            vals = cm.get(metric, pd.Series(dtype=float))
            if len(vals) > 0:
                row[col_name] = f"{vals.median():.3f} ({vals.std():.2f})"
            else:
                row[col_name] = "--- (---)"

        rows.append(row)

    if rows:
        table3 = pd.DataFrame(rows)
        # Sort by accuracy descending
        def _extract_mean(s):
            try:
                return float(str(s).split("(")[0].strip())
            except (ValueError, IndexError):
                return -999
        if "Accuracy" in table3.columns:
            table3["_sort"] = table3["Accuracy"].apply(_extract_mean)
            table3 = table3.sort_values("_sort", ascending=False).drop(columns="_sort")

        table3.to_csv(os.path.join(tables_dir, "table3_model_comparison_full.csv"),
                      index=False)
        print(f"  Saved table3_model_comparison_full.csv ({len(table3)} models)")


if __name__ == "__main__":
    main()
