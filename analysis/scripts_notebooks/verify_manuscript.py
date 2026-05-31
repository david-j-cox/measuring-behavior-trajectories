#!/usr/bin/env python3
"""
Comprehensive verification of every numerical claim in the manuscript
against current pipeline outputs. Reports pass/fail with actual vs claimed
values.

Usage:
    python verify_manuscript.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

OUT = os.path.join(ANALYSIS_DIR, "data", "03_analytic_outputs")


def _fmt(claimed, actual, tol=0.02, is_int=False):
    """Return a check-mark or X based on tolerance."""
    if claimed is None or actual is None:
        return "  ", "skip"
    try:
        if is_int:
            ok = int(claimed) == int(actual)
        else:
            ok = abs(float(claimed) - float(actual)) <= tol
    except (ValueError, TypeError):
        return "? ", "unknown"
    return ("OK" if ok else "XX"), ("pass" if ok else "FAIL")


def check(label, claimed, actual, tol=0.02, is_int=False, unit=""):
    """Print a single check row."""
    mark, _ = _fmt(claimed, actual, tol, is_int)
    claim_s = f"{claimed}" if claimed is not None else "---"
    act_s = (f"{actual:.3f}" if isinstance(actual, float) else f"{actual}")
    print(f"  [{mark}] {label:<55s} claim={claim_s:>10s}  actual={act_s}{unit}")


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_events_with_features():
    """Load events through feature engineering (reuses pipeline code)."""
    from pipeline.utils import load_config
    from pipeline.step01_load_data import load_data
    from pipeline.step02_validate_data import validate_events
    from pipeline.step03_feature_engineering import compute_derived_variables
    from pipeline.step04_session_metrics import compute_session_metrics

    config = load_config(os.path.join(SCRIPT_DIR, "config.yaml"))
    config["data_source"] = "local"
    for key in ("output_dir", "tables_dir", "transformed_data_dir",
                "figures_dir", "local_data_dir"):
        if key in config and not os.path.isabs(config[key]):
            config[key] = os.path.join(ANALYSIS_DIR, config[key])

    events_df, _ = load_data(config)
    events_df, _ = validate_events(events_df, config)
    events_df = compute_derived_variables(events_df, config)
    metrics_df = compute_session_metrics(events_df, config)
    return events_df, metrics_df, config


def main():
    events_df, metrics_df, config = load_events_with_features()
    mc = pd.read_csv(os.path.join(OUT, "model_comparison.csv"))
    cm = pd.read_csv(os.path.join(OUT, "classification_metrics.csv"))
    cm_full = cm[cm["phase"] == "Full"]
    phase_desc = pd.read_csv(os.path.join(OUT, "phase_descriptives.csv"))

    # ════════════════════════════════════════════════════════════
    section("PARTICIPANT & SESSION TOTALS")
    # ════════════════════════════════════════════════════════════
    n_sessions = metrics_df["session_id"].nunique()
    check("Sample size (N sessions)", 60, n_sessions, is_int=True)

    mean_clicks = metrics_df["total_clicks"].mean()
    sd_clicks = metrics_df["total_clicks"].std()
    check("Mean responses per session", 924, round(mean_clicks), tol=1, is_int=True)
    check("SD responses per session", 200, round(sd_clicks), tol=5, is_int=True)

    mean_rr = metrics_df["overall_reward_rate"].mean()
    sd_rr = metrics_df["overall_reward_rate"].std()
    check("Mean overall reward rate", 0.54, mean_rr)
    check("SD overall reward rate", 0.17, sd_rr)

    mean_sr = metrics_df["overall_switch_rate"].mean()
    sd_sr = metrics_df["overall_switch_rate"].std()
    check("Mean overall switch rate", 0.24, mean_sr)
    check("SD overall switch rate", 0.15, sd_sr)

    # ════════════════════════════════════════════════════════════
    section("TABLE 2 — PHASE DESCRIPTIVES (session-weighted)")
    # ════════════════════════════════════════════════════════════
    phase_claims = {
        1: {"choice": (0.51, 0.06), "rr": 0.71, "sr": 0.28, "ici": 0.42,
            "clicks": 13436},
        2: {"choice": (0.65, 0.08), "rr": 0.62, "sr": 0.21, "ici": 0.41,
            "clicks": 14111},
        3: {"choice": (0.35, 0.08), "rr": 0.60, "sr": 0.19, "ici": 0.42,
            "clicks": 14402},
        4: {"choice": (0.49, 0.04), "rr": 0.23, "sr": 0.30, "ici": 0.45,
            "clicks": 13505},
    }
    for pid, c in phase_claims.items():
        row = phase_desc[phase_desc["phase_id"] == pid].iloc[0]
        print(f"\n  Phase {pid} ({row['label']}):")
        check(f"  total clicks", c["clicks"], row["n_clicks"], is_int=True)
        check(f"  choice prop A mean",
              c["choice"][0], row["mean_choice_prop_a"])
        check(f"  choice prop A SD",
              c["choice"][1], row["sd_choice_prop_a"])
        # Phase 3 manuscript actually says "toward A = 0.35" so same interpretation
        check(f"  reward rate", c["rr"], row["mean_reward_rate"])
        check(f"  switch rate", c["sr"], row["mean_switch_rate"])
        check(f"  mean ICI (s)", c["ici"], row["mean_ici_s"])

    # ════════════════════════════════════════════════════════════
    section("TABLE 3 — MODEL COMPARISON (median AIC + full SD table)")
    # ════════════════════════════════════════════════════════════
    # Manuscript Table 3 AIC values (median, claimed_SD)
    table3_claims = {
        "hmm_4state":            (-3877.4, 869.4,   "HMM (4-state)"),
        "hmm_3state":            (-1539.6, 364.7,   "HMM (3-state)"),
        "hmm_2state":            (-460.3,  2996.4,  "HMM (2-state)"),
        "mvt_threshold":         (1021.2,  113.6,   "MVT Threshold"),
        "hill_climbing":         (1097.0,  92.9,    "Hill-Climbing"),
        "q_dual_alpha":          (1106.7,  66.2,    "Q-Dual-Alpha"),
        "melioration":           (1105.6,  68.6,    "Melioration"),
        "q_learning":            (1105.6,  68.6,    "Q-Learning"),
        "q_forgetting":          (1113.1,  67.8,    "Q-Forgetting"),
        "generalized_matching_law": (1533.0, 147.0, "Gen. Matching Law"),
        "ratio_invariance":      (1242.1,  62.6,    "Ratio Invariance"),
        "behavioral_momentum":   (1281.5,  73.3,    "Behav. Momentum"),
        "softmax_mvt":           (1202.9,  69.0,    "Softmax MVT"),
        "wsls":                  (1339.9,  69.5,    "WSLS"),
        "kinetic":               (1271.1,  63.6,    "Kinetic"),
        "patch_leaving":         (1469.6,  143.1,   "Patch-Leaving"),
        "random":                (1337.8,  69.3,    "Random"),
        "bias":                  (1339.4,  69.7,    "Bias"),
    }
    print(f"\n  {'Model':<25s} {'Median AIC':>12s} {'Manuscript':>12s} {'Actual SD':>10s} {'MS SD':>10s}")
    for model, (claim_med, claim_sd, label) in table3_claims.items():
        aic = mc[mc["model"] == model]["aic"].dropna()
        if len(aic) == 0:
            print(f"  {label:<25s} ---")
            continue
        med = aic.median()
        sd = aic.std()
        m_mark = "OK" if abs(med - claim_med) <= 1.0 else "XX"
        s_mark = "OK" if abs(sd - claim_sd) <= 2.0 else "XX"
        print(f"  {label:<25s} {med:>12.1f} {claim_med:>12.1f} [{m_mark}]  SD={sd:>8.1f}  MS={claim_sd:>6.1f} [{s_mark}]")

    # Classification accuracies (median)
    print("\n  Classification medians (accuracy / F1 / MCC):")
    acc_claims = {
        "hmm_4state":            (0.993, 0.993, 0.986, "HMM (4-state)"),
        "hmm_3state":            (0.993, 0.993, 0.986, "HMM (3-state)"),
        "hmm_2state":            (0.993, 0.993, 0.986, "HMM (2-state)"),
        "mvt_threshold":         (0.744, 0.740, 0.489, "MVT Threshold"),
        "hill_climbing":         (0.718, 0.710, 0.435, "Hill-Climbing"),
        "q_dual_alpha":          (0.675, 0.681, 0.352, "Q-Dual-Alpha"),
        "q_learning":            (0.675, 0.679, 0.351, "Q-Learning"),
        "melioration":           (0.675, 0.679, 0.351, "Melioration"),
        "q_forgetting":          (0.676, 0.672, 0.352, "Q-Forgetting"),
        "behavioral_momentum":   (0.639, 0.638, 0.287, "Behav. Momentum"),
        "generalized_matching_law": (0.636, 0.649, 0.273, "Gen. Matching Law"),
        "ratio_invariance":      (0.636, 0.639, 0.287, "Ratio Invariance"),
        "softmax_mvt":           (0.620, 0.688, 0.267, "Softmax MVT"),
        "wsls":                  (0.614, 0.457, 0.258, "WSLS"),
        "kinetic":               (0.611, 0.615, 0.228, "Kinetic"),
        "patch_leaving":         (0.605, 0.606, 0.212, "Patch-Leaving"),
        "random":                (0.497, 0.664, 0.000, "Random"),
        "bias":                  (0.503, 0.000, 0.000, "Bias"),
    }
    print(f"  {'Model':<22s} {'Acc MS/act':<16s} {'F1 MS/act':<16s} {'MCC MS/act':<16s}")
    for model, (a, f, m, label) in acc_claims.items():
        mdf = cm_full[cm_full["model"] == model]
        if len(mdf) == 0:
            continue
        act_a = mdf["accuracy"].median()
        act_f = mdf["f1"].median()
        act_m = mdf["mcc"].median()
        am = "OK" if abs(act_a - a) <= 0.01 else "XX"
        fm = "OK" if abs(act_f - f) <= 0.01 else "XX"
        mm = "OK" if abs(act_m - m) <= 0.01 else "XX"
        print(f"  {label:<22s} {a:.3f}/{act_a:.3f}[{am}]  {f:.3f}/{act_f:.3f}[{fm}]  {m:.3f}/{act_m:.3f}[{mm}]")

    # ════════════════════════════════════════════════════════════
    section("FAMILY-LEVEL AIC (Abstract & Results)")
    # ════════════════════════════════════════════════════════════
    families = {
        "Baseline":   ["random", "bias", "wsls"],
        "Historical": ["generalized_matching_law", "melioration",
                       "ratio_invariance", "kinetic", "behavioral_momentum",
                       "hill_climbing"],
        "Foraging":   ["mvt_threshold", "patch_leaving", "softmax_mvt"],
        "RL":         ["q_learning", "q_dual_alpha", "q_forgetting"],
        "HMM":        ["hmm_2state", "hmm_3state", "hmm_4state"],
    }
    family_claims = {
        "HMM":        (-3096.50, "mean"),
        "RL":         (1043, "mean"),
        "Historical": (1186, "mean"),
        "Foraging":   (1216, "mean"),
    }
    for fam, mods in families.items():
        aic = mc[mc["model"].isin(mods)]["aic"].dropna()
        if len(aic) == 0:
            continue
        mean_aic = aic.mean()
        med_aic = aic.median()
        claim = family_claims.get(fam)
        ms = f"MS: {claim[0]}" if claim else ""
        print(f"  {fam:<12s} mean={mean_aic:>9.1f}  median={med_aic:>9.1f}  {ms}")

    # HMM family AIC range (Abstract: -2010 to -4183)
    hmm_mean_by_model = {}
    for m in ["hmm_2state", "hmm_3state", "hmm_4state"]:
        vals = mc[mc["model"] == m]["aic"].dropna()
        hmm_mean_by_model[m] = (vals.mean(), vals.median())
    print("\n  Abstract HMM AIC range '-2010 to -4183' (means):")
    print(f"    HMM 2-state mean: {hmm_mean_by_model['hmm_2state'][0]:.1f}")
    print(f"    HMM 3-state mean: {hmm_mean_by_model['hmm_3state'][0]:.1f}")
    print(f"    HMM 4-state mean: {hmm_mean_by_model['hmm_4state'][0]:.1f}")

    # ════════════════════════════════════════════════════════════
    section("FAMILY NEXT-RESPONSE PREDICTIVE ACCURACY (Abstract)")
    # ════════════════════════════════════════════════════════════
    abstract_acc_claims = {
        "HMM":        0.88,
        "EDM":        0.74,
        "RL":         0.69,
        "Historical": 0.66,
        "Foraging":   0.66,
        "Baseline":   0.53,
    }
    # Build per-model median accuracy lookup
    med_acc = cm_full.groupby("model")["accuracy"].median()
    print(f"  Family accuracy (mean of per-model medians):")
    for fam, mods in families.items():
        vals = [med_acc.get(m) for m in mods if m in med_acc.index]
        vals = [v for v in vals if pd.notna(v)]
        if vals:
            fam_acc = np.mean(vals)
            claimed = abstract_acc_claims.get(fam)
            mark = "OK" if claimed and abs(fam_acc - claimed) <= 0.01 else "  "
            claim_s = f"MS={claimed:.2f}" if claimed else ""
            print(f"    [{mark}] {fam:<12s} actual={fam_acc:.3f}  {claim_s}")

    # ════════════════════════════════════════════════════════════
    section("ADAPTATION LAGS (from metrics_df)")
    # ════════════════════════════════════════════════════════════
    for pid, (mean_c, med_c, sd_c, label) in {
        2: (4.03, 3.01, 4.50, "Phase 1->2 (symmetric to A-adv)"),
        3: (6.65, 5.85, 6.81, "Phase 2->3 (A-adv to B-adv)"),
        4: (3.24, 1.39, 5.74, "Phase 3->4 (B-adv to scarcity)"),
    }.items():
        col = f"adaptation_lag_phase{pid}_s"
        if col not in metrics_df.columns:
            print(f"  {label}: col not found")
            continue
        vals = metrics_df[col].dropna()
        print(f"\n  {label}:")
        check(f"  mean lag (s)", mean_c, vals.mean(), tol=0.1)
        check(f"  median lag (s)", med_c, vals.median(), tol=0.1)
        check(f"  SD lag (s)", sd_c, vals.std(), tol=0.2)
        print(f"     n_adapted={len(vals)}/{len(metrics_df)}")

    # ════════════════════════════════════════════════════════════
    section("OPTIMALITY ANALYSIS")
    # ════════════════════════════════════════════════════════════
    # Manuscript claims:
    #  Phase 2: 25/60 (41.7%) within +/-5% of optimal
    #  Phase 3: 19/60 (31.7%) within +/-5% of optimal
    #  Phase 2: 21/60 (35.0%) over-exploited, 14/60 (23.3%) under-exploited
    #  Phase 3: 24/60 (40.0%) over-exploited, 17/60 (28.3%) under-exploited
    opt_path = os.path.join(OUT, "optimal_allocation.csv")
    if os.path.exists(opt_path):
        opt_df = pd.read_csv(opt_path)
        print(f"\n  optimal_allocation.csv columns: {list(opt_df.columns)}")
        print(f"  phases: {opt_df['phase'].unique() if 'phase' in opt_df.columns else 'none'}")

        for phase_val, label in [("Ph 2", "Phase 2"), ("Ph 3", "Phase 3")]:
            pdf = opt_df[opt_df["phase"] == phase_val] if "phase" in opt_df.columns else pd.DataFrame()
            if len(pdf) == 0:
                continue
            # distance_from_optimal column
            if "distance_from_optimal" in pdf.columns:
                d = pdf["distance_from_optimal"]
                n_within = (d.abs() <= 0.05).sum()
                n_over = (d > 0.05).sum()
                n_under = (d < -0.05).sum()
                n_tot = len(pdf)
                print(f"\n  {label}: n={n_tot}")
                print(f"    within +/-5%:    {n_within}/{n_tot} ({100*n_within/n_tot:.1f}%)")
                print(f"    over-exploited:  {n_over}/{n_tot} ({100*n_over/n_tot:.1f}%)")
                print(f"    under-exploited: {n_under}/{n_tot} ({100*n_under/n_tot:.1f}%)")

    # ════════════════════════════════════════════════════════════
    section("EDM DIMENSIONALITY (Simplex)")
    # ════════════════════════════════════════════════════════════
    # Manuscript: E=1: 28%, E=2: 30%, E=3-4: 23%, E=5-6: 18%, median E=2
    edm_path = os.path.join(OUT, "edm_simplex.csv")
    if os.path.exists(edm_path):
        edm = pd.read_csv(edm_path)
        if "best_E" in edm.columns:
            best_E = edm["best_E"]
            n = len(best_E)
            print(f"\n  n={n}")
            print(f"  median E = {best_E.median():.1f}")
            for E_range, label in [((1, 1), "E=1"), ((2, 2), "E=2"),
                                    ((3, 4), "E=3-4"), ((5, 6), "E=5-6")]:
                mask = (best_E >= E_range[0]) & (best_E <= E_range[1])
                pct = 100 * mask.sum() / n
                print(f"  {label}: {mask.sum()}/{n} ({pct:.1f}%)")

    # ════════════════════════════════════════════════════════════
    section("S-MAP PREDICTIVE SKILL")
    # ════════════════════════════════════════════════════════════
    # Manuscript: mean rho = 0.789, median = 0.804, median best theta = 0.1
    smap_path = os.path.join(OUT, "smap_results.csv")
    if os.path.exists(smap_path):
        smap = pd.read_csv(smap_path)
        print(f"  columns: {list(smap.columns)}")
        if "smap_rho" in smap.columns:
            rho = smap["smap_rho"].dropna()
            print(f"  S-Map rho: mean={rho.mean():.3f}, median={rho.median():.3f}")
        if "best_theta" in smap.columns:
            theta = smap["best_theta"].dropna()
            print(f"  Best theta: median={theta.median():.3f}")

    # ════════════════════════════════════════════════════════════
    section("CCM CAUSAL ASYMMETRY (% above identity)")
    # ════════════════════════════════════════════════════════════
    # Manuscript: P1: 39/60 (65%), P2: 39/58 (67%), P3: 42/58 (72%), P4: 36/60 (60%)
    ccm_path = os.path.join(OUT, "ccm_results.csv")
    if os.path.exists(ccm_path):
        ccm = pd.read_csv(ccm_path)
        print(f"  columns: {list(ccm.columns)}")
        if "phase_id" in ccm.columns and "rho_reward_causes_choice" in ccm.columns:
            for pid in [1, 2, 3, 4]:
                pdf = ccm[ccm["phase_id"] == pid]
                # Aggregate to session-level: take max rho per session for each direction
                agg = pdf.groupby("session_id").agg(
                    rho_rc=("rho_reward_causes_choice", "max"),
                    rho_cr=("rho_choice_causes_reward", "max"),
                )
                above = (agg["rho_rc"] > agg["rho_cr"]).sum()
                n = len(agg)
                print(f"  Phase {pid}: {above}/{n} ({100*above/n:.0f}%) reward->choice > choice->reward")

    # ════════════════════════════════════════════════════════════
    section("RQA METRICS (full session and by phase)")
    # ════════════════════════════════════════════════════════════
    # Manuscript:
    # Full: recurrence rate M=.332 (SD=.073)
    # Full determinism: M=.947, SD=.033, Md=.961
    # Full mean DL: M=5.97 (SD=2.07)
    # Full diagonal entropy: M=2.36 (SD=0.40)
    # Full laminarity: M=.982 (SD=.014)
    # Full trapping time: M=8.78 (SD=3.59)
    rqa_path = os.path.join(OUT, "rqa_metrics.csv")
    if os.path.exists(rqa_path):
        rqa = pd.read_csv(rqa_path)
        full = rqa[rqa["scope"] == "full_session"] if "scope" in rqa.columns else rqa
        print(f"\n  Full session (n={len(full)}):")
        rqa_claims = {
            "recurrence_rate":       (0.332, 0.073),
            "determinism":           (0.947, 0.033),
            "mean_diagonal_length":  (5.97, 2.07),
            "entropy_diagonal":      (2.36, 0.40),
            "laminarity":            (0.982, 0.014),
            "trapping_time":         (8.78, 3.59),
        }
        for col, (mc_m, mc_sd) in rqa_claims.items():
            if col in full.columns:
                vals = full[col].dropna()
                check(f"  {col} mean", mc_m, vals.mean(), tol=0.02)
                check(f"  {col} SD", mc_sd, vals.std(), tol=0.05)

        # Per-phase RQA summary (for Results section claims)
        print(f"\n  Per-phase means (for Results text):")
        phase_map = {1: "Symmetric", 2: "A-adv", 3: "B-adv", 4: "Scarcity"}
        for col in ["determinism", "mean_diagonal_length", "max_diagonal_length",
                    "entropy_diagonal", "laminarity", "trapping_time"]:
            if col not in rqa.columns:
                continue
            print(f"  {col}:")
            for pid in [1, 2, 3, 4]:
                pdf = rqa[(rqa["phase_id"] == pid) &
                          (rqa["scope"] != "full_session")] if "scope" in rqa.columns \
                      else rqa[rqa["phase_id"] == pid]
                if len(pdf) > 0:
                    print(f"    Phase {pid} ({phase_map[pid]:<10s}): "
                          f"mean={pdf[col].mean():.3f}, SD={pdf[col].std():.3f}")

    # ════════════════════════════════════════════════════════════
    section("HMM BEST-FIT MODEL COUNTS")
    # ════════════════════════════════════════════════════════════
    # Manuscript: 4-state best for 54/60 (90%), 3-state best for 6/60 (10%)
    hmm_only = mc[mc["model"].isin(["hmm_2state", "hmm_3state", "hmm_4state"])]
    pivot = hmm_only.pivot_table(index="session_id", columns="model",
                                  values="aic", aggfunc="first")
    if len(pivot) > 0:
        pivot["best"] = pivot.idxmin(axis=1)
        best_counts = pivot["best"].value_counts()
        n = len(pivot)
        print(f"  n={n}")
        for model in ["hmm_2state", "hmm_3state", "hmm_4state"]:
            cnt = best_counts.get(model, 0)
            print(f"  {model}: {cnt}/{n} ({100*cnt/n:.1f}%)")

    # Also: 2-state with negative AIC for how many
    hmm2 = mc[mc["model"] == "hmm_2state"]["aic"].dropna()
    n_neg = (hmm2 < 0).sum()
    n_pos = (hmm2 > 0).sum()
    print(f"\n  HMM 2-state AIC sign distribution:")
    print(f"    negative AIC: {n_neg}/{len(hmm2)} ({100*n_neg/len(hmm2):.1f}%)")
    print(f"    positive AIC: {n_pos}/{len(hmm2)} ({100*n_pos/len(hmm2):.1f}%)")
    print(f"  Manuscript claimed: 34/60 (56.7%) negative, 24/60 (40%) positive")

    print(f"\n{'=' * 70}\nDONE\n{'=' * 70}")


if __name__ == "__main__":
    main()
