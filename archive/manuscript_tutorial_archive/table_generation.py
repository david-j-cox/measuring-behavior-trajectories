"""Generate manuscript-ready tables (CSV + Markdown)."""

import os
import numpy as np
import pandas as pd


def generate_all_tables(events_df: pd.DataFrame, metrics_df: pd.DataFrame,
                        analysis_results: dict, config: dict,
                        output_dir: str) -> dict:
    """Produce all manuscript tables. Returns dict of DataFrames."""
    tbl_dir = os.path.join(output_dir, "manuscript", "tables")
    os.makedirs(tbl_dir, exist_ok=True)

    tables = {}

    tables["table1_session_summary"] = _table1_session_summary(metrics_df)
    tables["table2_phase_metrics"] = _table2_phase_metrics(events_df, config)
    tables["table3_transition_metrics"] = _table3_transition_metrics(
        events_df, metrics_df, config
    )
    tables["table4_pulse_response"] = _table4_pulse_response(events_df, config)
    tables["table5_matching_law"] = _table5_matching_law(analysis_results)
    tables["table6_model_fits"] = _table6_model_fits(analysis_results)
    tables["table7_dynamical_summary"] = _table7_dynamical_summary(analysis_results)

    # Glossary table is standalone (no data dependencies)
    glossary_df = generate_glossary_table(os.path.join(output_dir, "manuscript"))
    if glossary_df is not None and len(glossary_df) > 0:
        tables["glossary_dynamical_systems"] = glossary_df

    for name, df in tables.items():
        if df is not None and len(df) > 0:
            df.to_csv(os.path.join(tbl_dir, f"{name}.csv"), index=False)
            with open(os.path.join(tbl_dir, f"{name}.md"), "w") as f:
                f.write(df.to_markdown(index=False))
                f.write("\n")

    print(f"Tables: {len(tables)} generated in {tbl_dir}")
    return tables


# ── Table 1 ─────────────────────────────────────────────────────────────────

def _table1_session_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Session-level descriptive statistics."""
    measures = {
        "N (sessions)": len(metrics_df),
        "N (participants)": metrics_df["participant_id"].nunique(),
    }

    stat_cols = [
        ("total_clicks", "Clicks per session"),
        ("session_duration_s", "Session duration (s)"),
        ("total_rewards", "Rewards per session"),
        ("final_score", "Final score"),
        ("overall_reward_rate", "Overall reward rate"),
        ("overall_switch_rate", "Switch rate"),
        ("mean_run_length", "Mean run length"),
        ("choice_entropy", "Choice entropy (bits)"),
    ]

    rows = [{"Measure": k, "Value": str(v)} for k, v in measures.items()]
    for col, label in stat_cols:
        if col in metrics_df.columns:
            vals = metrics_df[col].dropna()
            # Use more decimal places for entropy to avoid hiding variation
            if col == "choice_entropy":
                rows.append({
                    "Measure": label,
                    "Value": f"{vals.mean():.4f} ({vals.std():.4f})",
                })
            else:
                rows.append({
                    "Measure": label,
                    "Value": f"{vals.mean():.2f} ({vals.std():.2f})",
                })

    return pd.DataFrame(rows)


# ── Table 2 ─────────────────────────────────────────────────────────────────

def _table2_phase_metrics(events_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Per-phase behavioral metrics."""
    phase_boundaries = config.get("phase_boundaries", [])
    rows = []
    for pb in phase_boundaries:
        pid = pb["id"]
        pdf = events_df[events_df["phase_id"] == pid]
        if len(pdf) == 0:
            continue
        per_session = pdf.groupby("session_id").agg(
            reward_rate=("reward_outcome", "mean"),
            switch_rate=("switch_flag_verified", "mean"),
            run_length=("run_length_current", "mean"),
        )
        row = {
            "Phase": f"{pid}: {pb['label']}",
            "N clicks": len(pdf),
        }
        for col in ["reward_rate", "switch_rate", "run_length"]:
            row[col.replace("_", " ").title()] = (
                f"{per_session[col].mean():.3f} ({per_session[col].std():.3f})"
            )
        if "choice_optimal" in pdf.columns:
            opt = pdf.groupby("session_id")["choice_optimal"].mean()
            row["Proportion Optimal"] = f"{opt.mean():.3f} ({opt.std():.3f})"
        rows.append(row)
    return pd.DataFrame(rows)


# ── Table 3 ─────────────────────────────────────────────────────────────────

def _table3_transition_metrics(events_df: pd.DataFrame,
                                metrics_df: pd.DataFrame,
                                config: dict) -> pd.DataFrame:
    """Transition-level dynamics: adaptation lag, overshoot, switch burst."""
    phase_boundaries = config.get("phase_boundaries", [])
    pre_s = config.get("transition_pre_window_s", 15)
    post_s = config.get("transition_post_window_s", 30)
    rows = []

    for i in range(1, len(phase_boundaries)):
        pb = phase_boundaries[i]
        t_trans = pb["start_ms"] / 1000
        label = f"Phase {phase_boundaries[i-1]['id']} -> {pb['id']}"

        # Adaptation lag
        lag_col = f"adaptation_lag_phase{pb['id']}_s"
        lag_vals = metrics_df[lag_col].dropna() if lag_col in metrics_df.columns else pd.Series(dtype=float)

        # Compute transition-window metrics per session
        overshoots = []
        switch_bursts = []
        instabilities = []

        for sid, sdf in events_df.groupby("session_id"):
            pre = sdf[(sdf["elapsed_time_s"] >= t_trans - pre_s) &
                       (sdf["elapsed_time_s"] < t_trans)]
            post = sdf[(sdf["elapsed_time_s"] >= t_trans) &
                        (sdf["elapsed_time_s"] < t_trans + post_s)]
            if len(pre) < 3 or len(post) < 3:
                continue

            pre_prop = pre["choice_a"].mean()
            post_prop = post["choice_a"].mean()

            # Overshoot: max deviation beyond post-window mean in first 10s
            early_post = sdf[(sdf["elapsed_time_s"] >= t_trans) &
                              (sdf["elapsed_time_s"] < t_trans + 10)]
            if "rolling_choice_prop_a_clicks" in early_post.columns and len(early_post) > 0:
                rolling = early_post["rolling_choice_prop_a_clicks"]
                direction = np.sign(post_prop - pre_prop)
                if direction > 0:
                    overshoot = rolling.max() - post_prop
                elif direction < 0:
                    overshoot = post_prop - rolling.min()
                else:
                    overshoot = 0
                overshoots.append(max(0, overshoot))

            # Switch burst: switch rate in first 5s post-transition
            burst_window = sdf[(sdf["elapsed_time_s"] >= t_trans) &
                                (sdf["elapsed_time_s"] < t_trans + 5)]
            if "switch_flag_verified" in burst_window.columns and len(burst_window) > 0:
                switch_bursts.append(burst_window["switch_flag_verified"].mean())

            # Post-transition instability: variance of rolling prop in post window
            if "rolling_choice_prop_a_clicks" in post.columns:
                instabilities.append(post["rolling_choice_prop_a_clicks"].var())

        row = {"Transition": label}

        if len(lag_vals) > 0:
            row["Adaptation Lag (s)"] = f"{lag_vals.mean():.1f} ({lag_vals.std():.1f})"
            row["N Adapted"] = f"{len(lag_vals)}/{len(metrics_df)}"
        else:
            row["Adaptation Lag (s)"] = "N/A"
            row["N Adapted"] = "N/A"

        row["Overshoot"] = (
            f"{np.mean(overshoots):.3f} ({np.std(overshoots):.3f})"
            if overshoots else "N/A"
        )
        row["Switch Burst Rate"] = (
            f"{np.mean(switch_bursts):.3f} ({np.std(switch_bursts):.3f})"
            if switch_bursts else "N/A"
        )
        row["Post-Transition Instability"] = (
            f"{np.mean(instabilities):.4f} ({np.std(instabilities):.4f})"
            if instabilities else "N/A"
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ── Table 4 ─────────────────────────────────────────────────────────────────

def _table4_pulse_response(events_df: pd.DataFrame,
                           config: dict) -> pd.DataFrame:
    """Pulse response metrics during Phase 4."""
    bp = config.get("bonus_pulses", {})
    if not bp:
        return pd.DataFrame()

    phase4_start = bp.get("phase4_start_ms", 270000)
    interval = bp.get("interval_ms", 12000)
    duration = bp.get("duration_ms", 3000)
    sequence = bp.get("sequence", [])

    phase4 = events_df[events_df["phase_id"] == 4].copy()
    if len(phase4) == 0:
        return pd.DataFrame()

    pulse_rows = []
    for i, target in enumerate(sequence):
        onset_ms = phase4_start + i * interval
        offset_ms = onset_ms + duration
        onset_s = onset_ms / 1000
        offset_s = offset_ms / 1000

        latencies = []
        exploit_rates = []
        reward_gains = []

        for sid, sdf in phase4.groupby("session_id"):
            # Pre-pulse baseline (5s before)
            pre = sdf[(sdf["elapsed_time_s"] >= onset_s - 5) &
                       (sdf["elapsed_time_s"] < onset_s)]
            # During pulse
            during = sdf[(sdf["elapsed_time_s"] >= onset_s) &
                          (sdf["elapsed_time_s"] < offset_s)]

            if len(during) == 0:
                continue

            # Exploitation rate: fraction choosing the pulsed option during pulse
            chose_target = (during["chosen_option"] == target).mean()
            exploit_rates.append(chose_target)

            # Response latency: time from pulse onset to first choice of target
            target_clicks = during[during["chosen_option"] == target]
            if len(target_clicks) > 0:
                first_target_t = target_clicks["elapsed_time_s"].iloc[0]
                latencies.append(first_target_t - onset_s)

            # Reward gain vs pre-pulse
            if len(pre) > 0:
                reward_gains.append(during["reward_outcome"].mean() - pre["reward_outcome"].mean())

        pulse_rows.append({
            "Pulse": f"{i+1} ({target})",
            "Response Latency (s)": (
                f"{np.mean(latencies):.2f} ({np.std(latencies):.2f})"
                if latencies else "N/A"
            ),
            "Exploitation Rate": (
                f"{np.mean(exploit_rates):.3f} ({np.std(exploit_rates):.3f})"
                if exploit_rates else "N/A"
            ),
            "Reward Gain": (
                f"{np.mean(reward_gains):.3f} ({np.std(reward_gains):.3f})"
                if reward_gains else "N/A"
            ),
        })

    return pd.DataFrame(pulse_rows)


# ── Table 5 ─────────────────────────────────────────────────────────────────

def _table5_matching_law(analysis_results: dict) -> pd.DataFrame:
    """Generalized matching law (Baum, 1974) parameter estimates per session.

    Reports sensitivity (s), bias (b), R², n bins, and classification
    (undermatching / strict matching / overmatching).
    """
    mc = analysis_results.get("model_comparison")
    if mc is None or len(mc) == 0:
        return pd.DataFrame()

    matching_df = mc[mc["model"] == "generalized_matching_law"].copy()
    if len(matching_df) == 0:
        return pd.DataFrame()

    def _classify(s):
        if s < 0.9:
            return "Undermatching"
        elif s > 1.1:
            return "Overmatching"
        else:
            return "Strict matching"

    rows = []
    for _, r in matching_df.iterrows():
        rows.append({
            "Session": r["session_id"],
            "Sensitivity (s)": round(r["sensitivity"], 3),
            "Bias (b)": round(r["bias"], 3),
            "R²": round(r["r_squared"], 3),
            "n bins": int(r["n_bins"]) if pd.notna(r["n_bins"]) else "",
            "Classification": _classify(r["sensitivity"]),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Append summary row
    s_vals = df["Sensitivity (s)"]
    b_vals = df["Bias (b)"]
    r2_vals = df["R²"]
    n_under = (df["Classification"] == "Undermatching").sum()
    n_strict = (df["Classification"] == "Strict matching").sum()
    n_over = (df["Classification"] == "Overmatching").sum()
    summary_row = pd.DataFrame([{
        "Session": f"Mean (SD), N={len(df)}",
        "Sensitivity (s)": f"{s_vals.mean():.3f} ({s_vals.std():.3f})",
        "Bias (b)": f"{b_vals.mean():.3f} ({b_vals.std():.3f})",
        "R²": f"{r2_vals.mean():.3f} ({r2_vals.std():.3f})",
        "n bins": "",
        "Classification": f"{n_under} under / {n_strict} strict / {n_over} over",
    }])
    df = pd.concat([df, summary_row], ignore_index=True)

    return df


# ── Table 6 ─────────────────────────────────────────────────────────────────

def _table6_model_fits(analysis_results: dict) -> pd.DataFrame:
    """Trial-level model comparison summary table.

    Matching-law models are excluded from the trial-level BIC comparison
    because they use log-ratio regression (session-level R²) rather than
    trial-level likelihoods; including them produces NaN BIC values.
    """
    mc = analysis_results.get("model_comparison")
    if mc is None or len(mc) == 0:
        return pd.DataFrame()

    # Exclude matching-law models from the trial-level comparison table
    # (they use a different fitting approach and produce NaN likelihoods)
    trial_models = mc[mc["model"] != "generalized_matching_law"].copy()

    agg = trial_models.groupby("model").agg(
        mean_nll=("nll", "mean"),
        sd_nll=("nll", "std"),
        mean_aic=("aic", "mean"),
        sd_aic=("aic", "std"),
        mean_bic=("bic", "mean"),
        sd_bic=("bic", "std"),
        n_params=("n_params", "first"),
        n_sessions=("session_id", "nunique"),
    ).sort_values("mean_bic")

    rows = []
    for model, r in agg.iterrows():
        rows.append({
            "Model": model,
            "k": int(r["n_params"]),
            "NLL": f"{r['mean_nll']:.1f} ({r['sd_nll']:.1f})",
            "AIC": f"{r['mean_aic']:.1f} ({r['sd_aic']:.1f})",
            "BIC": f"{r['mean_bic']:.1f} ({r['sd_bic']:.1f})",
            "N sessions": int(r["n_sessions"]),
        })

    return pd.DataFrame(rows)


# ── Table 7 ─────────────────────────────────────────────────────────────────

def _table7_dynamical_summary(analysis_results: dict) -> pd.DataFrame:
    """Summary table of dynamical analysis metrics."""
    rows = []

    # RQA
    rqa = analysis_results.get("rqa")
    if rqa is not None and isinstance(rqa, pd.DataFrame) and len(rqa) > 0:
        for col, label in [("recurrence_rate", "Recurrence Rate (RQA)"),
                           ("determinism", "Determinism (RQA)"),
                           ("laminarity", "Laminarity (RQA)"),
                           ("trapping_time", "Trapping Time (RQA)")]:
            if col in rqa.columns:
                vals = rqa[col].dropna()
                if len(vals) > 0:
                    rows.append({
                        "Metric": label,
                        "Mean (SD)": f"{vals.mean():.3f} ({vals.std():.3f})",
                        "Median": f"{vals.median():.3f}",
                        "Range": f"[{vals.min():.3f}, {vals.max():.3f}]",
                        "N": len(vals),
                    })

    # DFA
    dfa = analysis_results.get("dfa")
    if dfa is not None and isinstance(dfa, pd.DataFrame) and len(dfa) > 0:
        full = dfa[dfa["scope"] == "full_session"] if "scope" in dfa.columns else dfa
        if "dfa_alpha" in full.columns:
            vals = full["dfa_alpha"].dropna()
            if len(vals) > 0:
                rows.append({
                    "Metric": "DFA Exponent (alpha)",
                    "Mean (SD)": f"{vals.mean():.3f} ({vals.std():.3f})",
                    "Median": f"{vals.median():.3f}",
                    "Range": f"[{vals.min():.3f}, {vals.max():.3f}]",
                    "N": len(vals),
                })

    # Sample Entropy
    se = analysis_results.get("sample_entropy")
    if se is not None and isinstance(se, pd.DataFrame) and len(se) > 0:
        full = se[se["scope"] == "full_session"] if "scope" in se.columns else se
        if "sample_entropy" in full.columns:
            vals = full["sample_entropy"].dropna()
            if len(vals) > 0:
                rows.append({
                    "Metric": "Sample Entropy",
                    "Mean (SD)": f"{vals.mean():.3f} ({vals.std():.3f})",
                    "Median": f"{vals.median():.3f}",
                    "Range": f"[{vals.min():.3f}, {vals.max():.3f}]",
                    "N": len(vals),
                })

    # EDM simplex
    edm = analysis_results.get("edm_simplex")
    if edm is not None and isinstance(edm, pd.DataFrame) and len(edm) > 0:
        if "rho" in edm.columns:
            vals = edm["rho"].dropna()
            if len(vals) > 0:
                rows.append({
                    "Metric": "Simplex Projection rho (EDM)",
                    "Mean (SD)": f"{vals.mean():.3f} ({vals.std():.3f})",
                    "Median": f"{vals.median():.3f}",
                    "Range": f"[{vals.min():.3f}, {vals.max():.3f}]",
                    "N": len(vals),
                })

    # S-Map nonlinearity
    smap = analysis_results.get("smap")
    if smap is not None and isinstance(smap, pd.DataFrame) and len(smap) > 0:
        if "nonlinearity" in smap.columns:
            vals = smap["nonlinearity"].dropna()
            if len(vals) > 0:
                rows.append({
                    "Metric": "Nonlinearity (S-Map delta-rho)",
                    "Mean (SD)": f"{vals.mean():.4f} ({vals.std():.4f})",
                    "Median": f"{vals.median():.4f}",
                    "Range": f"[{vals.min():.4f}, {vals.max():.4f}]",
                    "N": len(vals),
                })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ── Glossary Table ─────────────────────────────────────────────────────────

def generate_glossary_table(output_dir: str) -> pd.DataFrame:
    """Generate a glossary mapping dynamical systems concepts to
    behavior-analytic equivalents.

    Writes both CSV and Markdown to ``output_dir/tables/``.  Returns the
    glossary as a DataFrame.
    """
    entries = [
        {
            "Dynamical Systems Term": "Attractor",
            "Definition": (
                "A state or set of states toward which a system tends to evolve "
                "over time, regardless of starting conditions."
            ),
            "Behavior-Analytic Equivalent/Analog": (
                "Steady-state preference / equilibrium response allocation"
            ),
            "Analysis Method(s)": "Phase-space reconstruction; RQA",
        },
        {
            "Dynamical Systems Term": "Trajectory",
            "Definition": (
                "The path traced by a system through its state space as it "
                "evolves over time."
            ),
            "Behavior-Analytic Equivalent/Analog": (
                "Behavioral time series / response path over time"
            ),
            "Analysis Method(s)": "Time-series construction; all pipeline steps",
        },
        {
            "Dynamical Systems Term": "Phase space / State space",
            "Definition": (
                "A multi-dimensional space in which each axis represents one "
                "variable of the system, so that every possible state "
                "corresponds to a unique point."
            ),
            "Behavior-Analytic Equivalent/Analog": (
                "Multi-dimensional representation of behavior (e.g., choice "
                "proportion \u00d7 reward rate plotted together)"
            ),
            "Analysis Method(s)": "Delay embedding; phase-space reconstruction",
        },
        {
            "Dynamical Systems Term": "Bifurcation",
            "Definition": (
                "A qualitative change in the system's dynamics (e.g., number "
                "or stability of attractors) caused by a smooth change in a "
                "control parameter."
            ),
            "Behavior-Analytic Equivalent/Analog": (
                "Regime shift / abrupt change in contingency\u2013behavior "
                "relationship"
            ),
            "Analysis Method(s)": "Phase-transition analysis",
        },
        {
            "Dynamical Systems Term": "Recurrence",
            "Definition": (
                "The property of a trajectory returning to a neighborhood of "
                "a previously visited state in phase space."
            ),
            "Behavior-Analytic Equivalent/Analog": (
                "Pattern revisitation / returning to a previously observed "
                "behavioral state"
            ),
            "Analysis Method(s)": "Recurrence Quantification Analysis (RQA)",
        },
        {
            "Dynamical Systems Term": "Determinism (RQA)",
            "Definition": (
                "The proportion of recurrent points that fall on diagonal "
                "lines in a recurrence plot, indicating predictable sequential "
                "structure."
            ),
            "Behavior-Analytic Equivalent/Analog": (
                "Sequential predictability / the degree to which behavioral "
                "sequences form repeating patterns"
            ),
            "Analysis Method(s)": "Recurrence Quantification Analysis (RQA)",
        },
        {
            "Dynamical Systems Term": "Hurst exponent / DFA \u03b1",
            "Definition": (
                "A scaling exponent that quantifies the degree of long-range "
                "temporal correlations in a time series. Values above 0.5 "
                "indicate persistent (positively correlated) fluctuations."
            ),
            "Behavior-Analytic Equivalent/Analog": (
                "Long-range dependence / behavioral persistence or memory "
                "across time scales"
            ),
            "Analysis Method(s)": "Detrended Fluctuation Analysis (DFA); fractal analysis",
        },
        {
            "Dynamical Systems Term": "Sample entropy",
            "Definition": (
                "A measure of the complexity or irregularity of a time series, "
                "defined as the negative natural logarithm of the conditional "
                "probability that similar patterns remain similar at the next "
                "point."
            ),
            "Behavior-Analytic Equivalent/Analog": (
                "Behavioral complexity/regularity / unpredictability of the "
                "choice sequence"
            ),
            "Analysis Method(s)": "Entropy estimation; fractal analysis",
        },
        {
            "Dynamical Systems Term": "Embedding dimension",
            "Definition": (
                "The number of time-delayed copies of a scalar time series "
                "needed to unfold the system's dynamics in a reconstructed "
                "phase space (per Takens' theorem)."
            ),
            "Behavior-Analytic Equivalent/Analog": (
                "Effective degrees of freedom / the minimum number of lagged "
                "variables needed to reconstruct the system's dynamics"
            ),
            "Analysis Method(s)": "Delay embedding; simplex projection",
        },
        {
            "Dynamical Systems Term": "Convergent cross-mapping (CCM)",
            "Definition": (
                "A method for detecting causal relationships between time "
                "series by testing whether the state-space reconstruction of "
                "one variable can predict the other, with prediction improving "
                "as more data are used."
            ),
            "Behavior-Analytic Equivalent/Analog": (
                "Causal coupling / testing whether one variable (e.g., reward) "
                "causally drives another (e.g., choice) beyond correlation"
            ),
            "Analysis Method(s)": "Convergent cross-mapping (CCM)",
        },
        {
            "Dynamical Systems Term": "S-Map (state-dependent mapping)",
            "Definition": (
                "A locally weighted forecasting method that allows the "
                "relationship between variables to change depending on the "
                "current location in state space."
            ),
            "Behavior-Analytic Equivalent/Analog": (
                "Context-sensitive dynamics / whether the rules governing "
                "behavior change depending on the current behavioral state"
            ),
            "Analysis Method(s)": "S-Map analysis",
        },
        {
            "Dynamical Systems Term": "Simplex projection",
            "Definition": (
                "A nearest-neighbor forecasting method that predicts future "
                "values by finding similar past states in a delay-embedded "
                "phase space and averaging their successors."
            ),
            "Behavior-Analytic Equivalent/Analog": (
                "Nonlinear forecasting / predicting future behavior from past "
                "trajectories using nearest-neighbor methods"
            ),
            "Analysis Method(s)": "Simplex projection; embedding dimension estimation",
        },
        {
            "Dynamical Systems Term": "Hysteresis",
            "Definition": (
                "The dependence of a system's state on its history, so that "
                "the path followed when a parameter increases differs from "
                "the path when it decreases."
            ),
            "Behavior-Analytic Equivalent/Analog": (
                "Path dependence / behavior depending not just on current "
                "contingencies but on the history of contingency changes"
            ),
            "Analysis Method(s)": "Phase-transition analysis; transition metrics",
        },
        {
            "Dynamical Systems Term": "Laminarity (RQA)",
            "Definition": (
                "The proportion of recurrent points that form vertical lines "
                "in a recurrence plot, reflecting epochs where the system "
                "remains in roughly the same state."
            ),
            "Behavior-Analytic Equivalent/Analog": (
                "Behavioral trapping / tendency to persist in the same state "
                "for extended periods"
            ),
            "Analysis Method(s)": "Recurrence Quantification Analysis (RQA)",
        },
        {
            "Dynamical Systems Term": "Hidden Markov Model (HMM)",
            "Definition": (
                "A probabilistic model positing a sequence of unobserved "
                "(hidden) states, each generating observations according to "
                "state-specific emission distributions, with transitions "
                "governed by a Markov chain."
            ),
            "Behavior-Analytic Equivalent/Analog": (
                "Latent behavioral states / unobserved internal states that "
                "generate observable choice patterns (cf. \u201cresponse states\u201d "
                "in behavior analysis)"
            ),
            "Analysis Method(s)": "Hidden Markov Model fitting; state classification",
        },
        {
            "Dynamical Systems Term": "Logistic regression model",
            "Definition": (
                "A statistical model predicting each choice as a weighted "
                "function of recent reward history and choice history. Unlike "
                "RL models, it makes no assumptions about value updating; "
                "unlike matching, it operates at the trial level."
            ),
            "Behavior-Analytic Equivalent/Analog": (
                "Recent-history weighting / predicting current choice from "
                "a weighted window of recent rewards and responses"
            ),
            "Analysis Method(s)": "Model comparison (BIC/AIC)",
        },
        {
            "Dynamical Systems Term": "Matching law",
            "Definition": (
                "The empirical generalization that the relative rate of "
                "responding on an alternative matches the relative rate of "
                "reinforcement obtained from that alternative."
            ),
            "Behavior-Analytic Equivalent/Analog": (
                "Molar contingency tracking / proportional allocation of "
                "behavior to relative reinforcement rates"
            ),
            "Analysis Method(s)": "Matching-law regression; RL model comparison",
        },
    ]

    df = pd.DataFrame(entries)

    tbl_dir = os.path.join(output_dir, "tables")
    os.makedirs(tbl_dir, exist_ok=True)

    csv_path = os.path.join(tbl_dir, "glossary_dynamical_systems.csv")
    md_path = os.path.join(tbl_dir, "glossary_dynamical_systems.md")

    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print(f"Glossary table written to {tbl_dir}")
    return df
