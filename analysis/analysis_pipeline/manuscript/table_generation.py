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
    tables["table5_model_fits"] = _table5_model_fits(analysis_results)

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

def _table5_model_fits(analysis_results: dict) -> pd.DataFrame:
    """Model comparison summary table."""
    mc = analysis_results.get("model_comparison")
    if mc is None or len(mc) == 0:
        return pd.DataFrame()

    agg = mc.groupby("model").agg(
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
