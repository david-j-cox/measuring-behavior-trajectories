"""Phase transition analysis: pre/post comparison and adaptation dynamics."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def run_phase_analysis(events_df: pd.DataFrame, metrics_df: pd.DataFrame,
                       config: dict, output_dir: str) -> dict:
    """Run all phase-level analyses and generate plots."""
    results = {}
    phase_boundaries = config.get("phase_boundaries", [])
    fmt = config.get("plot_format", "png")
    fig_dir = os.path.join(output_dir, "figures", "phase")

    # Per-phase descriptives
    results["phase_descriptives"] = _phase_descriptives(events_df, phase_boundaries)

    # Pre-post phase comparison (run before plotting so effect sizes are available)
    results["pre_post_tests"] = _pre_post_comparisons(events_df, phase_boundaries, config)

    # Phase transition plots with effect size annotations
    _plot_phase_transitions(events_df, config, fig_dir, fmt,
                            pre_post_tests=results["pre_post_tests"])

    # Adaptation lag analysis
    if "adaptation_lag_phase2_s" in metrics_df.columns:
        results["adaptation_lags"] = _adaptation_lag_summary(metrics_df)
        _plot_adaptation_lags(metrics_df, fig_dir, fmt, config)

    return results


def _phase_descriptives(events_df: pd.DataFrame, phase_boundaries: list) -> pd.DataFrame:
    """Compute descriptive stats per phase across all sessions."""
    rows = []
    for pb in phase_boundaries:
        pid = pb["id"]
        phase_df = events_df[events_df["phase_id"] == pid]
        if len(phase_df) == 0:
            continue
        row = {
            "phase_id": pid,
            "label": pb["label"],
            "n_clicks": len(phase_df),
            "n_sessions": phase_df["session_id"].nunique(),
            "mean_choice_prop_a": phase_df["choice_a"].mean(),
            "sd_choice_prop_a": phase_df.groupby("session_id")["choice_a"].mean().std(),
            "mean_reward_rate": phase_df["reward_outcome"].mean(),
            "mean_switch_rate": phase_df["switch_flag_verified"].mean()
                if "switch_flag_verified" in phase_df else np.nan,
            "mean_ici_s": phase_df["ici_s"].mean() if "ici_s" in phase_df else np.nan,
        }
        if "choice_optimal" in phase_df.columns:
            row["mean_proportion_optimal"] = phase_df["choice_optimal"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_phase_transitions(events_df: pd.DataFrame, config: dict,
                            fig_dir: str, fmt: str,
                            pre_post_tests: list | None = None):
    """Plot behavior around each phase transition."""
    phase_boundaries = config.get("phase_boundaries", [])
    pre_s = config.get("transition_pre_window_s", 15)
    post_s = config.get("transition_post_window_s", 30)

    transitions = []
    for i in range(1, len(phase_boundaries)):
        transitions.append({
            "from": phase_boundaries[i - 1],
            "to": phase_boundaries[i],
            "time_s": phase_boundaries[i]["start_ms"] / 1000,
        })

    if not transitions:
        return

    fig, axes = plt.subplots(1, len(transitions), figsize=(6 * len(transitions), 5),
                             sharey=True)
    if len(transitions) == 1:
        axes = [axes]

    for ax, tr in zip(axes, transitions):
        t_trans = tr["time_s"]
        window_df = events_df[
            (events_df["elapsed_time_s"] >= t_trans - pre_s) &
            (events_df["elapsed_time_s"] <= t_trans + post_s)
        ].copy()
        window_df["rel_time"] = window_df["elapsed_time_s"] - t_trans

        if len(window_df) == 0:
            continue

        # Bin into 1-second bins
        window_df["time_bin"] = (window_df["rel_time"] // 1).astype(int)

        for sid, sdf in window_df.groupby("session_id"):
            binned = sdf.groupby("time_bin")["choice_a"].mean()
            ax.plot(binned.index, binned.values, alpha=0.15, linewidth=0.5, color="steelblue")

        group_mean = window_df.groupby("time_bin")["choice_a"].mean()
        ax.plot(group_mean.index, group_mean.values, color="darkblue", linewidth=2)

        ax.axvline(0, color="red", linestyle="--", alpha=0.7)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("Time relative to transition (s)")
        ax.set_ylabel("P(Choose A)")
        title = f"{tr['from']['label']} → {tr['to']['label']}"
        # Add effect size annotation if available
        if pre_post_tests:
            trans_key = f"Phase {tr['from']['id']} → {tr['to']['id']}"
            for ppt in pre_post_tests:
                if ppt["transition"] == trans_key:
                    title += f"\nd = {ppt['cohens_d']:.2f}, p = {ppt['p_value']:.3f}"
                    break
        ax.set_title(title)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"phase_transitions.{fmt}"))
    plt.close(fig)


def _adaptation_lag_summary(metrics_df: pd.DataFrame) -> dict:
    """Summarize adaptation lags across sessions."""
    summary = {}
    for phase_id in [2, 3]:
        col = f"adaptation_lag_phase{phase_id}_s"
        if col in metrics_df.columns:
            vals = metrics_df[col].dropna()
            summary[f"phase{phase_id}"] = {
                "n_adapted": len(vals),
                "n_total": len(metrics_df),
                "mean_lag_s": vals.mean() if len(vals) > 0 else np.nan,
                "median_lag_s": vals.median() if len(vals) > 0 else np.nan,
                "sd_lag_s": vals.std() if len(vals) > 0 else np.nan,
            }
    return summary


def _plot_adaptation_lags(metrics_df: pd.DataFrame, fig_dir: str,
                          fmt: str, config: dict):
    """Plot adaptation lag distributions."""
    lag_cols = [c for c in metrics_df.columns if c.startswith("adaptation_lag_")]
    if not lag_cols:
        return

    fig, axes = plt.subplots(1, len(lag_cols), figsize=(5 * len(lag_cols), 4))
    if len(lag_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, lag_cols):
        vals = metrics_df[col].dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=15, color="teal", edgecolor="white", alpha=0.8)
        ax.set_xlabel("Adaptation Lag (s)")
        ax.set_ylabel("Count")
        phase_num = col.split("phase")[1].split("_")[0]
        ax.set_title(f"Phase {phase_num} Adaptation Lag")

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"adaptation_lags.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)


def _cohens_d_paired(pre, post):
    """Compute Cohen's d for paired samples: d = mean(diff) / std(diff)."""
    diff = np.array(post) - np.array(pre)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return np.nan
    return np.mean(diff) / sd


def _pre_post_comparisons(events_df: pd.DataFrame, phase_boundaries: list,
                          config: dict) -> list:
    """Paired pre-post comparisons of choice proportion at phase transitions."""
    pre_s = config.get("transition_pre_window_s", 15)
    post_s = config.get("transition_post_window_s", 30)
    results = []

    for i in range(1, len(phase_boundaries)):
        t_trans = phase_boundaries[i]["start_ms"] / 1000

        pre_data = []
        post_data = []
        for sid, sdf in events_df.groupby("session_id"):
            pre = sdf[(sdf["elapsed_time_s"] >= t_trans - pre_s) &
                       (sdf["elapsed_time_s"] < t_trans)]
            post = sdf[(sdf["elapsed_time_s"] >= t_trans) &
                        (sdf["elapsed_time_s"] < t_trans + post_s)]
            if len(pre) > 0 and len(post) > 0:
                pre_data.append(pre["choice_a"].mean())
                post_data.append(post["choice_a"].mean())

        if len(pre_data) >= 3:
            t_stat, p_val = stats.ttest_rel(pre_data, post_data)
            d = _cohens_d_paired(pre_data, post_data)
            results.append({
                "transition": f"Phase {phase_boundaries[i-1]['id']} → {phase_boundaries[i]['id']}",
                "n_sessions": len(pre_data),
                "mean_pre": np.mean(pre_data),
                "mean_post": np.mean(post_data),
                "mean_diff": np.mean(np.array(post_data) - np.array(pre_data)),
                "t_stat": t_stat,
                "p_value": p_val,
                "cohens_d": d,
            })

    return results
