"""Plotting functions for individual and group-level visualizations."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Optional


def set_style():
    """Set consistent plot style."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })


def plot_individual_session(sdf: pd.DataFrame, session_id: str,
                            config: dict, output_dir: str):
    """Multi-panel figure for a single session."""
    set_style()
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(5, 2, hspace=0.4, wspace=0.3)

    phase_boundaries = config.get("phase_boundaries", [])
    fmt = config.get("plot_format", "png")

    # Panel 1: Choice sequence (raster)
    ax1 = fig.add_subplot(gs[0, :])
    _plot_choice_raster(ax1, sdf, phase_boundaries)
    ax1.set_title(f"Session {session_id}: Choice Sequence")

    # Panel 2: Rolling choice proportion
    ax2 = fig.add_subplot(gs[1, :])
    _plot_rolling_prop(ax2, sdf, phase_boundaries)

    # Panel 3: Cumulative score
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(sdf["elapsed_time_s"], sdf["cumulative_score"], color="steelblue")
    _add_phase_lines(ax3, phase_boundaries)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Cumulative Score")
    ax3.set_title("Score Trajectory")

    # Panel 4: Rolling reward rate
    ax4 = fig.add_subplot(gs[2, 1])
    if "rolling_reward_rate_clicks" in sdf.columns:
        ax4.plot(sdf["elapsed_time_s"], sdf["rolling_reward_rate_clicks"],
                 color="darkorange", alpha=0.8)
    _add_phase_lines(ax4, phase_boundaries)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Rolling Reward Rate")
    ax4.set_title("Local Reward Rate (20-click window)")

    # Panel 5: ICI over time
    ax5 = fig.add_subplot(gs[3, 0])
    if "ici_s" in sdf.columns:
        ax5.scatter(sdf["elapsed_time_s"], sdf["ici_s"], s=3, alpha=0.4, color="gray")
        if "rolling_mean_ici_clicks" in sdf.columns:
            ax5.plot(sdf["elapsed_time_s"], sdf["rolling_mean_ici_clicks"],
                     color="crimson", linewidth=1.5)
    _add_phase_lines(ax5, phase_boundaries)
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("ICI (s)")
    ax5.set_title("Inter-Click Interval")

    # Panel 6: Run lengths
    ax6 = fig.add_subplot(gs[3, 1])
    if "run_length_current" in sdf.columns:
        ax6.scatter(sdf["elapsed_time_s"], sdf["run_length_current"],
                    s=5, alpha=0.4, color="teal")
    _add_phase_lines(ax6, phase_boundaries)
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Run Length")
    ax6.set_title("Consecutive Same-Option Runs")

    # Panel 7: Latent values (if available)
    ax7 = fig.add_subplot(gs[4, 0])
    if "latent_value_a_pre" in sdf.columns:
        ax7.plot(sdf["elapsed_time_s"], sdf["latent_value_a_pre"],
                 label="V_A", color="dodgerblue", alpha=0.7)
        ax7.plot(sdf["elapsed_time_s"], sdf["latent_value_b_pre"],
                 label="V_B", color="orangered", alpha=0.7)
        ax7.legend(fontsize=8)
    _add_phase_lines(ax7, phase_boundaries)
    ax7.set_xlabel("Time (s)")
    ax7.set_ylabel("Latent Value")
    ax7.set_title("Hidden State Dynamics")

    # Panel 8: Local reward rate by option
    ax8 = fig.add_subplot(gs[4, 1])
    if "local_reward_rate_a" in sdf.columns:
        ax8.plot(sdf["elapsed_time_s"], sdf["local_reward_rate_a"],
                 label="Reward Rate A", color="dodgerblue", alpha=0.7)
        ax8.plot(sdf["elapsed_time_s"], sdf["local_reward_rate_b"],
                 label="Reward Rate B", color="orangered", alpha=0.7)
        ax8.legend(fontsize=8)
    _add_phase_lines(ax8, phase_boundaries)
    ax8.set_xlabel("Time (s)")
    ax8.set_ylabel("Local Reward Rate")
    ax8.set_title("Experienced Reward Rate by Option")

    fig_path = os.path.join(output_dir, "figures", "individual", f"session_{session_id}.{fmt}")
    fig.savefig(fig_path, dpi=config.get("dpi", 150))
    plt.close(fig)


def plot_group_summary(metrics_df: pd.DataFrame, config: dict, output_dir: str):
    """Group-level summary plots from session metrics."""
    set_style()
    fmt = config.get("plot_format", "png")
    fig_dir = os.path.join(output_dir, "figures", "group")

    # Distribution of key metrics
    metric_cols = [
        ("total_clicks", "Total Clicks"),
        ("session_duration_s", "Session Duration (s)"),
        ("overall_reward_rate", "Overall Reward Rate"),
        ("overall_switch_rate", "Overall Switch Rate"),
        ("mean_run_length", "Mean Run Length"),
        ("choice_entropy", "Choice Entropy"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for i, (col, label) in enumerate(metric_cols):
        if col in metrics_df.columns and i < len(axes):
            axes[i].hist(metrics_df[col].dropna(), bins=15, color="steelblue",
                         edgecolor="white", alpha=0.8)
            axes[i].set_xlabel(label)
            axes[i].set_ylabel("Count")
            axes[i].set_title(label)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"metric_distributions.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)

    # Correlation matrix
    corr_cols = [c for c, _ in metric_cols if c in metrics_df.columns]
    if len(corr_cols) > 2:
        fig, ax = plt.subplots(figsize=(8, 7))
        corr = metrics_df[corr_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    ax=ax, square=True)
        ax.set_title("Session Metric Correlations")
        fig.savefig(os.path.join(fig_dir, f"metric_correlations.{fmt}"),
                    dpi=config.get("dpi", 150))
        plt.close(fig)

    # Win-stay / lose-shift scatter
    if "win_stay_rate" in metrics_df.columns and "lose_shift_rate" in metrics_df.columns:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(metrics_df["win_stay_rate"], metrics_df["lose_shift_rate"],
                   s=40, alpha=0.6, color="teal")
        ax.set_xlabel("Win-Stay Rate")
        ax.set_ylabel("Lose-Shift Rate")
        ax.set_title("Win-Stay vs Lose-Shift")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
        fig.savefig(os.path.join(fig_dir, f"wslf_scatter.{fmt}"),
                    dpi=config.get("dpi", 150))
        plt.close(fig)


def plot_group_trajectories(events_df: pd.DataFrame, config: dict, output_dir: str):
    """Overlay and averaged trajectory plots across participants."""
    set_style()
    fmt = config.get("plot_format", "png")
    fig_dir = os.path.join(output_dir, "figures", "group")
    phase_boundaries = config.get("phase_boundaries", [])

    if "rolling_choice_prop_a_clicks" not in events_df.columns:
        return

    # Bin time into 1-second bins and compute group average
    events_df = events_df.copy()
    events_df["time_bin"] = (events_df["elapsed_time_s"] // 1).astype(int)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Individual traces (thin, transparent)
    for sid, sdf in events_df.groupby("session_id"):
        binned = sdf.groupby("time_bin")["rolling_choice_prop_a_clicks"].mean()
        ax.plot(binned.index, binned.values, alpha=0.15, linewidth=0.5, color="steelblue")

    # Group mean
    group_mean = events_df.groupby("time_bin")["rolling_choice_prop_a_clicks"].mean()
    ax.plot(group_mean.index, group_mean.values, color="darkblue", linewidth=2, label="Group Mean")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    _add_phase_lines(ax, phase_boundaries)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("P(Choose A)")
    ax.set_title("Choice Proportion Trajectories")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.savefig(os.path.join(fig_dir, f"choice_trajectories.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)

    # Reward rate trajectories
    if "rolling_reward_rate_clicks" in events_df.columns:
        fig, ax = plt.subplots(figsize=(14, 5))
        for sid, sdf in events_df.groupby("session_id"):
            binned = sdf.groupby("time_bin")["rolling_reward_rate_clicks"].mean()
            ax.plot(binned.index, binned.values, alpha=0.15, linewidth=0.5, color="darkorange")
        group_mean = events_df.groupby("time_bin")["rolling_reward_rate_clicks"].mean()
        ax.plot(group_mean.index, group_mean.values, color="darkred", linewidth=2, label="Group Mean")
        _add_phase_lines(ax, phase_boundaries)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Rolling Reward Rate")
        ax.set_title("Reward Rate Trajectories")
        ax.legend()
        fig.savefig(os.path.join(fig_dir, f"reward_trajectories.{fmt}"),
                    dpi=config.get("dpi", 150))
        plt.close(fig)


# ---- Helpers ----

def _plot_choice_raster(ax, sdf, phase_boundaries):
    """Raster plot of choices over time."""
    a_clicks = sdf[sdf["choice_a"] == 1]
    b_clicks = sdf[sdf["choice_b"] == 1]
    ax.scatter(a_clicks["elapsed_time_s"], [1] * len(a_clicks), s=2,
               color="dodgerblue", label="A", marker="|")
    ax.scatter(b_clicks["elapsed_time_s"], [0] * len(b_clicks), s=2,
               color="orangered", label="B", marker="|")
    _add_phase_lines(ax, phase_boundaries)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["B", "A"])
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8, loc="upper right")


def _plot_rolling_prop(ax, sdf, phase_boundaries):
    """Rolling choice proportion plot."""
    if "rolling_choice_prop_a_clicks" in sdf.columns:
        ax.plot(sdf["elapsed_time_s"], sdf["rolling_choice_prop_a_clicks"],
                color="steelblue", linewidth=1, label="Click-based")
    if "rolling_choice_prop_a_time" in sdf.columns:
        ax.plot(sdf["elapsed_time_s"], sdf["rolling_choice_prop_a_time"],
                color="coral", linewidth=1, alpha=0.7, label="Time-based")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    _add_phase_lines(ax, phase_boundaries)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("P(Choose A)")
    ax.set_title("Rolling Choice Proportion")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)


def _add_phase_lines(ax, phase_boundaries):
    """Add vertical lines at phase transitions."""
    for pb in phase_boundaries:
        if pb["start_ms"] > 0:
            ax.axvline(pb["start_ms"] / 1000, color="black", linestyle=":",
                       alpha=0.4, linewidth=0.8)
