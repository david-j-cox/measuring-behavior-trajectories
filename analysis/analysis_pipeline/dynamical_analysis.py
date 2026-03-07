"""Dynamical systems analysis: state-space trajectories, hysteresis, autocorrelation."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import correlate


def run_dynamical_analysis(events_df: pd.DataFrame, config: dict,
                           output_dir: str) -> dict:
    """Run dynamical systems analyses."""
    results = {}
    fmt = config.get("plot_format", "png")
    fig_dir = os.path.join(output_dir, "figures", "dynamical")

    # State-space trajectories
    _plot_state_space(events_df, config, fig_dir, fmt)

    # Choice autocorrelation
    acf_results = _compute_choice_autocorrelation(events_df, max_lag=50)
    results["choice_autocorrelation"] = acf_results
    _plot_autocorrelation(acf_results, fig_dir, fmt, config)

    # Hysteresis analysis
    hyst_results = _compute_hysteresis(events_df, config)
    results["hysteresis"] = hyst_results
    if hyst_results:
        _plot_hysteresis(hyst_results, fig_dir, fmt, config)

    # Recurrence analysis (simplified)
    _plot_recurrence(events_df, config, fig_dir, fmt)

    return results


def _plot_state_space(events_df: pd.DataFrame, config: dict,
                      fig_dir: str, fmt: str):
    """Plot 2D state-space trajectory (choice prop vs reward rate)."""
    if "rolling_choice_prop_a_clicks" not in events_df.columns:
        return
    if "rolling_reward_rate_clicks" not in events_df.columns:
        return

    phase_colors = {1: "gray", 2: "dodgerblue", 3: "orangered", 4: "green"}

    # Plot a few example sessions
    session_ids = events_df["session_id"].unique()[:6]

    n_cols = min(3, len(session_ids))
    n_rows = int(np.ceil(len(session_ids) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for i, sid in enumerate(session_ids):
        if i >= len(axes):
            break
        ax = axes[i]
        sdf = events_df[events_df["session_id"] == sid]

        for pid, color in phase_colors.items():
            phase_df = sdf[sdf["phase_id"] == pid]
            if len(phase_df) > 0:
                ax.scatter(phase_df["rolling_choice_prop_a_clicks"],
                           phase_df["rolling_reward_rate_clicks"],
                           s=3, alpha=0.5, color=color, label=f"Phase {pid}")
                # Draw trajectory line
                ax.plot(phase_df["rolling_choice_prop_a_clicks"],
                        phase_df["rolling_reward_rate_clicks"],
                        alpha=0.2, linewidth=0.5, color=color)

        ax.set_xlabel("P(Choose A)")
        ax.set_ylabel("Reward Rate")
        ax.set_title(f"Session {sid[:8]}..." if len(str(sid)) > 8 else f"Session {sid}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Add legend to first axis
    if len(axes) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=8)

    for j in range(len(session_ids), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"state_space_trajectories.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)


def _compute_choice_autocorrelation(events_df: pd.DataFrame,
                                     max_lag: int = 50) -> pd.DataFrame:
    """Compute autocorrelation of choice sequence per session."""
    rows = []
    for sid, sdf in events_df.groupby("session_id"):
        choices = sdf["choice_a"].values
        if len(choices) < max_lag + 10:
            continue

        # Center the series
        x = choices - choices.mean()
        var = np.var(x)
        if var == 0:
            continue

        for lag in range(1, max_lag + 1):
            acf = np.mean(x[lag:] * x[:-lag]) / var
            rows.append({"session_id": sid, "lag": lag, "acf": acf})

    return pd.DataFrame(rows)


def _plot_autocorrelation(acf_df: pd.DataFrame, fig_dir: str,
                          fmt: str, config: dict):
    """Plot group-averaged autocorrelation function."""
    if len(acf_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    # Individual traces
    for sid, sdf in acf_df.groupby("session_id"):
        ax.plot(sdf["lag"], sdf["acf"], alpha=0.1, linewidth=0.5, color="steelblue")

    # Group mean
    group = acf_df.groupby("lag")["acf"].agg(["mean", "sem"]).reset_index()
    ax.plot(group["lag"], group["mean"], color="darkblue", linewidth=2, label="Group mean")
    ax.fill_between(group["lag"],
                    group["mean"] - group["sem"],
                    group["mean"] + group["sem"],
                    alpha=0.2, color="steelblue")

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Lag (clicks)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Choice Autocorrelation Function")
    ax.legend()

    fig.savefig(os.path.join(fig_dir, f"choice_autocorrelation.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)


def _compute_hysteresis(events_df: pd.DataFrame, config: dict) -> dict:
    """
    Detect hysteresis: does the path through state space differ
    depending on direction of environmental change?
    Compare Phase 2 (A-advantage rising) vs Phase 3 (B-advantage / A-declining).
    """
    if "rolling_choice_prop_a_clicks" not in events_df.columns:
        return {}
    if "latent_advantage_pre" not in events_df.columns:
        return {}

    phase2 = events_df[events_df["phase_id"] == 2].copy()
    phase3 = events_df[events_df["phase_id"] == 3].copy()

    if len(phase2) == 0 or len(phase3) == 0:
        return {}

    # Bin latent advantage and compute mean choice prop
    def bin_advantage(df, n_bins=10):
        df = df.copy()
        df["adv_bin"] = pd.cut(df["latent_advantage_pre"], bins=n_bins, labels=False)
        return df.groupby("adv_bin").agg(
            mean_advantage=("latent_advantage_pre", "mean"),
            mean_choice_a=("rolling_choice_prop_a_clicks", "mean"),
            n=("choice_a", "count"),
        ).reset_index()

    return {
        "phase2_trajectory": bin_advantage(phase2),
        "phase3_trajectory": bin_advantage(phase3),
    }


def _plot_hysteresis(hyst_data: dict, fig_dir: str, fmt: str, config: dict):
    """Plot hysteresis loop: choice proportion vs latent advantage."""
    p2 = hyst_data.get("phase2_trajectory")
    p3 = hyst_data.get("phase3_trajectory")
    if p2 is None or p3 is None or len(p2) == 0 or len(p3) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(p2["mean_advantage"], p2["mean_choice_a"],
            "o-", color="dodgerblue", label="Phase 2 (A rising)", markersize=5)
    ax.plot(p3["mean_advantage"], p3["mean_choice_a"],
            "s-", color="orangered", label="Phase 3 (B rising)", markersize=5)

    ax.set_xlabel("Latent Advantage (V_A - V_B)")
    ax.set_ylabel("P(Choose A)")
    ax.set_title("Hysteresis: Choice vs Latent Advantage")
    ax.legend()
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.4)

    fig.savefig(os.path.join(fig_dir, f"hysteresis.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)


def _plot_recurrence(events_df: pd.DataFrame, config: dict,
                     fig_dir: str, fmt: str):
    """Simplified recurrence plot for an example session."""
    if "rolling_choice_prop_a_clicks" not in events_df.columns:
        return

    # Pick first session
    sid = events_df["session_id"].unique()[0]
    sdf = events_df[events_df["session_id"] == sid]

    signal = sdf["rolling_choice_prop_a_clicks"].values
    if len(signal) < 20:
        return

    # Subsample if too long
    if len(signal) > 300:
        step = len(signal) // 300
        signal = signal[::step]

    n = len(signal)
    threshold = 0.1  # similarity threshold

    # Build recurrence matrix
    dist = np.abs(signal[:, None] - signal[None, :])
    recurrence = (dist < threshold).astype(int)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(recurrence, cmap="Greys", origin="lower", aspect="equal")
    ax.set_xlabel("Click index")
    ax.set_ylabel("Click index")
    ax.set_title(f"Recurrence Plot (Session {sid[:8] if len(str(sid)) > 8 else sid})")

    fig.savefig(os.path.join(fig_dir, f"recurrence_example.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)
