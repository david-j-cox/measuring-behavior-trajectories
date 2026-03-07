"""Bonus pulse (perturbation) analysis: pulse-triggered averaging and response dynamics."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_perturbation_analysis(events_df: pd.DataFrame, config: dict,
                              output_dir: str) -> dict:
    """Analyze behavioral responses to Phase 4 bonus pulses."""
    results = {}
    pulse_schedule = _build_pulse_schedule(config)
    if not pulse_schedule:
        return results

    fmt = config.get("plot_format", "png")
    fig_dir = os.path.join(output_dir, "figures", "pulse")
    pre_s = config.get("pulse_pre_window_s", 10)
    post_s = config.get("pulse_post_window_s", 15)

    # Pulse-triggered averages
    pta_data = _compute_pulse_triggered_averages(
        events_df, pulse_schedule, pre_s, post_s
    )
    results["pulse_triggered_averages"] = pta_data

    # Plots
    if pta_data:
        _plot_pulse_triggered_averages(pta_data, fig_dir, fmt, config)
        _plot_pulse_response_by_target(pta_data, fig_dir, fmt, config)

    return results


def _build_pulse_schedule(config: dict) -> list:
    """Build list of pulse events with absolute times."""
    bp = config.get("bonus_pulses", {})
    if not bp:
        return []

    phase4_start = bp.get("phase4_start_ms", 270000)
    interval = bp.get("interval_ms", 12000)
    duration = bp.get("duration_ms", 3000)
    sequence = bp.get("sequence", [])

    pulses = []
    for i, target in enumerate(sequence):
        onset_ms = phase4_start + i * interval
        offset_ms = onset_ms + duration
        pulses.append({
            "index": i,
            "target": target,
            "onset_ms": onset_ms,
            "offset_ms": offset_ms,
            "onset_s": onset_ms / 1000,
            "offset_s": offset_ms / 1000,
        })
    return pulses


def _compute_pulse_triggered_averages(events_df: pd.DataFrame, pulse_schedule: list,
                                       pre_s: float, post_s: float) -> dict:
    """Compute pulse-triggered averages for choice and reward rate."""
    all_traces = {"choice_a": [], "reward": [], "ici": []}
    pulse_meta = []

    for pulse in pulse_schedule:
        onset_s = pulse["onset_s"]

        for sid, sdf in events_df.groupby("session_id"):
            window = sdf[
                (sdf["elapsed_time_s"] >= onset_s - pre_s) &
                (sdf["elapsed_time_s"] <= onset_s + post_s)
            ].copy()

            if len(window) < 3:
                continue

            window["rel_time"] = window["elapsed_time_s"] - onset_s
            window["time_bin"] = (window["rel_time"] * 2).round() / 2  # 0.5s bins

            binned = window.groupby("time_bin").agg({
                "choice_a": "mean",
                "reward_outcome": "mean",
            }).reset_index()

            if "ici_s" in window.columns:
                ici_binned = window.groupby("time_bin")["ici_s"].mean().reset_index()
                binned = binned.merge(ici_binned, on="time_bin", how="left")

            for _, row in binned.iterrows():
                all_traces["choice_a"].append({
                    "session_id": sid,
                    "pulse_index": pulse["index"],
                    "pulse_target": pulse["target"],
                    "time_bin": row["time_bin"],
                    "value": row["choice_a"],
                })
                all_traces["reward"].append({
                    "session_id": sid,
                    "pulse_index": pulse["index"],
                    "pulse_target": pulse["target"],
                    "time_bin": row["time_bin"],
                    "value": row["reward_outcome"],
                })
                if "ici_s" in row.index:
                    all_traces["ici"].append({
                        "session_id": sid,
                        "pulse_index": pulse["index"],
                        "pulse_target": pulse["target"],
                        "time_bin": row["time_bin"],
                        "value": row["ici_s"],
                    })

            pulse_meta.append({
                "session_id": sid,
                "pulse_index": pulse["index"],
                "pulse_target": pulse["target"],
                "n_clicks_window": len(window),
            })

    result = {
        "choice_a": pd.DataFrame(all_traces["choice_a"]) if all_traces["choice_a"] else pd.DataFrame(),
        "reward": pd.DataFrame(all_traces["reward"]) if all_traces["reward"] else pd.DataFrame(),
        "ici": pd.DataFrame(all_traces["ici"]) if all_traces["ici"] else pd.DataFrame(),
        "pulse_meta": pd.DataFrame(pulse_meta) if pulse_meta else pd.DataFrame(),
    }
    return result


def _plot_pulse_triggered_averages(pta_data: dict, fig_dir: str,
                                    fmt: str, config: dict):
    """Plot overall pulse-triggered average of choice proportion."""
    choice_df = pta_data.get("choice_a")
    if choice_df is None or len(choice_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    group_mean = choice_df.groupby("time_bin")["value"].agg(["mean", "sem"]).reset_index()
    ax.plot(group_mean["time_bin"], group_mean["mean"], color="steelblue", linewidth=2)
    ax.fill_between(group_mean["time_bin"],
                    group_mean["mean"] - group_mean["sem"],
                    group_mean["mean"] + group_mean["sem"],
                    alpha=0.2, color="steelblue")

    ax.axvline(0, color="red", linestyle="--", alpha=0.7, label="Pulse onset")
    duration_s = config.get("bonus_pulses", {}).get("duration_ms", 3000) / 1000
    ax.axvline(duration_s, color="red", linestyle=":", alpha=0.5, label="Pulse offset")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)

    ax.set_xlabel("Time relative to pulse onset (s)")
    ax.set_ylabel("P(Choose A)")
    ax.set_title("Pulse-Triggered Average: Choice Proportion")
    ax.legend()
    fig.savefig(os.path.join(fig_dir, f"pta_choice_overall.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)


def _plot_pulse_response_by_target(pta_data: dict, fig_dir: str,
                                    fmt: str, config: dict):
    """Plot pulse-triggered averages split by pulse target (A vs B)."""
    choice_df = pta_data.get("choice_a")
    if choice_df is None or len(choice_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    duration_s = config.get("bonus_pulses", {}).get("duration_ms", 3000) / 1000

    for target, color, label in [("A", "dodgerblue", "A-pulse"), ("B", "orangered", "B-pulse")]:
        tdf = choice_df[choice_df["pulse_target"] == target]
        if len(tdf) == 0:
            continue
        group = tdf.groupby("time_bin")["value"].agg(["mean", "sem"]).reset_index()
        ax.plot(group["time_bin"], group["mean"], color=color, linewidth=2, label=label)
        ax.fill_between(group["time_bin"],
                        group["mean"] - group["sem"],
                        group["mean"] + group["sem"],
                        alpha=0.15, color=color)

    ax.axvline(0, color="red", linestyle="--", alpha=0.7)
    ax.axvline(duration_s, color="red", linestyle=":", alpha=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Time relative to pulse onset (s)")
    ax.set_ylabel("P(Choose A)")
    ax.set_title("Pulse-Triggered Average by Target")
    ax.legend()
    fig.savefig(os.path.join(fig_dir, f"pta_choice_by_target.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)
