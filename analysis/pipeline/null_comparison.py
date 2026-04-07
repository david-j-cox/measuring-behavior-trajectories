"""Null comparison module: generate data from known null processes and compare to real data."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


# ============================================================
# Null data generation
# ============================================================

def generate_null_data(n_sessions=20, n_clicks=300, config=None):
    """
    Generate synthetic event DataFrames from three known null processes.

    Returns a dict mapping process name to a DataFrame with the same column
    schema as the real experiment data so that existing analysis functions
    can operate on them directly.

    Processes
    ---------
    random : Independent Bernoulli(0.5) choices, Bernoulli(0.5) rewards. No dynamics.
    matching : Fixed choice probability proportional to a preset reward ratio
               (70/30), no adaptation over time.
    wsls : Win-stay/lose-shift with noise (p_stay_win=0.8, p_shift_lose=0.6).
    """
    if config is None:
        config = {}

    phase_boundaries = config.get("phase_boundaries", [
        {"id": 1, "start_ms": 0, "end_ms": 90000},
        {"id": 2, "start_ms": 90000, "end_ms": 180000},
        {"id": 3, "start_ms": 180000, "end_ms": 270000},
        {"id": 4, "start_ms": 270000, "end_ms": 360000},
    ])
    task_duration_ms = phase_boundaries[-1]["end_ms"]

    generators = {
        "random": _generate_random,
        "matching": _generate_matching,
        "wsls": _generate_wsls,
    }

    null_datasets = {}
    for name, gen_func in generators.items():
        all_sessions = []
        for i in range(n_sessions):
            session_df = gen_func(
                session_id=f"null_{name}_{i:03d}",
                n_clicks=n_clicks,
                task_duration_ms=task_duration_ms,
                phase_boundaries=phase_boundaries,
            )
            all_sessions.append(session_df)
        null_datasets[name] = pd.concat(all_sessions, ignore_index=True)

    return null_datasets


def _assign_phases(timestamps_ms, phase_boundaries):
    """Assign phase_id based on timestamp."""
    phases = np.ones(len(timestamps_ms), dtype=int)
    for pb in phase_boundaries:
        mask = (timestamps_ms >= pb["start_ms"]) & (timestamps_ms < pb["end_ms"])
        phases[mask] = pb["id"]
    return phases


def _make_session_df(session_id, choices, rewards, timestamps_ms, phase_boundaries):
    """Build a DataFrame matching the real data schema."""
    n = len(choices)
    chosen_option = np.where(choices == 1, "A", "B")

    # Compute time_since_prev_click_ms
    time_since_prev = np.zeros(n)
    time_since_prev[1:] = np.diff(timestamps_ms)

    df = pd.DataFrame({
        "session_id": session_id,
        "click_index": np.arange(n),
        "chosen_option": chosen_option,
        "reward_outcome": rewards.astype(int),
        "timestamp_ms": timestamps_ms,
        "timestamp_ms_from_task_start": timestamps_ms,
        "phase_id": _assign_phases(timestamps_ms, phase_boundaries),
        "time_since_prev_click_ms": time_since_prev,
    })

    # Aliases used by transform.py
    df["timestamp_ms"] = df["timestamp_ms_from_task_start"]

    return df


def _evenly_spaced_timestamps(n_clicks, task_duration_ms):
    """Generate roughly evenly spaced timestamps with small jitter."""
    mean_ici = task_duration_ms / n_clicks
    icis = np.random.exponential(mean_ici, size=n_clicks)
    timestamps = np.cumsum(icis)
    # Rescale to fit within task duration
    timestamps = timestamps / timestamps[-1] * (task_duration_ms - 100)
    return timestamps.astype(int)


def _generate_random(session_id, n_clicks, task_duration_ms, phase_boundaries):
    """Independent Bernoulli(0.5) choices and rewards."""
    choices = np.random.binomial(1, 0.5, size=n_clicks)
    rewards = np.random.binomial(1, 0.5, size=n_clicks)
    timestamps = _evenly_spaced_timestamps(n_clicks, task_duration_ms)
    return _make_session_df(session_id, choices, rewards, timestamps, phase_boundaries)


def _generate_matching(session_id, n_clicks, task_duration_ms, phase_boundaries,
                       p_choose_a=0.7, reward_rate_a=0.7, reward_rate_b=0.3):
    """Static matching law: fixed choice probability, independent rewards."""
    choices = np.random.binomial(1, p_choose_a, size=n_clicks)
    rewards = np.where(
        choices == 1,
        np.random.binomial(1, reward_rate_a, size=n_clicks),
        np.random.binomial(1, reward_rate_b, size=n_clicks),
    )
    timestamps = _evenly_spaced_timestamps(n_clicks, task_duration_ms)
    return _make_session_df(session_id, choices, rewards, timestamps, phase_boundaries)


def _generate_wsls(session_id, n_clicks, task_duration_ms, phase_boundaries,
                   p_stay_win=0.8, p_shift_lose=0.6, reward_rate=0.5):
    """Win-stay/lose-shift with noise."""
    choices = np.zeros(n_clicks, dtype=int)
    rewards = np.zeros(n_clicks, dtype=int)

    # First click random
    choices[0] = np.random.binomial(1, 0.5)
    rewards[0] = np.random.binomial(1, reward_rate)

    for i in range(1, n_clicks):
        if rewards[i - 1] == 1:
            # Win: stay with probability p_stay_win
            if np.random.random() < p_stay_win:
                choices[i] = choices[i - 1]
            else:
                choices[i] = 1 - choices[i - 1]
        else:
            # Lose: shift with probability p_shift_lose
            if np.random.random() < p_shift_lose:
                choices[i] = 1 - choices[i - 1]
            else:
                choices[i] = choices[i - 1]
        rewards[i] = np.random.binomial(1, reward_rate)

    timestamps = _evenly_spaced_timestamps(n_clicks, task_duration_ms)
    return _make_session_df(session_id, choices, rewards, timestamps, phase_boundaries)


# ============================================================
# Run null comparison analyses
# ============================================================

def run_null_comparison(events_df, config, output_dir):
    """
    Generate null datasets, run key analyses on them, and compare to real data.

    Parameters
    ----------
    events_df : pd.DataFrame
        Real experimental data (already transformed with derived variables).
    config : dict
        Pipeline configuration.
    output_dir : str
        Output directory for results.

    Returns
    -------
    dict with keys:
        comparison_table : pd.DataFrame  (metric x process summary)
        null_results : dict of per-process metric DataFrames
    """
    from pipeline.transform import compute_derived_variables
    from pipeline.fractal_analysis import (
        _compute_dfa, _sample_entropy,
    )
    from pipeline.dynamical_analysis import _compute_rqa_metrics

    tables_dir = os.path.join(output_dir, "tables")
    fig_dir = os.path.join(output_dir, "figures", "null_comparison")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    fmt = config.get("plot_format", "png")
    dpi = config.get("dpi", 150)

    rolling_window = config.get("rolling_window_clicks", 20)

    # --- Generate null data ---
    print("  Generating null datasets...")
    null_datasets = generate_null_data(n_sessions=20, n_clicks=300, config=config)

    # --- Transform null data (add derived variables) ---
    print("  Computing derived variables on null data...")
    for name in null_datasets:
        null_datasets[name] = compute_derived_variables(null_datasets[name], config)

    # --- Compute metrics on real data ---
    print("  Computing metrics on real data...")
    real_metrics = _compute_null_metrics(
        events_df, _compute_dfa, _sample_entropy, _compute_rqa_metrics,
        rolling_window,
    )

    # --- Compute metrics on each null process ---
    null_results = {}
    for name, null_df in null_datasets.items():
        print(f"  Computing metrics on null data: {name}...")
        null_results[name] = _compute_null_metrics(
            null_df, _compute_dfa, _sample_entropy, _compute_rqa_metrics,
            rolling_window,
        )

    # --- Build comparison table ---
    metric_names = ["dfa_alpha", "sample_entropy", "recurrence_rate",
                    "determinism", "laminarity", "trapping_time"]

    rows = []
    for metric in metric_names:
        row = {"metric": metric}

        # Real data
        real_vals = real_metrics[metric]
        row["real_mean"] = np.nanmean(real_vals) if len(real_vals) > 0 else np.nan
        row["real_sd"] = np.nanstd(real_vals) if len(real_vals) > 0 else np.nan

        # Null processes
        for proc_name in ["random", "matching", "wsls"]:
            null_vals = null_results[proc_name][metric]
            row[f"{proc_name}_mean"] = np.nanmean(null_vals) if len(null_vals) > 0 else np.nan
            row[f"{proc_name}_sd"] = np.nanstd(null_vals) if len(null_vals) > 0 else np.nan

        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(os.path.join(tables_dir, "null_comparison.csv"), index=False)
    print(f"  Saved null_comparison.csv with {len(comparison_df)} metrics.")

    # --- Generate comparison figure ---
    _plot_null_comparison(real_metrics, null_results, metric_names, fig_dir, fmt, dpi)

    return {
        "null_comparison_table": comparison_df,
        "null_results": null_results,
    }


def _compute_null_metrics(events_df, compute_dfa_fn, sample_entropy_fn,
                          compute_rqa_fn, rolling_window):
    """
    Compute key metrics per session for null comparison.

    Returns a dict mapping metric name to a list of per-session values.
    """
    metrics = {
        "dfa_alpha": [],
        "sample_entropy": [],
        "recurrence_rate": [],
        "determinism": [],
        "laminarity": [],
        "trapping_time": [],
    }

    for sid, sdf in events_df.groupby("session_id"):
        sdf = sdf.sort_values("timestamp_ms")

        # Choice sequence
        if "choice_a" not in sdf.columns:
            choices = (sdf["chosen_option"] == "A").astype(int).values
        else:
            choices = sdf["choice_a"].values

        if len(choices) < 30:
            continue

        # DFA
        _, alpha = compute_dfa_fn(choices.astype(float))
        metrics["dfa_alpha"].append(alpha)

        # Sample entropy on rolling choice proportion
        if "rolling_choice_prop_a_clicks" in sdf.columns:
            signal = sdf["rolling_choice_prop_a_clicks"].dropna().values
        else:
            signal = choices.astype(float)
        se = sample_entropy_fn(signal)
        metrics["sample_entropy"].append(se)

        # RQA on rolling choice proportion
        if "rolling_choice_prop_a_clicks" in sdf.columns:
            rqa_signal = sdf["rolling_choice_prop_a_clicks"].dropna().values
        else:
            rqa_signal = choices.astype(float)

        if len(rqa_signal) >= 20:
            rqa = compute_rqa_fn(rqa_signal, threshold=0.1)
            metrics["recurrence_rate"].append(rqa["recurrence_rate"])
            metrics["determinism"].append(rqa["determinism"])
            metrics["laminarity"].append(rqa["laminarity"])
            metrics["trapping_time"].append(rqa["trapping_time"])
        else:
            for k in ["recurrence_rate", "determinism", "laminarity", "trapping_time"]:
                metrics[k].append(np.nan)

    return metrics


def _plot_null_comparison(real_metrics, null_results, metric_names,
                          fig_dir, fmt, dpi):
    """Generate side-by-side violin/strip plots comparing real vs null distributions."""
    n_metrics = len(metric_names)
    n_cols = min(4, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).ravel()

    process_labels = ["Real", "Random", "Matching", "WSLS"]
    process_keys = [None, "random", "matching", "wsls"]
    colors = ["#2196F3", "#9E9E9E", "#FF9800", "#4CAF50"]

    for i, metric in enumerate(metric_names):
        if i >= len(axes):
            break
        ax = axes[i]

        all_vals = []
        all_labels = []
        for label, key, color in zip(process_labels, process_keys, colors):
            if key is None:
                vals = np.array(real_metrics[metric], dtype=float)
            else:
                vals = np.array(null_results[key][metric], dtype=float)
            vals = vals[~np.isnan(vals)]
            all_vals.append(vals)
            all_labels.append(label)

        # Box plots
        positions = np.arange(len(all_labels))
        bp = ax.boxplot(all_vals, positions=positions, widths=0.5,
                        patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Overlay individual points with jitter
        for j, (vals, pos) in enumerate(zip(all_vals, positions)):
            if len(vals) > 0:
                jitter = np.random.uniform(-0.15, 0.15, size=len(vals))
                ax.scatter(pos + jitter, vals, s=12, alpha=0.5,
                           color=colors[j], edgecolors="none", zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(all_labels, fontsize=9)

        # Format metric name for title
        title = metric.replace("_", " ").title()
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(metric)

    # Hide unused axes
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Null Comparison: Real Data vs Simulated Null Processes",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"null_comparison.{fmt}"),
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)
