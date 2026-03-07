"""Generate manuscript-ready primary and supplementary figures."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

_PHASE_COLORS = {1: "#999999", 2: "#4393C3", 3: "#D6604D", 4: "#5AAE61"}
_PHASE_LABELS = {1: "Symmetric", 2: "A-advantage", 3: "B-advantage", 4: "Scarcity"}


def _style():
    sns.set_theme(style="ticks", font_scale=1.15)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })


def _phase_lines(ax, config):
    for pb in config.get("phase_boundaries", []):
        if pb["start_ms"] > 0:
            ax.axvline(pb["start_ms"] / 1000, color="black", ls=":",
                       alpha=0.35, lw=0.8)


def _phase_spans(ax, config, ymin=0, ymax=1, alpha=0.06):
    for pb in config.get("phase_boundaries", []):
        c = _PHASE_COLORS.get(pb["id"], "gray")
        ax.axvspan(pb["start_ms"]/1000, pb["end_ms"]/1000,
                   ymin=ymin, ymax=ymax, color=c, alpha=alpha)


def _save(fig, path, config):
    dpi = config.get("dpi", 300)
    fig.savefig(path, dpi=dpi)
    # Also save PDF
    pdf_path = path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)


def _bootstrap_ci(data_by_bin: dict, n_boot=1000, ci=95, rng=None):
    """Bootstrap mean and CI from {bin: [values]} dict."""
    if rng is None:
        rng = np.random.default_rng(0)
    bins = sorted(data_by_bin.keys())
    means, lo, hi = [], [], []
    for b in bins:
        arr = np.array(data_by_bin[b])
        if len(arr) == 0:
            means.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            continue
        boot = np.array([rng.choice(arr, len(arr), replace=True).mean()
                         for _ in range(n_boot)])
        means.append(np.mean(arr))
        plo = (100 - ci) / 2
        lo.append(np.percentile(boot, plo))
        hi.append(np.percentile(boot, 100 - plo))
    return np.array(bins), np.array(means), np.array(lo), np.array(hi)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_figures(events_df: pd.DataFrame, metrics_df: pd.DataFrame,
                         analysis_results: dict, config: dict,
                         output_dir: str):
    """Generate all primary and supplementary figures."""
    _style()
    fig_dir = os.path.join(output_dir, "manuscript", "figures")
    sup_dir = os.path.join(output_dir, "manuscript", "supplementary")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(sup_dir, exist_ok=True)

    fmt = "png"
    n_boot = config.get("n_bootstrap", 1000)

    # Primary figures
    _figure1_task_schematic(fig_dir, fmt, config)
    _figure2_example_trajectories(events_df, fig_dir, fmt, config)
    _figure3_phase_transitions(events_df, fig_dir, fmt, config, n_boot)
    _figure4_pulse_response(events_df, fig_dir, fmt, config, n_boot)
    _figure5_state_space(events_df, fig_dir, fmt, config)
    _figure6_individual_differences(metrics_df, fig_dir, fmt, config)
    _figure7_model_comparison(analysis_results, fig_dir, fmt, config)

    # Supplementary
    _supp_run_length_distribution(events_df, sup_dir, fmt, config)
    _supp_ici_distribution(events_df, sup_dir, fmt, config)
    _supp_reward_trajectories(events_df, sup_dir, fmt, config)
    _supp_choice_autocorrelation(events_df, sup_dir, fmt, config)
    _supp_observed_vs_simulated(events_df, analysis_results, sup_dir, fmt, config)

    print(f"Figures: primary in {fig_dir}, supplementary in {sup_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 – Task Schematic
# ═══════════════════════════════════════════════════════════════════════════

def _figure1_task_schematic(fig_dir, fmt, config):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.set_aspect("equal"); ax.axis("off")

    # Patches
    for cx, label, color in [(3, "Option A", "#4393C3"), (7, "Option B", "#D6604D")]:
        circle = plt.Circle((cx, 4.5), 1.0, color=color, alpha=0.25, ec=color, lw=2)
        ax.add_patch(circle)
        ax.text(cx, 4.5, label, ha="center", va="center", fontsize=13, fontweight="bold")
        # Depletion arrow (down)
        ax.annotate("", xy=(cx, 3.2), xytext=(cx, 3.5),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
        ax.text(cx, 2.9, "Deplete\non click", ha="center", fontsize=8, color=color)
        # Recovery arrow (up)
        ax.annotate("", xy=(cx, 5.8), xytext=(cx, 5.5),
                    arrowprops=dict(arrowstyle="->", color="green", lw=2))
        ax.text(cx, 6.1, "Recover\nover time", ha="center", fontsize=8, color="green")

    # Participant
    ax.text(5, 1.0, "Participant", ha="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="#F0F0F0", ec="gray"))
    # Choice arrows
    ax.annotate("", xy=(3, 3.2), xytext=(4.3, 1.3),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--"))
    ax.annotate("", xy=(7, 3.2), xytext=(5.7, 1.3),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--"))

    # Reward
    ax.text(5, 2.2, "P(reward) = V_chosen", ha="center", fontsize=9,
            style="italic", color="gray")

    # Phase timeline at bottom
    phase_x = [(0.5, 3, "Phase 1\nSymmetric"),
               (3, 5.5, "Phase 2\nA-adv."),
               (5.5, 7.5, "Phase 3\nB-adv."),
               (7.5, 9.5, "Phase 4\nScarcity\n+ Pulses")]
    for x0, x1, label in phase_x:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, 0.0), x1 - x0, 0.5, boxstyle="round,pad=0.05",
            fc="#F5F5F5", ec="gray", lw=0.8))
        ax.text((x0 + x1) / 2, 0.25, label, ha="center", va="center", fontsize=7)

    ax.set_title("Task Structure", fontsize=14, pad=10)
    _save(fig, os.path.join(fig_dir, f"figure1_task_schematic.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 – Example Trajectories
# ═══════════════════════════════════════════════════════════════════════════

def _figure2_example_trajectories(events_df, fig_dir, fmt, config):
    session_ids = events_df["session_id"].unique()
    # Pick up to 4 diverse sessions by spread of switch rates
    per_session = events_df.groupby("session_id").agg(
        sr=("switch_flag_verified", "mean")).sort_values("sr")
    n = len(per_session)
    pick_idx = np.linspace(0, n - 1, min(4, n)).astype(int)
    selected = per_session.index[pick_idx]

    fig, axes = plt.subplots(len(selected), 1, figsize=(14, 3.5 * len(selected)),
                             sharex=True)
    if len(selected) == 1:
        axes = [axes]

    for ax, sid in zip(axes, selected):
        sdf = events_df[events_df["session_id"] == sid].sort_values("timestamp_ms")
        t = sdf["elapsed_time_s"].values

        # Choice raster (background)
        a_t = t[sdf["choice_a"].values == 1]
        b_t = t[sdf["choice_a"].values == 0]
        ax.scatter(a_t, [1.05] * len(a_t), s=2, color="#4393C3", marker="|", alpha=0.5)
        ax.scatter(b_t, [-0.05] * len(b_t), s=2, color="#D6604D", marker="|", alpha=0.5)

        # Rolling proportion
        ax.plot(t, sdf["rolling_choice_prop_a_clicks"].values,
                color="black", lw=1.3, label="P(A)")
        ax.axhline(0.5, color="gray", ls="--", alpha=0.4, lw=0.6)

        _phase_spans(ax, config)
        _phase_lines(ax, config)

        ax.set_ylim(-0.1, 1.1); ax.set_ylabel("P(Choose A)")
        sid_label = sid[:12] + "..." if len(str(sid)) > 12 else str(sid)
        ax.set_title(sid_label, fontsize=10, loc="left")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Figure 2: Example Behavioral Trajectories", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure2_example_trajectories.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 – Phase Transition Dynamics
# ═══════════════════════════════════════════════════════════════════════════

def _figure3_phase_transitions(events_df, fig_dir, fmt, config, n_boot):
    pbs = config.get("phase_boundaries", [])
    pre_s = config.get("transition_pre_window_s", 15)
    post_s = config.get("transition_post_window_s", 30)
    rng = np.random.default_rng(0)

    transitions = []
    for i in range(1, len(pbs)):
        transitions.append({
            "label": f"Phase {pbs[i-1]['id']}→{pbs[i]['id']}",
            "t_s": pbs[i]["start_ms"] / 1000,
        })

    if not transitions:
        return

    measures = [
        ("choice_a", "P(Choose A)"),
        ("switch_flag_verified", "Switch Rate"),
        ("reward_outcome", "Reward Rate"),
    ]

    fig, axes = plt.subplots(len(measures), len(transitions),
                             figsize=(5 * len(transitions), 3.5 * len(measures)),
                             sharex="col")
    if len(transitions) == 1:
        axes = axes.reshape(-1, 1)

    for col, tr in enumerate(transitions):
        t_trans = tr["t_s"]
        for row, (measure, ylabel) in enumerate(measures):
            ax = axes[row, col]
            bins_data = {}
            for sid, sdf in events_df.groupby("session_id"):
                w = sdf[(sdf["elapsed_time_s"] >= t_trans - pre_s) &
                         (sdf["elapsed_time_s"] <= t_trans + post_s)].copy()
                if len(w) < 3 or measure not in w.columns:
                    continue
                w["rel"] = w["elapsed_time_s"] - t_trans
                w["tb"] = (w["rel"] // 1).astype(int)
                for tb, grp in w.groupby("tb"):
                    bins_data.setdefault(tb, []).append(grp[measure].mean())

            x, m, lo, hi = _bootstrap_ci(bins_data, n_boot=n_boot, rng=rng)
            ax.plot(x, m, color="black", lw=1.8)
            ax.fill_between(x, lo, hi, alpha=0.15, color="steelblue")
            ax.axvline(0, color="red", ls="--", alpha=0.6, lw=0.9)
            if measure == "choice_a":
                ax.axhline(0.5, color="gray", ls="--", alpha=0.3, lw=0.6)
            ax.set_ylabel(ylabel if col == 0 else "")
            if row == 0:
                ax.set_title(tr["label"])
            if row == len(measures) - 1:
                ax.set_xlabel("Time rel. to transition (s)")

    fig.suptitle("Figure 3: Phase-Transition Dynamics", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure3_phase_transitions.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 – Pulse Response
# ═══════════════════════════════════════════════════════════════════════════

def _figure4_pulse_response(events_df, fig_dir, fmt, config, n_boot):
    bp = config.get("bonus_pulses", {})
    if not bp:
        return

    phase4_start = bp.get("phase4_start_ms", 270000)
    interval = bp.get("interval_ms", 12000)
    duration_s = bp.get("duration_ms", 3000) / 1000
    sequence = bp.get("sequence", [])
    pre_s = config.get("pulse_pre_window_s", 10)
    post_s = config.get("pulse_post_window_s", 15)
    rng = np.random.default_rng(1)

    # Collect per-target bins
    target_bins = {"A": {}, "B": {}}
    for i, target in enumerate(sequence):
        onset_s = (phase4_start + i * interval) / 1000
        for sid, sdf in events_df.groupby("session_id"):
            w = sdf[(sdf["elapsed_time_s"] >= onset_s - pre_s) &
                     (sdf["elapsed_time_s"] <= onset_s + post_s)].copy()
            if len(w) < 2:
                continue
            w["rel"] = w["elapsed_time_s"] - onset_s
            w["tb"] = (w["rel"] * 2).round() / 2
            # Recode to "chose pulsed option"
            w["chose_target"] = (w["chosen_option"] == target).astype(float)
            for tb, grp in w.groupby("tb"):
                target_bins[target].setdefault(tb, []).append(grp["chose_target"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: separate by target
    ax = axes[0]
    for target, color, ls in [("A", "#4393C3", "-"), ("B", "#D6604D", "-")]:
        if target_bins[target]:
            x, m, lo, hi = _bootstrap_ci(target_bins[target], n_boot, rng=rng)
            ax.plot(x, m, color=color, lw=2, ls=ls, label=f"{target}-pulse")
            ax.fill_between(x, lo, hi, alpha=0.12, color=color)
    ax.axvline(0, color="red", ls="--", alpha=0.5, lw=0.8)
    ax.axvline(duration_s, color="red", ls=":", alpha=0.4, lw=0.8)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.3)
    ax.set_xlabel("Time rel. to pulse onset (s)")
    ax.set_ylabel("P(Choose pulsed option)")
    ax.set_title("Choice toward pulsed option")
    ax.legend(fontsize=9)

    # Right: reward rate around pulses (pooled)
    rew_bins = {}
    for i, target in enumerate(sequence):
        onset_s = (phase4_start + i * interval) / 1000
        for sid, sdf in events_df.groupby("session_id"):
            w = sdf[(sdf["elapsed_time_s"] >= onset_s - pre_s) &
                     (sdf["elapsed_time_s"] <= onset_s + post_s)].copy()
            if len(w) < 2:
                continue
            w["rel"] = w["elapsed_time_s"] - onset_s
            w["tb"] = (w["rel"] * 2).round() / 2
            for tb, grp in w.groupby("tb"):
                rew_bins.setdefault(tb, []).append(grp["reward_outcome"].mean())

    ax = axes[1]
    if rew_bins:
        x, m, lo, hi = _bootstrap_ci(rew_bins, n_boot, rng=rng)
        ax.plot(x, m, color="darkorange", lw=2)
        ax.fill_between(x, lo, hi, alpha=0.15, color="darkorange")
    ax.axvline(0, color="red", ls="--", alpha=0.5, lw=0.8)
    ax.axvline(duration_s, color="red", ls=":", alpha=0.4, lw=0.8)
    ax.set_xlabel("Time rel. to pulse onset (s)")
    ax.set_ylabel("Reward rate")
    ax.set_title("Reward rate around pulses")

    fig.suptitle("Figure 4: Perturbation Response to Bonus Pulses", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure4_pulse_response.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 – State Space Trajectories
# ═══════════════════════════════════════════════════════════════════════════

def _figure5_state_space(events_df, fig_dir, fmt, config):
    if "rolling_choice_prop_a_clicks" not in events_df.columns:
        return

    sessions = events_df["session_id"].unique()[:6]
    n = len(sessions)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_2d(axes).ravel()

    y_col = ("rolling_reward_rate_clicks"
             if "rolling_reward_rate_clicks" in events_df.columns
             else "reward_outcome")

    for i, sid in enumerate(sessions):
        ax = axes[i]
        sdf = events_df[events_df["session_id"] == sid].sort_values("timestamp_ms")
        for pid, color in _PHASE_COLORS.items():
            pdf = sdf[sdf["phase_id"] == pid]
            if len(pdf) == 0:
                continue
            ax.plot(pdf["rolling_choice_prop_a_clicks"], pdf[y_col],
                    color=color, alpha=0.35, lw=0.4)
            ax.scatter(pdf["rolling_choice_prop_a_clicks"], pdf[y_col],
                       s=4, alpha=0.45, color=color)
        ax.set_xlim(0, 1)
        ax.set_xlabel("P(Choose A)")
        ax.set_ylabel("Reward Rate" if i % cols == 0 else "")
        sid_short = str(sid)[:10]
        ax.set_title(sid_short, fontsize=9)

    # Legend
    handles = [Line2D([0], [0], marker="o", color=c, lw=0, ms=6, label=l)
               for pid, (c, l) in
               {1: ("#999999", "Symmetric"), 2: ("#4393C3", "A-adv"),
                3: ("#D6604D", "B-adv"), 4: ("#5AAE61", "Scarcity")}.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)

    for j in range(len(sessions), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Figure 5: State-Space Trajectories", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure5_state_space.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6 – Individual Differences
# ═══════════════════════════════════════════════════════════════════════════

def _figure6_individual_differences(metrics_df, fig_dir, fmt, config):
    panels = [
        ("overall_switch_rate", "Switch Rate"),
        ("mean_run_length", "Mean Run Length"),
        ("choice_entropy", "Choice Entropy (bits)"),
    ]
    if "proportion_optimal" in metrics_df.columns:
        panels.append(("proportion_optimal", "Proportion Optimal"))
    if "adaptation_lag_phase2_s" in metrics_df.columns:
        panels.append(("adaptation_lag_phase2_s", "Adaptation Lag Phase 2 (s)"))

    n_panels = len(panels)
    cols = min(3, n_panels)
    rows = int(np.ceil(n_panels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    axes = np.atleast_2d(axes).ravel()

    for i, (col, label) in enumerate(panels):
        ax = axes[i]
        vals = metrics_df[col].dropna()
        ax.hist(vals, bins=min(15, max(5, len(vals) // 2)),
                color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(vals.mean(), color="red", ls="--", lw=1.2)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")

    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Figure 6: Individual Differences in Strategy", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure6_individual_differences.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7 – Model Comparison
# ═══════════════════════════════════════════════════════════════════════════

def _figure7_model_comparison(analysis_results, fig_dir, fmt, config):
    mc = analysis_results.get("model_comparison")
    if mc is None or len(mc) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: mean BIC bar chart
    ax = axes[0]
    agg = mc.groupby("model")["bic"].agg(["mean", "sem"]).sort_values("mean")
    colors = sns.color_palette("Set2", len(agg))
    bars = ax.barh(range(len(agg)), agg["mean"], xerr=agg["sem"],
                   color=colors, edgecolor="white")
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels(agg.index, fontsize=9)
    ax.set_xlabel("Mean BIC")
    ax.set_title("Mean BIC by Model")

    # Right: delta-BIC heatmap
    ax = axes[1]
    bic_pivot = mc.pivot(index="session_id", columns="model", values="bic")
    delta = bic_pivot.sub(bic_pivot.min(axis=1), axis=0)
    # Order columns by mean
    col_order = delta.mean().sort_values().index
    delta = delta[col_order]
    sns.heatmap(delta, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "ΔBIC"})
    ax.set_title("ΔBIC from Best (per session)")
    ax.set_ylabel("Session")

    fig.suptitle("Figure 7: Model Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure7_model_comparison.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Supplementary Figures
# ═══════════════════════════════════════════════════════════════════════════

def _supp_run_length_distribution(events_df, sup_dir, fmt, config):
    fig, ax = plt.subplots(figsize=(8, 5))
    all_runs = events_df.groupby("session_id")["run_length_current"].apply(list)
    pooled = [r for runs in all_runs for r in runs]
    ax.hist(pooled, bins=np.arange(1, min(max(pooled) + 1, 50)),
            density=True, color="teal", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Run Length (clicks)")
    ax.set_ylabel("Density")
    ax.set_title("Supplementary: Run-Length Distribution (all sessions pooled)")
    _save(fig, os.path.join(sup_dir, f"supp_run_length_dist.{fmt}"), config)


def _supp_ici_distribution(events_df, sup_dir, fmt, config):
    fig, ax = plt.subplots(figsize=(8, 5))
    ici = events_df["ici_s"].dropna()
    ici_clipped = ici[ici < ici.quantile(0.99)]  # clip extreme outliers
    ax.hist(ici_clipped, bins=50, density=True, color="slategray",
            edgecolor="white", alpha=0.8)
    ax.set_xlabel("Inter-Click Interval (s)")
    ax.set_ylabel("Density")
    ax.set_title("Supplementary: ICI Distribution (99th-percentile clipped)")
    ax.axvline(ici.median(), color="red", ls="--", lw=1, label=f"Median = {ici.median():.2f}s")
    ax.legend()
    _save(fig, os.path.join(sup_dir, f"supp_ici_dist.{fmt}"), config)


def _supp_reward_trajectories(events_df, sup_dir, fmt, config):
    if "rolling_reward_rate_clicks" not in events_df.columns:
        return
    edf = events_df.copy()
    edf["time_bin"] = (edf["elapsed_time_s"] // 1).astype(int)

    fig, ax = plt.subplots(figsize=(14, 5))
    for sid, sdf in edf.groupby("session_id"):
        binned = sdf.groupby("time_bin")["rolling_reward_rate_clicks"].mean()
        ax.plot(binned.index, binned.values, alpha=0.2, lw=0.5, color="darkorange")
    group = edf.groupby("time_bin")["rolling_reward_rate_clicks"].mean()
    ax.plot(group.index, group.values, color="darkred", lw=2.5, label="Group mean")
    _phase_lines(ax, config)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Rolling Reward Rate")
    ax.set_title("Supplementary: Reward-Rate Trajectories")
    ax.legend()
    _save(fig, os.path.join(sup_dir, f"supp_reward_trajectories.{fmt}"), config)


def _supp_choice_autocorrelation(events_df, sup_dir, fmt, config):
    max_lag = 40
    all_acfs = {}
    for sid, sdf in events_df.groupby("session_id"):
        c = sdf["choice_a"].values
        if len(c) < max_lag + 10:
            continue
        x = c - c.mean()
        v = np.var(x)
        if v == 0:
            continue
        for lag in range(1, max_lag + 1):
            acf = np.mean(x[lag:] * x[:-lag]) / v
            all_acfs.setdefault(lag, []).append(acf)

    if not all_acfs:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    lags = sorted(all_acfs.keys())
    means = [np.mean(all_acfs[l]) for l in lags]
    sems = [np.std(all_acfs[l]) / np.sqrt(len(all_acfs[l])) for l in lags]
    ax.plot(lags, means, color="darkblue", lw=2)
    ax.fill_between(lags, np.array(means) - np.array(sems),
                    np.array(means) + np.array(sems), alpha=0.2, color="steelblue")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Lag (clicks)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Supplementary: Choice Autocorrelation Function")
    _save(fig, os.path.join(sup_dir, f"supp_choice_acf.{fmt}"), config)


def _supp_observed_vs_simulated(events_df, analysis_results, sup_dir, fmt, config):
    """Observed vs model-simulated rolling choice proportion for each session."""
    mc = analysis_results.get("model_comparison")
    if mc is None or len(mc) == 0:
        return

    # Use forgetting Q-learning params if available
    rl_fits = mc[mc["model"] == "q_forgetting"]
    if len(rl_fits) == 0:
        rl_fits = mc[mc["model"] == "q_learning"]
    if len(rl_fits) == 0:
        return

    sessions = rl_fits["session_id"].unique()[:4]
    fig, axes = plt.subplots(len(sessions), 1, figsize=(14, 3.5 * len(sessions)),
                             sharex=True)
    if len(sessions) == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    window = 20

    for ax, sid in zip(axes, sessions):
        sdf = events_df[events_df["session_id"] == sid].sort_values("timestamp_ms")
        params = rl_fits[rl_fits["session_id"] == sid].iloc[0]
        alpha = params.get("alpha", 0.1)
        beta = params.get("beta", 3.0)
        forget = params.get("forget", 0.0)

        va = sdf["latent_value_a_pre"].values if "latent_value_a_pre" in sdf.columns else np.full(len(sdf), 0.5)
        vb = sdf["latent_value_b_pre"].values if "latent_value_b_pre" in sdf.columns else np.full(len(sdf), 0.5)
        elapsed = sdf["elapsed_time_s"].values

        # Simulate 30 runs
        for _ in range(30):
            QA, QB = 0.5, 0.5
            sim_choices = np.zeros(len(sdf))
            for t in range(len(sdf)):
                dv = np.clip(beta * (QA - QB), -500, 500)
                p_a = 1.0 / (1.0 + np.exp(-dv))
                chose_a = int(rng.random() < p_a)
                sim_choices[t] = chose_a
                v_chosen = va[t] if chose_a else vb[t]
                rew = int(rng.random() < min(1.0, v_chosen))
                if chose_a:
                    QA += alpha * (rew - QA)
                    QB += forget * (0.5 - QB)
                else:
                    QB += alpha * (rew - QB)
                    QA += forget * (0.5 - QA)
            sim_roll = pd.Series(sim_choices).rolling(window, min_periods=1).mean().values
            ax.plot(elapsed, sim_roll, alpha=0.06, color="coral", lw=0.5)

        # Observed
        ax.plot(elapsed, sdf["rolling_choice_prop_a_clicks"].values,
                color="darkblue", lw=1.8, label="Observed")
        _phase_lines(ax, config)
        ax.axhline(0.5, color="gray", ls="--", alpha=0.3)
        ax.set_ylim(0, 1); ax.set_ylabel("P(A)")
        sid_short = str(sid)[:12]
        ax.set_title(f"{sid_short}", fontsize=9, loc="left")

    axes[-1].set_xlabel("Time (s)")
    # Custom legend
    obs_line = Line2D([0], [0], color="darkblue", lw=2, label="Observed")
    sim_line = Line2D([0], [0], color="coral", lw=1, alpha=0.5, label="Simulated (n=30)")
    fig.legend(handles=[obs_line, sim_line], loc="upper right", fontsize=9)
    fig.suptitle("Supplementary: Observed vs Model-Simulated Trajectories",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, os.path.join(sup_dir, f"supp_obs_vs_sim.{fmt}"), config)
