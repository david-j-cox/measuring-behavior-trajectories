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
    """Phase-transition dynamics: individual traces + group mean ± bootstrap CI.

    Each column is one transition (e.g., Symmetric→A-advantage).
    Each row is one dependent variable (choice, switching, reward).
    Thin colored lines = individual participants; bold black = group mean.
    """
    pbs = config.get("phase_boundaries", [])
    pre_s = config.get("transition_pre_window_s", 15)
    post_s = config.get("transition_post_window_s", 30)
    rng = np.random.default_rng(0)

    # Transition descriptors with human-readable annotations
    transition_meta = {
        (1, 2): {"label": "Symmetric → A-advantage",
                 "note": "A becomes higher-value"},
        (2, 3): {"label": "A-advantage → B-advantage",
                 "note": "Optimal switches from A to B"},
        (3, 4): {"label": "B-advantage → Scarcity + Pulses",
                 "note": "Both options deplete; pulses begin"},
    }

    transitions = []
    for i in range(1, len(pbs)):
        key = (pbs[i-1]["id"], pbs[i]["id"])
        meta = transition_meta.get(key, {})
        transitions.append({
            "label": meta.get("label", f"Phase {key[0]}→{key[1]}"),
            "note": meta.get("note", ""),
            "t_s": pbs[i]["start_ms"] / 1000,
        })

    if not transitions:
        return

    measures = [
        ("choice_a", "P(Choose A)"),
        ("switch_flag_verified", "Switch Rate"),
        ("reward_outcome", "Reward Rate"),
    ]

    n_sessions = events_df["session_id"].nunique()
    # Use 2-second bins for smoother individual traces
    bin_width = 2

    fig, axes = plt.subplots(len(measures), len(transitions),
                             figsize=(5 * len(transitions), 3.5 * len(measures)),
                             sharex="col")
    if len(transitions) == 1:
        axes = axes.reshape(-1, 1)

    indiv_palette = sns.color_palette("husl", n_sessions)

    for col, tr in enumerate(transitions):
        t_trans = tr["t_s"]
        for row, (measure, ylabel) in enumerate(measures):
            ax = axes[row, col]
            bins_data = {}
            indiv_idx = 0

            for sid, sdf in events_df.groupby("session_id"):
                w = sdf[(sdf["elapsed_time_s"] >= t_trans - pre_s) &
                         (sdf["elapsed_time_s"] <= t_trans + post_s)].copy()
                if len(w) < 3 or measure not in w.columns:
                    continue
                w["rel"] = w["elapsed_time_s"] - t_trans
                w["tb"] = (w["rel"] // bin_width * bin_width).astype(int)

                # Individual trace
                indiv = w.groupby("tb")[measure].mean()
                ax.plot(indiv.index, indiv.values,
                        color=indiv_palette[indiv_idx % len(indiv_palette)],
                        alpha=0.35, lw=0.8)
                indiv_idx += 1

                for tb, grp in w.groupby("tb"):
                    bins_data.setdefault(tb, []).append(grp[measure].mean())

            # Group mean + bootstrap CI
            x, m, lo, hi = _bootstrap_ci(bins_data, n_boot=n_boot, rng=rng)
            ax.plot(x, m, color="black", lw=2.2, zorder=5)
            ax.fill_between(x, lo, hi, alpha=0.18, color="steelblue", zorder=4)

            # Transition line
            ax.axvline(0, color="red", ls="--", alpha=0.7, lw=1.0, zorder=6)

            if measure == "choice_a":
                ax.axhline(0.5, color="gray", ls="--", alpha=0.3, lw=0.6)

            ax.set_ylabel(ylabel if col == 0 else "")
            if row == 0:
                ax.set_title(tr["label"], fontsize=10, fontweight="bold")
                if tr["note"]:
                    ax.text(0.5, 1.0, tr["note"], transform=ax.transAxes,
                            ha="center", va="top", fontsize=7.5,
                            color="gray", style="italic")
            if row == len(measures) - 1:
                ax.set_xlabel("Time relative to transition (s)")

            sns.despine(ax=ax)

    # Legend
    handles = [
        Line2D([0], [0], color="black", lw=2.2, label="Group mean"),
        mpatches.Patch(color="steelblue", alpha=0.18, label="95% bootstrap CI"),
        Line2D([0], [0], color="gray", lw=0.8, alpha=0.5,
               label="Individual participants"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.03), frameon=True, edgecolor="lightgray")

    fig.suptitle("Figure 3: Phase-Transition Dynamics", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure3_phase_transitions.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 – Pulse Response
# ═══════════════════════════════════════════════════════════════════════════

def _figure4_pulse_response(events_df, fig_dir, fmt, config, n_boot):
    """Pulse-triggered averaging: how behavior shifts when a bonus pulse activates.

    Left panel: P(choose pulsed option) around A-pulses vs B-pulses.
    Center panel: Reward rate around all pulses (pooled).
    Right panel: Switch rate around all pulses — do participants switch toward
    the pulsed option?

    Shaded green region = pulse active. Thin lines = individual participants.
    Bold line = group mean ± bootstrap CI.
    """
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

    n_sessions = events_df["session_id"].nunique()
    # Use 1-second bins for cleaner traces
    bin_width = 1.0

    # ── Collect per-target and pooled data ──
    target_bins = {"A": {}, "B": {}}
    target_indiv = {"A": {}, "B": {}}   # {target: {sid: {tb: mean}}}
    rew_bins = {}
    rew_indiv = {}
    switch_bins = {}
    switch_indiv = {}

    for i, target in enumerate(sequence):
        onset_s = (phase4_start + i * interval) / 1000
        for sid, sdf in events_df.groupby("session_id"):
            w = sdf[(sdf["elapsed_time_s"] >= onset_s - pre_s) &
                     (sdf["elapsed_time_s"] <= onset_s + post_s)].copy()
            if len(w) < 2:
                continue
            w["rel"] = w["elapsed_time_s"] - onset_s
            w["tb"] = (w["rel"] // bin_width * bin_width).astype(float)

            # Choice toward pulsed option
            w["chose_target"] = (w["chosen_option"] == target).astype(float)
            for tb, grp in w.groupby("tb"):
                target_bins[target].setdefault(tb, []).append(grp["chose_target"].mean())
                # Individual: average across pulses for this participant
                key = (sid, target)
                target_indiv[target].setdefault(key, {})
                target_indiv[target][key].setdefault(tb, [])
                target_indiv[target][key][tb].append(grp["chose_target"].mean())

            # Reward rate (pooled across targets)
            for tb, grp in w.groupby("tb"):
                rew_bins.setdefault(tb, []).append(grp["reward_outcome"].mean())
                rew_indiv.setdefault(sid, {}).setdefault(tb, [])
                rew_indiv[sid][tb].append(grp["reward_outcome"].mean())

            # Switch rate
            if "switch_flag_verified" in w.columns:
                for tb, grp in w.groupby("tb"):
                    switch_bins.setdefault(tb, []).append(
                        grp["switch_flag_verified"].mean())
                    switch_indiv.setdefault(sid, {}).setdefault(tb, [])
                    switch_indiv[sid][tb].append(grp["switch_flag_verified"].mean())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def _shade_pulse(ax):
        """Shade the pulse-active region."""
        ax.axvspan(0, duration_s, color="#5AAE61", alpha=0.10, zorder=0,
                   label="_nolegend_")
        ax.axvline(0, color="#5AAE61", ls="-", alpha=0.4, lw=0.8)
        ax.axvline(duration_s, color="#5AAE61", ls=":", alpha=0.4, lw=0.8)

    # ── Panel 1: Choice toward pulsed option by target ──
    ax = axes[0]
    _shade_pulse(ax)
    for target, color in [("A", "#4393C3"), ("B", "#D6604D")]:
        # Individual traces (averaged across pulses per participant)
        for key, tb_dict in target_indiv[target].items():
            xs = sorted(tb_dict.keys())
            ys = [np.mean(tb_dict[t]) for t in xs]
            ax.plot(xs, ys, color=color, alpha=0.15, lw=0.5)

        # Group mean + CI
        if target_bins[target]:
            x, m, lo, hi = _bootstrap_ci(target_bins[target], n_boot, rng=rng)
            ax.plot(x, m, color=color, lw=2.2, label=f"{target}-pulse", zorder=5)
            ax.fill_between(x, lo, hi, alpha=0.15, color=color, zorder=4)

    ax.axhline(0.5, color="gray", ls="--", alpha=0.3, lw=0.6)
    ax.set_xlabel("Time relative to pulse onset (s)")
    ax.set_ylabel("P(Choose pulsed option)")
    ax.set_title("Choice toward pulsed option", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    sns.despine(ax=ax)

    # ── Panel 2: Reward rate around pulses (pooled) ──
    ax = axes[1]
    _shade_pulse(ax)
    # Individual traces
    for sid, tb_dict in rew_indiv.items():
        xs = sorted(tb_dict.keys())
        ys = [np.mean(tb_dict[t]) for t in xs]
        ax.plot(xs, ys, color="darkorange", alpha=0.15, lw=0.5)
    # Group
    if rew_bins:
        x, m, lo, hi = _bootstrap_ci(rew_bins, n_boot, rng=rng)
        ax.plot(x, m, color="darkorange", lw=2.2, zorder=5)
        ax.fill_between(x, lo, hi, alpha=0.15, color="darkorange", zorder=4)
    ax.set_xlabel("Time relative to pulse onset (s)")
    ax.set_ylabel("Reward rate")
    ax.set_title("Reward rate around pulses", fontsize=10, fontweight="bold")
    sns.despine(ax=ax)

    # ── Panel 3: Switch rate around pulses ──
    ax = axes[2]
    _shade_pulse(ax)
    for sid, tb_dict in switch_indiv.items():
        xs = sorted(tb_dict.keys())
        ys = [np.mean(tb_dict[t]) for t in xs]
        ax.plot(xs, ys, color="purple", alpha=0.15, lw=0.5)
    if switch_bins:
        x, m, lo, hi = _bootstrap_ci(switch_bins, n_boot, rng=rng)
        ax.plot(x, m, color="purple", lw=2.2, zorder=5)
        ax.fill_between(x, lo, hi, alpha=0.15, color="purple", zorder=4)
    ax.set_xlabel("Time relative to pulse onset (s)")
    ax.set_ylabel("Switch rate")
    ax.set_title("Switching around pulses", fontsize=10, fontweight="bold")
    sns.despine(ax=ax)

    # Shared legend for pulse shading
    pulse_patch = mpatches.Patch(color="#5AAE61", alpha=0.15, label="Pulse active")
    indiv_line = Line2D([0], [0], color="gray", alpha=0.4, lw=0.8,
                        label="Individual participants")
    fig.legend(handles=[pulse_patch, indiv_line], loc="lower center", ncol=2,
               fontsize=8, bbox_to_anchor=(0.5, -0.04), frameon=True,
               edgecolor="lightgray")

    fig.suptitle("Figure 4: Perturbation Response to Bonus Pulses",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure4_pulse_response.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 – State Space Trajectories
# ═══════════════════════════════════════════════════════════════════════════

def _figure5_state_space(events_df, fig_dir, fmt, config):
    """State-space trajectories: KDE density per phase + time-colored paths.

    Left panel:  Group-level KDE contours showing where behavior clusters in
                 P(A) × reward-rate space, one contour set per phase.
    Right panels: Individual session trajectories colored by elapsed time
                  (light→dark) so temporal order is visible.
    """
    if "rolling_choice_prop_a_clicks" not in events_df.columns:
        return

    from scipy.stats import gaussian_kde

    y_col = ("rolling_reward_rate_clicks"
             if "rolling_reward_rate_clicks" in events_df.columns
             else "reward_outcome")

    sessions = events_df["session_id"].unique()
    n_indiv = min(4, len(sessions))

    # Layout: 1 KDE panel + up to 4 individual panels
    n_cols = 1 + n_indiv
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    # ── Left panel: Group KDE contours by phase ──
    ax_kde = axes[0]
    grid_x = np.linspace(0, 1, 80)
    grid_y = np.linspace(0, 1, 80)
    X, Y = np.meshgrid(grid_x, grid_y)
    grid_pts = np.vstack([X.ravel(), Y.ravel()])

    for pid, color in _PHASE_COLORS.items():
        pdf = events_df[events_df["phase_id"] == pid].dropna(
            subset=["rolling_choice_prop_a_clicks", y_col])
        if len(pdf) < 10:
            continue
        xv = pdf["rolling_choice_prop_a_clicks"].values
        yv = pdf[y_col].values
        # Add small jitter to avoid singular matrix
        xv = xv + np.random.default_rng(pid).normal(0, 0.005, len(xv))
        yv = yv + np.random.default_rng(pid + 10).normal(0, 0.005, len(yv))
        try:
            kde = gaussian_kde(np.vstack([xv, yv]), bw_method=0.15)
            Z = kde(grid_pts).reshape(X.shape)
            ax_kde.contour(X, Y, Z, levels=4, colors=[color],
                           alpha=0.7, linewidths=1.2)
            ax_kde.contourf(X, Y, Z, levels=4, colors=[color],
                            alpha=0.08)
        except np.linalg.LinAlgError:
            # Fallback: just scatter
            ax_kde.scatter(xv, yv, s=3, alpha=0.2, color=color)

    ax_kde.set_xlim(0, 1)
    ax_kde.set_ylim(0, 1)
    ax_kde.set_xlabel("P(Choose A)")
    ax_kde.set_ylabel("Reward Rate")
    ax_kde.set_title("Group density by phase", fontsize=10, fontweight="bold")

    phase_handles = [mpatches.Patch(color=c, alpha=0.4, label=l)
                     for c, l in [("#999999", "Symmetric"), ("#4393C3", "A-adv"),
                                  ("#D6604D", "B-adv"), ("#5AAE61", "Scarcity")]]
    ax_kde.legend(handles=phase_handles, fontsize=7, loc="lower left",
                  framealpha=0.9)
    sns.despine(ax=ax_kde)

    # ── Right panels: Individual trajectories colored by time ──
    # Pick diverse sessions by spread of switch rates
    per_session = events_df.groupby("session_id").agg(
        sr=("switch_flag_verified", "mean")).sort_values("sr")
    n = len(per_session)
    pick_idx = np.linspace(0, n - 1, n_indiv).astype(int)
    selected = per_session.index[pick_idx]

    for i, sid in enumerate(selected):
        ax = axes[1 + i]
        sdf = events_df[events_df["session_id"] == sid].sort_values("timestamp_ms")
        xv = sdf["rolling_choice_prop_a_clicks"].values
        yv = sdf[y_col].values
        t_norm = np.linspace(0, 1, len(sdf))  # normalized time for colormap

        # Draw trajectory segments colored by time
        for j in range(len(sdf) - 1):
            ax.plot(xv[j:j+2], yv[j:j+2],
                    color=plt.cm.viridis(t_norm[j]),
                    alpha=0.5, lw=0.8)

        # Scatter points colored by time
        sc = ax.scatter(xv, yv, c=t_norm, cmap="viridis",
                        s=6, alpha=0.6, zorder=3)

        # Mark start and end
        ax.scatter(xv[0], yv[0], marker="o", s=50, color="lime",
                   edgecolor="black", zorder=5, label="Start")
        ax.scatter(xv[-1], yv[-1], marker="s", s=50, color="red",
                   edgecolor="black", zorder=5, label="End")

        ax.set_xlim(0, 1)
        ax.set_xlabel("P(Choose A)")
        ax.set_ylabel("")
        sid_short = str(sid)[:10]
        ax.set_title(sid_short, fontsize=9)
        sns.despine(ax=ax)

        if i == 0:
            ax.legend(fontsize=7, loc="lower left", framealpha=0.9)

    # Colorbar for time
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis"),
                        ax=axes[-1], fraction=0.05, pad=0.04)
    cbar.set_label("Time (early → late)", fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Start", "Mid", "End"])

    fig.suptitle("Figure 5: State-Space Trajectories", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure5_state_space.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6 – Individual Differences
# ═══════════════════════════════════════════════════════════════════════════

def _figure6_individual_differences(metrics_df, fig_dir, fmt, config):
    """Individual differences in foraging strategy.

    Panel A: Participant profiles — parallel coordinates showing each participant's
             z-scored metrics. Each line is one participant; diverging lines reveal
             distinct strategy types (e.g., high-switching explorers vs. sticky exploiters).
    Panel B: Key bivariate scatter plots showing how behavioral dimensions relate:
             switch rate vs. proportion optimal, mean run length vs. reward rate,
             and adaptation lag vs. choice entropy.
    Panel C: Correlation matrix of all individual-difference metrics.
    """
    # ── Collect available metrics ──
    metric_defs = [
        ("overall_switch_rate", "Switch Rate"),
        ("mean_run_length", "Mean Run Length"),
        ("choice_entropy", "Choice Entropy"),
        ("proportion_optimal", "Prop. Optimal"),
        ("adaptation_lag_phase2_s", "Adaptation Lag (s)"),
        ("wsls_score", "WSLS Score"),
    ]
    metric_defs = [(c, l) for c, l in metric_defs
                   if c in metrics_df.columns and metrics_df[c].notna().sum() > 0]

    # Also collect reward rate if available
    reward_col = None
    for rc in ["overall_reward_rate", "mean_reward_rate"]:
        if rc in metrics_df.columns:
            reward_col = rc
            break

    n_participants = len(metrics_df)
    cols_available = [c for c, _ in metric_defs]

    # ── Define scatter-plot pairs (x_col, y_col, xlabel, ylabel) ──
    scatter_pairs = []
    if "overall_switch_rate" in cols_available and "proportion_optimal" in cols_available:
        scatter_pairs.append(("overall_switch_rate", "proportion_optimal",
                              "Switch Rate", "Proportion Optimal"))
    if "mean_run_length" in cols_available and reward_col:
        scatter_pairs.append(("mean_run_length", reward_col,
                              "Mean Run Length", "Reward Rate"))
    if "adaptation_lag_phase2_s" in cols_available and "choice_entropy" in cols_available:
        scatter_pairs.append(("adaptation_lag_phase2_s", "choice_entropy",
                              "Adaptation Lag (s)", "Choice Entropy"))
    # Fallback: if fewer than 2 pairs, add whatever we can
    if len(scatter_pairs) < 2 and len(cols_available) >= 2:
        for i in range(len(cols_available)):
            for j in range(i + 1, len(cols_available)):
                pair = (cols_available[i], cols_available[j],
                        dict(metric_defs)[cols_available[i]],
                        dict(metric_defs)[cols_available[j]])
                if pair not in scatter_pairs:
                    scatter_pairs.append(pair)
                if len(scatter_pairs) >= 3:
                    break
            if len(scatter_pairs) >= 3:
                break
    scatter_pairs = scatter_pairs[:3]

    n_scatter = len(scatter_pairs)

    # ── Layout ──
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, max(n_scatter, 2), height_ratios=[1, 1],
                           hspace=0.35, wspace=0.35)

    # ── Panel A (top-left): Participant strategy profiles ──
    ax_profile = fig.add_subplot(gs[0, :n_scatter])

    if len(metric_defs) >= 2 and n_participants >= 2:
        profile_data = metrics_df[cols_available].dropna()
        if len(profile_data) >= 2:
            # Z-score each metric
            z_data = (profile_data - profile_data.mean()) / profile_data.std().replace(0, 1)
            labels = [l for _, l in metric_defs]

            palette = sns.color_palette("husl", len(z_data))
            x_pos = np.arange(len(labels))

            for idx, (row_idx, row) in enumerate(z_data.iterrows()):
                # Try to get a participant label
                if "session_id" in metrics_df.columns:
                    sid = metrics_df.loc[row_idx, "session_id"]
                    plabel = str(sid)[:10]
                else:
                    plabel = f"P{idx+1}"
                ax_profile.plot(x_pos, row.values, marker="o", ms=8,
                                color=palette[idx], lw=2, alpha=0.8,
                                label=plabel, zorder=3)

            ax_profile.axhline(0, color="gray", ls="--", alpha=0.4, lw=0.8)
            ax_profile.set_xticks(x_pos)
            ax_profile.set_xticklabels(labels, fontsize=9)
            ax_profile.set_ylabel("z-score")
            ax_profile.set_title("A. Participant strategy profiles",
                                 fontsize=11, fontweight="bold", loc="left")
            ax_profile.legend(fontsize=7, loc="best", framealpha=0.9,
                              title="Participant", title_fontsize=8)
            sns.despine(ax=ax_profile)
        else:
            ax_profile.text(0.5, 0.5, "Insufficient data for profiles",
                            ha="center", va="center", transform=ax_profile.transAxes,
                            fontsize=11, color="gray")
            ax_profile.axis("off")
    else:
        ax_profile.text(0.5, 0.5, "Insufficient metrics for profiles",
                        ha="center", va="center", transform=ax_profile.transAxes,
                        fontsize=11, color="gray")
        ax_profile.axis("off")

    # ── Panel B (bottom): Bivariate scatter plots ──
    scatter_colors = sns.color_palette("husl", n_participants)
    for i, (xc, yc, xl, yl) in enumerate(scatter_pairs):
        ax = fig.add_subplot(gs[1, i])
        valid = metrics_df[[xc, yc]].dropna()
        if len(valid) == 0:
            ax.axis("off")
            continue

        for j, (_, row) in enumerate(valid.iterrows()):
            ax.scatter(row[xc], row[yc], s=70, color=scatter_colors[j % len(scatter_colors)],
                       edgecolor="white", linewidth=0.5, zorder=3)

        # Trend line if enough data
        if len(valid) >= 4:
            z = np.polyfit(valid[xc], valid[yc], 1)
            x_range = np.linspace(valid[xc].min(), valid[xc].max(), 50)
            ax.plot(x_range, np.polyval(z, x_range), color="black",
                    ls="--", lw=1.2, alpha=0.5)
            # Pearson r annotation
            r = valid[xc].corr(valid[yc])
            ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                    fontsize=9, va="top", color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl if i == 0 else yl, fontsize=9)
        panel_letter = chr(ord("B") + i)
        ax.set_title(f"{panel_letter}. {xl} vs. {yl}",
                     fontsize=10, fontweight="bold", loc="left")
        sns.despine(ax=ax)

    # Hide unused bottom panels
    for j in range(n_scatter, gs.ncols):
        ax_empty = fig.add_subplot(gs[1, j])
        ax_empty.set_visible(False)

    fig.suptitle("Figure 6: Individual Differences in Foraging Strategy",
                 fontsize=13, y=1.01)
    _save(fig, os.path.join(fig_dir, f"figure6_individual_differences.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7 – Model Comparison
# ═══════════════════════════════════════════════════════════════════════════

def _figure7_model_comparison(analysis_results, fig_dir, fmt, config):
    mc = analysis_results.get("model_comparison")
    if mc is None or len(mc) == 0:
        return

    # Publication-ready model names
    _MODEL_LABELS = {
        "random": "Random",
        "bias_only": "Bias Only",
        "wsls": "Win–Stay, Lose–Shift",
        "logistic": "Logistic Regression",
        "q_learning": "Q-Learning",
        "q_dual_alpha": "Q-Learning (Dual α)",
        "q_forgetting": "Q-Learning (Forgetting)",
        "matching_strict": "Strict Matching",
        "matching_generalized": "Generalized Matching",
    }

    mc = mc.copy()
    mc["model_label"] = mc["model"].map(
        lambda m: _MODEL_LABELS.get(m, m.replace("_", " ").title()))

    # Shorten session IDs for display
    mc["session_short"] = mc["session_id"].astype(str).str[:10]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: mean BIC bar chart
    ax = axes[0]
    agg = mc.groupby("model_label")["bic"].agg(["mean", "sem"]).sort_values("mean")
    colors = sns.color_palette("Set2", len(agg))
    ax.barh(range(len(agg)), agg["mean"], xerr=agg["sem"],
            color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels(agg.index, fontsize=9)
    ax.set_xlabel("Mean BIC (lower = better fit)")
    ax.set_title("Mean BIC by Model", fontsize=10, fontweight="bold")
    sns.despine(ax=ax)

    # Right: delta-BIC heatmap
    ax = axes[1]
    bic_pivot = mc.pivot(index="session_short", columns="model_label", values="bic")
    delta = bic_pivot.sub(bic_pivot.min(axis=1), axis=0)
    col_order = delta.mean().sort_values().index
    delta = delta[col_order]
    sns.heatmap(delta, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "ΔBIC"}, linewidths=0.5)
    ax.set_title("ΔBIC from Best (per session)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Session")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("")  # remove default "model_label"

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
