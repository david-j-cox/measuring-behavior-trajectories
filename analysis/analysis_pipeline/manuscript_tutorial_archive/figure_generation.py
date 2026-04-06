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
    _figure8_rqa(analysis_results, events_df, fig_dir, fmt, config)
    _figure9_dfa(analysis_results, fig_dir, fmt, config)
    _figure10_entropy(analysis_results, fig_dir, fmt, config)
    _figure11_ccm(analysis_results, fig_dir, fmt, config)
    _figure12_smap(analysis_results, fig_dir, fmt, config)
    _figure13_hmm(analysis_results, events_df, fig_dir, fmt, config)
    _figure14_phenotypes(analysis_results, fig_dir, fmt, config)
    _figure15_null_comparison(analysis_results, fig_dir, fmt, config)
    _figure16_robustness(analysis_results, fig_dir, fmt, config)

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

    n_rows = len(selected) + 1  # 4 individual + 1 group average
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.5 * n_rows),
                             sharex=True)
    if n_rows == 1:
        axes = [axes]

    # ── Rows 1–4: Individual example trajectories ──
    for ax, sid in zip(axes[:len(selected)], selected):
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

    # ── Row 5: Group-average P(A) and P(B) in 10-s bins with 95% CI ──
    ax_group = axes[-1]
    bin_width = 10  # seconds
    n_boot = config.get("n_bootstrap", 1000)
    rng = np.random.default_rng(0)

    # Compute per-session proportion in each time bin
    pa_by_bin = {}  # {bin_center: [prop_A for each session]}
    pb_by_bin = {}
    for sid, sdf in events_df.groupby("session_id"):
        sdf = sdf.sort_values("elapsed_time_s")
        sdf_binned = sdf.copy()
        sdf_binned["time_bin"] = (sdf_binned["elapsed_time_s"] // bin_width) * bin_width + bin_width / 2
        for tb, grp in sdf_binned.groupby("time_bin"):
            pa = grp["choice_a"].mean()
            pa_by_bin.setdefault(tb, []).append(pa)
            pb_by_bin.setdefault(tb, []).append(1 - pa)

    bins_a, mean_a, lo_a, hi_a = _bootstrap_ci(pa_by_bin, n_boot=n_boot, rng=rng)
    bins_b, mean_b, lo_b, hi_b = _bootstrap_ci(pb_by_bin, n_boot=n_boot, rng=rng)

    ax_group.plot(bins_a, mean_a, color="#4393C3", lw=2, label="P(A)")
    ax_group.fill_between(bins_a, lo_a, hi_a, color="#4393C3", alpha=0.2)
    ax_group.plot(bins_b, mean_b, color="#D6604D", lw=2, label="P(B)")
    ax_group.fill_between(bins_b, lo_b, hi_b, color="#D6604D", alpha=0.2)
    ax_group.axhline(0.5, color="gray", ls="--", alpha=0.4, lw=0.6)

    _phase_spans(ax_group, config)
    _phase_lines(ax_group, config)

    ax_group.set_ylim(0, 1)
    ax_group.set_ylabel("Proportion")
    ax_group.set_xlabel("Time (s)")
    ax_group.set_title("Group average (N = %d)" % events_df["session_id"].nunique(),
                       fontsize=10, loc="left")
    ax_group.legend(loc="upper right", fontsize=9, framealpha=0.9)

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
                        color="gray",
                        alpha=0.12, lw=0.5)
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

            x_pos = np.arange(len(labels))

            # Sort by first metric for visual clustering
            sort_col = cols_available[0]
            sorted_idx = profile_data[sort_col].sort_values().index
            z_sorted = z_data.loc[sorted_idx]
            im = ax_profile.imshow(z_sorted.values, aspect="auto", cmap="RdBu_r",
                                    vmin=-3, vmax=3)
            ax_profile.set_xticks(x_pos)
            ax_profile.set_xticklabels(labels, fontsize=9)
            ax_profile.set_ylabel("Participant (sorted)")
            ax_profile.set_yticks([])  # too many to label
            plt.colorbar(im, ax=ax_profile, label="z-score", shrink=0.8)
            ax_profile.set_title("A. Participant strategy profiles",
                                 fontsize=11, fontweight="bold", loc="left")
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
        "generalized_matching_law": "Generalized Matching Law",
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

    # Right: proportion of sessions each model wins
    ax = axes[1]
    best_per_session = mc.loc[mc.groupby("session_id")["bic"].idxmin()]
    win_counts = best_per_session["model_label"].value_counts()
    # Order by mean BIC
    model_order = agg.index.tolist()
    win_counts = win_counts.reindex(model_order, fill_value=0)
    bars = ax.barh(range(len(win_counts)), win_counts.values,
                   color=[colors[i] for i in range(len(win_counts))],
                   edgecolor="white", height=0.7)
    ax.set_yticks(range(len(win_counts)))
    ax.set_yticklabels(win_counts.index, fontsize=9)
    ax.set_xlabel("Number of sessions where model is best")
    ax.set_title("Best Model per Session", fontsize=10, fontweight="bold")
    # Add count labels
    for bar, count in zip(bars, win_counts.values):
        if count > 0:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    str(count), va="center", fontsize=8)
    sns.despine(ax=ax)

    fig.suptitle("Figure 7: Model Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure7_model_comparison.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8 – RQA
# ═══════════════════════════════════════════════════════════════════════════

def _figure8_rqa(analysis_results, events_df, fig_dir, fmt, config):
    """RQA: (A) example recurrence plot, (B) metric distributions, (C) by phase."""
    rqa_df = analysis_results.get("rqa")
    if rqa_df is None or len(rqa_df) == 0:
        print("  Figure 8 skipped: no RQA results.")
        return

    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.2, 1])

    # (A) Example recurrence plot
    ax_rp = fig.add_subplot(gs[0])
    sids = events_df["session_id"].unique()
    if len(sids) > 0 and "rolling_choice_prop_a_clicks" in events_df.columns:
        sid = sids[len(sids) // 2]  # pick a middle session
        sdf = events_df[events_df["session_id"] == sid].sort_values("timestamp_ms")
        signal = sdf["rolling_choice_prop_a_clicks"].dropna().values
        if len(signal) > 300:
            signal = signal[::len(signal) // 300]
        dist = np.abs(signal[:, None] - signal[None, :])
        recurrence = (dist < 0.1).astype(int)
        np.fill_diagonal(recurrence, 0)
        ax_rp.imshow(recurrence, cmap="Greys", origin="lower", aspect="equal")
        ax_rp.set_xlabel("Click index")
        ax_rp.set_ylabel("Click index")
    ax_rp.set_title("A. Example Recurrence Plot", fontsize=10, fontweight="bold",
                     loc="left")

    # (B) Metric distributions
    ax_dist = fig.add_subplot(gs[1])
    full = rqa_df[rqa_df["scope"] == "full_session"]
    metrics = ["determinism", "recurrence_rate", "laminarity"]
    labels = ["Determinism", "Recurrence Rate", "Laminarity"]
    colors = ["#4393C3", "#D6604D", "#5AAE61"]
    positions = np.arange(len(metrics))

    parts = ax_dist.violinplot(
        [full[m].dropna().values for m in metrics],
        positions=positions, showmeans=True, showextrema=False
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    ax_dist.set_xticks(positions)
    ax_dist.set_xticklabels(labels, fontsize=9)
    ax_dist.set_ylabel("Value")
    ax_dist.set_title("B. RQA Metric Distributions", fontsize=10,
                       fontweight="bold", loc="left")
    sns.despine(ax=ax_dist)

    # (C) Determinism by phase
    ax_phase = fig.add_subplot(gs[2])
    phase_data = rqa_df[rqa_df["phase_id"] > 0].copy()
    if len(phase_data) > 0:
        phase_data["phase_label"] = "Phase " + phase_data["phase_id"].astype(str)
        for pid in sorted(phase_data["phase_id"].unique()):
            color = _PHASE_COLORS.get(pid, "gray")
            vals = phase_data.loc[phase_data["phase_id"] == pid, "determinism"].dropna()
            ax_phase.boxplot(
                [vals.values], positions=[pid], widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.5),
                medianprops=dict(color="black"),
                showfliers=False,
            )
            jitter = np.random.default_rng(pid).uniform(-0.12, 0.12, len(vals))
            ax_phase.scatter(pid + jitter, vals.values, s=15, alpha=0.4,
                             color=color, zorder=3)
        ax_phase.set_xticks([1, 2, 3, 4])
        ax_phase.set_xticklabels(["Ph 1\nSymmetric", "Ph 2\nA-adv",
                                   "Ph 3\nB-adv", "Ph 4\nScarcity"], fontsize=8)
    ax_phase.set_ylabel("Determinism")
    ax_phase.set_title("C. Determinism by Phase", fontsize=10,
                        fontweight="bold", loc="left")
    sns.despine(ax=ax_phase)

    fig.suptitle("Figure 8: Recurrence Quantification Analysis", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure8_rqa.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 9 – DFA
# ═══════════════════════════════════════════════════════════════════════════

def _figure9_dfa(analysis_results, fig_dir, fmt, config):
    """DFA: (A) distribution with reference lines, (B) by phase."""
    dfa_df = analysis_results.get("dfa")
    if dfa_df is None or len(dfa_df) == 0:
        print("  Figure 9 skipped: no DFA results.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (A) Distribution
    ax = axes[0]
    full = dfa_df[dfa_df["scope"] == "full_session"]
    vals = full["dfa_alpha"].dropna()
    ax.hist(vals, bins=15, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0.5, color="red", ls="--", lw=1.5, label="Random (0.5)")
    ax.axvline(1.0, color="orange", ls="--", lw=1.5, label="1/f noise (1.0)")
    ax.axvline(vals.median(), color="darkblue", ls="-", lw=2,
               label=f"Median ({vals.median():.2f})")
    ax.set_xlabel("DFA α")
    ax.set_ylabel("Count")
    ax.set_title("A. DFA Exponent Distribution", fontsize=10,
                  fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    sns.despine(ax=ax)

    # (B) By phase
    ax = axes[1]
    phase_data = dfa_df[dfa_df["phase_id"] > 0].copy()
    # Clip implausible DFA values (valid range ~0-2)
    phase_data = phase_data[phase_data["dfa_alpha"] <= 2.5].copy()
    if len(phase_data) > 0:
        for pid in sorted(phase_data["phase_id"].unique()):
            color = _PHASE_COLORS.get(pid, "gray")
            vals_p = phase_data.loc[phase_data["phase_id"] == pid,
                                     "dfa_alpha"].dropna()
            ax.boxplot(
                [vals_p.values], positions=[pid], widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.5),
                medianprops=dict(color="black"),
                showfliers=False,
            )
            jitter = np.random.default_rng(pid).uniform(-0.12, 0.12, len(vals_p))
            ax.scatter(pid + jitter, vals_p.values, s=15, alpha=0.4,
                       color=color, zorder=3)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["Ph 1\nSymmetric", "Ph 2\nA-adv",
                             "Ph 3\nB-adv", "Ph 4\nScarcity"], fontsize=8)
    ax.axhline(0.5, color="red", ls="--", alpha=0.5, lw=0.8)
    ax.set_ylabel("DFA α")
    ax.set_title("B. DFA by Phase", fontsize=10, fontweight="bold", loc="left")
    sns.despine(ax=ax)

    fig.suptitle("Figure 9: Detrended Fluctuation Analysis", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure9_dfa.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 10 – Sample Entropy
# ═══════════════════════════════════════════════════════════════════════════

def _figure10_entropy(analysis_results, fig_dir, fmt, config):
    """Sample entropy: (A) distribution, (B) by phase."""
    se_df = analysis_results.get("sample_entropy")
    if se_df is None or len(se_df) == 0:
        print("  Figure 10 skipped: no sample entropy results.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (A) Distribution
    ax = axes[0]
    full = se_df[se_df["scope"] == "full_session"]
    vals = full["sample_entropy"].dropna()
    ax.hist(vals, bins=15, color="teal", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Sample Entropy")
    ax.set_ylabel("Count")
    ax.set_title("A. Sample Entropy Distribution", fontsize=10,
                  fontweight="bold", loc="left")
    sns.despine(ax=ax)

    # (B) By phase
    ax = axes[1]
    phase_data = se_df[se_df["phase_id"] > 0].copy()
    if len(phase_data) > 0:
        for pid in sorted(phase_data["phase_id"].unique()):
            color = _PHASE_COLORS.get(pid, "gray")
            vals_p = phase_data.loc[phase_data["phase_id"] == pid,
                                     "sample_entropy"].dropna()
            ax.boxplot(
                [vals_p.values], positions=[pid], widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.5),
                medianprops=dict(color="black"),
                showfliers=False,
            )
            jitter = np.random.default_rng(pid).uniform(-0.12, 0.12, len(vals_p))
            ax.scatter(pid + jitter, vals_p.values, s=15, alpha=0.4,
                       color=color, zorder=3)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["Ph 1\nSymmetric", "Ph 2\nA-adv",
                             "Ph 3\nB-adv", "Ph 4\nScarcity"], fontsize=8)
    ax.set_ylabel("Sample Entropy")
    ax.set_title("B. Sample Entropy by Phase", fontsize=10,
                  fontweight="bold", loc="left")
    sns.despine(ax=ax)

    fig.suptitle("Figure 10: Sample Entropy", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure10_entropy.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 11 – CCM
# ═══════════════════════════════════════════════════════════════════════════

def _figure11_ccm(analysis_results, fig_dir, fmt, config):
    """CCM convergence: reward→choice vs choice→reward."""
    ccm_df = analysis_results.get("ccm")
    if ccm_df is None or len(ccm_df) == 0:
        print("  Figure 11 skipped: no CCM results.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (A) Group-average convergence curves
    ax = axes[0]
    group = ccm_df.groupby("lib_size").agg(
        rc_mean=("rho_reward_causes_choice", "mean"),
        rc_sem=("rho_reward_causes_choice", "sem"),
        cr_mean=("rho_choice_causes_reward", "mean"),
        cr_sem=("rho_choice_causes_reward", "sem"),
    ).reset_index()

    ax.plot(group["lib_size"], group["rc_mean"], "o-", color="#4393C3",
            lw=2, markersize=5, label="Reward → Choice")
    ax.fill_between(group["lib_size"],
                    group["rc_mean"] - group["rc_sem"],
                    group["rc_mean"] + group["rc_sem"],
                    alpha=0.15, color="#4393C3")
    ax.plot(group["lib_size"], group["cr_mean"], "s-", color="#D6604D",
            lw=2, markersize=5, label="Choice → Reward")
    ax.fill_between(group["lib_size"],
                    group["cr_mean"] - group["cr_sem"],
                    group["cr_mean"] + group["cr_sem"],
                    alpha=0.15, color="#D6604D")

    ax.set_xlabel("Library Size")
    ax.set_ylabel("Cross-Mapping ρ")
    ax.set_title("A. Group-Average Convergence", fontsize=10,
                  fontweight="bold", loc="left")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    # (B) Per-session asymmetry at max library size
    ax = axes[1]
    max_lib = ccm_df.loc[ccm_df.groupby("session_id")["lib_size"].idxmax()]
    if "rho_reward_causes_choice" in max_lib.columns:
        rc = max_lib["rho_reward_causes_choice"].values
        cr = max_lib["rho_choice_causes_reward"].values
        ax.scatter(cr, rc, s=40, alpha=0.7, color="steelblue", edgecolors="white")
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "--", color="gray", alpha=0.5)
        ax.set_xlabel("Choice → Reward ρ")
        ax.set_ylabel("Reward → Choice ρ")
    ax.set_title("B. Per-Session Asymmetry", fontsize=10,
                  fontweight="bold", loc="left")
    sns.despine(ax=ax)

    fig.suptitle("Figure 11: Convergent Cross-Mapping", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure11_ccm.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 12 – S-Map
# ═══════════════════════════════════════════════════════════════════════════

def _figure12_smap(analysis_results, fig_dir, fmt, config):
    """S-Map nonlinearity: simplex vs S-Map prediction."""
    smap_df = analysis_results.get("smap")
    if smap_df is None or len(smap_df) == 0:
        print("  Figure 12 skipped: no S-Map results.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (A) Simplex vs S-Map rho
    ax = axes[0]
    ax.scatter(smap_df["simplex_rho"], smap_df["smap_rho"],
               s=40, alpha=0.7, color="steelblue", edgecolors="white")
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5)
    ax.set_xlabel("Simplex ρ (linear)")
    ax.set_ylabel("S-Map ρ (nonlinear)")
    ax.set_title("A. Linear vs Nonlinear Prediction", fontsize=10,
                  fontweight="bold", loc="left")
    sns.despine(ax=ax)

    # (B) Nonlinearity distribution
    ax = axes[1]
    ax.hist(smap_df["nonlinearity"].dropna(), bins=15, color="teal",
            edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Nonlinearity (Δρ = S-Map − Simplex)")
    ax.set_ylabel("Count")
    ax.set_title("B. Nonlinearity Score Distribution", fontsize=10,
                  fontweight="bold", loc="left")
    sns.despine(ax=ax)

    fig.suptitle("Figure 12: S-Map Nonlinearity Analysis", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure12_smap.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 13 – HMM
# ═══════════════════════════════════════════════════════════════════════════

def _figure13_hmm(analysis_results, events_df, fig_dir, fmt, config):
    """HMM: (A) state assignments on trajectories, (B) state occupancy by phase."""
    # Try state sequences first (has per-click assignments)
    hmm_seqs = analysis_results.get("hmm_state_sequences", {})
    if not hmm_seqs:
        # Fall back to summary
        hmm_summary = analysis_results.get("hmm_state_summary")
        if hmm_summary is None or len(hmm_summary) == 0:
            print("  Figure 13 skipped: no HMM results.")
            # Still create empty figure with note
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].text(0.5, 0.5, "HMM fitting did not converge",
                         ha="center", va="center", transform=axes[0].transAxes)
            axes[0].set_title("A. HMM State Assignments", fontsize=10, fontweight="bold", loc="left")
            axes[1].text(0.5, 0.5, "No state occupancy data",
                         ha="center", va="center", transform=axes[1].transAxes)
            axes[1].set_title("B. State Occupancy by Phase", fontsize=10, fontweight="bold", loc="left")
            fig.suptitle("Figure 13: Hidden Markov Model Analysis", fontsize=13, y=1.02)
            plt.tight_layout()
            _save(fig, os.path.join(fig_dir, f"figure13_hmm.{fmt}"), config)
            return
        hmm_seqs = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (A) State assignments on example session
    ax = axes[0]
    state_colors = ["#4393C3", "#D6604D", "#5AAE61", "#984EA3"]
    if hmm_seqs:
        sids = list(hmm_seqs.keys())
        sid = sids[0]
        sdf = events_df[events_df["session_id"] == sid].sort_values("timestamp_ms")
        seq = hmm_seqs[sid]
        states = seq["states"]
        n_states = seq["n_states"]

        signal_col = "rolling_choice_prop_a_clicks" if "rolling_choice_prop_a_clicks" in sdf.columns else "choice_a"
        t = sdf["elapsed_time_s"].values[:len(states)]
        signal = sdf[signal_col].values[:len(states)]

        for s in range(n_states):
            mask = states == s
            if mask.any():
                ax.scatter(t[mask], signal[mask], s=8, alpha=0.5,
                           color=state_colors[s % len(state_colors)],
                           label=f"State {s}")
        ax.plot(t, signal, color="black", alpha=0.2, lw=0.5)
        _phase_lines(ax, config)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No per-click state data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("P(Choose A)")
    ax.set_title("A. HMM State Assignments", fontsize=10, fontweight="bold", loc="left")
    sns.despine(ax=ax)

    # (B) State occupancy by phase
    ax = axes[1]
    if hmm_seqs:
        phase_boundaries = config.get("phase_boundaries", [])
        phase_state_counts = {}
        for sid, seq in hmm_seqs.items():
            sdf = events_df[events_df["session_id"] == sid].sort_values("timestamp_ms")
            states = seq["states"]
            phases = sdf["phase_id"].values[:len(states)]
            for pb in phase_boundaries:
                pid = pb["id"]
                phase_mask = phases == pid
                if phase_mask.sum() == 0:
                    continue
                for s in range(seq["n_states"]):
                    key = (pid, s)
                    phase_state_counts.setdefault(key, []).append((states[phase_mask] == s).mean())

        if phase_state_counts:
            # Build aggregated data
            all_phases = sorted(set(k[0] for k in phase_state_counts.keys()))
            all_states_list = sorted(set(k[1] for k in phase_state_counts.keys()))
            x = np.arange(len(all_phases))
            width = 0.8 / max(len(all_states_list), 1)
            for i, s in enumerate(all_states_list):
                means = [np.mean(phase_state_counts.get((p, s), [0])) for p in all_phases]
                ax.bar(x + i * width - 0.4 + width/2, means, width,
                       color=state_colors[s % len(state_colors)],
                       label=f"State {s}", edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels([f"Phase {p}" for p in all_phases], fontsize=9)
            ax.set_ylabel("Proportion of Time")
            ax.legend(title="State", fontsize=8, title_fontsize=9)
            ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "No state occupancy data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")

    ax.set_title("B. State Occupancy by Phase", fontsize=10, fontweight="bold", loc="left")
    sns.despine(ax=ax)

    fig.suptitle("Figure 13: Hidden Markov Model Analysis", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure13_hmm.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 14 – Behavioral Phenotypes
# ═══════════════════════════════════════════════════════════════════════════

def _figure14_phenotypes(analysis_results, fig_dir, fmt, config):
    """Behavioral phenotypes: (A) PCA biplot, (B) cluster profile heatmap."""
    id_results = analysis_results.get("individual_differences", {})
    pca_df = id_results.get("pca")
    cluster_df = id_results.get("cluster_assignments")
    profile_df = id_results.get("cluster_profiles")
    pca_model = id_results.get("pca_model")
    feature_names = id_results.get("feature_names", [])

    if pca_df is None or cluster_df is None:
        print("  Figure 14 skipped: no individual differences results.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (A) PCA biplot colored by cluster
    ax = axes[0]
    labels = cluster_df["cluster"].values
    n_clusters = len(np.unique(labels))
    palette = sns.color_palette("Set2", n_clusters)

    if "PC1" in pca_df.columns and "PC2" in pca_df.columns:
        pc1 = pca_df["PC1"].values
        pc2 = pca_df["PC2"].values
        for cl in range(n_clusters):
            mask = labels == cl
            ax.scatter(pc1[mask], pc2[mask], s=60, alpha=0.8,
                       color=palette[cl], edgecolors="white", linewidths=0.5,
                       label=f"Cluster {cl} (n={mask.sum()})")

        # Add loading vectors if PCA model available
        if pca_model is not None and len(feature_names) > 0:
            loadings = pca_model.components_[:2].T
            score_range = max(np.abs(np.concatenate([pc1, pc2])).max(), 1e-6)
            load_max = max(np.abs(loadings).max(), 1e-6)
            scale = score_range / load_max * 0.7
            for i, feat in enumerate(feature_names):
                dx, dy = loadings[i, 0] * scale, loadings[i, 1] * scale
                ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                             arrowprops=dict(arrowstyle="->", color="firebrick",
                                             lw=1.2, alpha=0.7))
                short = feat.replace("overall_", "").replace("_", " ")[:12]
                ax.text(dx * 1.1, dy * 1.1, short, fontsize=6.5,
                         ha="center", color="firebrick")

        ev = id_results.get("explained_variance", [0, 0])
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)" if len(ev) > 0 else "PC1")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)" if len(ev) > 1 else "PC2")
    ax.legend(fontsize=8)
    ax.set_title("A. PCA Biplot by Cluster", fontsize=10,
                  fontweight="bold", loc="left")
    sns.despine(ax=ax)

    # (B) Cluster profile heatmap
    ax = axes[1]
    if profile_df is not None and len(profile_df) > 0:
        best_k = id_results.get("best_k", n_clusters)
        pivot = profile_df.pivot(index="feature", columns="cluster",
                                  values="mean_z")
        # Sort by max absolute difference between clusters
        pivot = pivot.loc[pivot.std(axis=1).sort_values(ascending=False).index]
        short_labels = [f.replace("overall_", "").replace("_", " ")[:15]
                        for f in pivot.index]
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, linewidths=0.5, ax=ax,
                    yticklabels=short_labels,
                    xticklabels=[f"Cluster {c}" for c in range(best_k)])
        ax.set_ylabel("")
    ax.set_title("B. Cluster Profiles (z-scored)", fontsize=10,
                  fontweight="bold", loc="left")

    fig.suptitle("Figure 14: Behavioral Phenotypes", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure14_phenotypes.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 15 – Null Comparison
# ═══════════════════════════════════════════════════════════════════════════

def _figure15_null_comparison(analysis_results, fig_dir, fmt, config):
    """Null comparison: real data vs synthetic null processes."""
    null_table = analysis_results.get("null_comparison_table")
    if null_table is None or len(null_table) == 0:
        print("  Figure 15 skipped: no null comparison results.")
        return

    metrics = null_table["metric"].values
    n_metrics = len(metrics)
    n_cols = min(4, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    process_labels = ["Real", "Random", "Matching", "WSLS"]
    bar_colors = ["#2196F3", "#9E9E9E", "#FF9800", "#4CAF50"]

    for i, (_, row) in enumerate(null_table.iterrows()):
        if i >= len(axes):
            break
        ax = axes[i]
        means = [row["real_mean"], row["random_mean"],
                 row["matching_mean"], row["wsls_mean"]]
        sds = [row["real_sd"], row["random_sd"],
               row["matching_sd"], row["wsls_sd"]]
        positions = np.arange(len(process_labels))

        ax.bar(positions, means, yerr=sds, color=bar_colors,
               edgecolor="white", width=0.6, capsize=4, alpha=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(process_labels, fontsize=8)
        title = row["metric"].replace("_", " ").title()
        ax.set_title(title, fontsize=10)
        sns.despine(ax=ax)

    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Figure 15: Null Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure15_null_comparison.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 16 – Robustness Checks
# ═══════════════════════════════════════════════════════════════════════════

def _figure16_robustness(analysis_results, fig_dir, fmt, config):
    """Robustness checks: parameter sensitivity across analyses."""
    rob = analysis_results.get("robustness", {})
    if not rob:
        print("  Figure 16 skipped: no robustness results.")
        return

    rw_results = rob.get("rolling_window", {})
    dfa_results = rob.get("dfa_window", {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (A) DFA alpha vs rolling window
    ax = axes[0]
    if rw_results:
        windows = sorted(rw_results.keys())
        means = [rw_results[w]["dfa_alpha"].mean() for w in windows
                 if len(rw_results[w]) > 0]
        sds = [rw_results[w]["dfa_alpha"].std() for w in windows
               if len(rw_results[w]) > 0]
        valid_windows = [w for w in windows if len(rw_results[w]) > 0]
        if valid_windows:
            ax.errorbar(valid_windows, means, yerr=sds, fmt="o-",
                        color="steelblue", capsize=4, lw=1.5, markersize=6)
            default_win = config.get("rolling_window_clicks", 20)
            ax.axvline(default_win, color="red", ls="--", alpha=0.5,
                       label=f"Default ({default_win})")
            ax.legend(fontsize=8)
    ax.set_xlabel("Rolling Window Size (clicks)")
    ax.set_ylabel("DFA α")
    ax.set_title("A. DFA vs Rolling Window", fontsize=10,
                  fontweight="bold", loc="left")
    sns.despine(ax=ax)

    # (B) DFA cross-correlation heatmap
    ax = axes[1]
    if dfa_results and "pair_correlations" in dfa_results:
        records = dfa_results["records"]
        pair_corrs = dfa_results["pair_correlations"]
        min_windows = sorted(records.keys())
        n_mw = len(min_windows)
        corr_matrix = np.ones((n_mw, n_mw))
        for (w1, w2), r in pair_corrs.items():
            i1 = min_windows.index(w1)
            i2 = min_windows.index(w2)
            corr_matrix[i1, i2] = r
            corr_matrix[i2, i1] = r
        im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(n_mw))
        ax.set_xticklabels(min_windows)
        ax.set_yticks(range(n_mw))
        ax.set_yticklabels(min_windows)
        for i_idx in range(n_mw):
            for j_idx in range(n_mw):
                val = corr_matrix[i_idx, j_idx]
                if not np.isnan(val):
                    ax.text(j_idx, i_idx, f"{val:.2f}", ha="center",
                            va="center", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("DFA Min Window")
    ax.set_ylabel("DFA Min Window")
    ax.set_title("B. DFA Cross-Correlation", fontsize=10,
                  fontweight="bold", loc="left")

    fig.suptitle("Figure 16: Sensitivity and Robustness Checks",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure16_robustness.{fmt}"), config)


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
