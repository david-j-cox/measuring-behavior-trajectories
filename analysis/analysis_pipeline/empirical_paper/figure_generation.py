"""Generate figures for the empirical paper:
'Testing Behavior Dynamic Models to Account for Choice Trajectories
 in Non-Stationary Environments'

Produces 9 primary figures:
  1. Phase transitions in choice behavior (group rolling P(A))
  2. Model comparison (mean AIC, 14 models by family)
  3. Phase-transition dynamics (adaptation lags at boundaries)
  4. Adaptation lag distributions by phase
  5. Hysteresis in choice allocation (choice vs latent advantage)
  6. HMM state characteristics (choice x ICI state space)
  7. HMM phase alignment (state bands + phase boundaries)
  8. Convergent cross-mapping (CCM convergence)
  9. Recurrence quantification analysis (RQA summary)
"""

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
_PHASE_LABELS = {1: "Symmetric", 2: "A-advantage", 3: "B-advantage",
                 4: "Scarcity + Pulses"}

_MODEL_FAMILIES = {
    "random": "Baseline",
    "bias": "Baseline",
    "wsls": "Baseline",
    "generalized_matching_law": "Historical",
    "melioration": "Historical",
    "kinetic": "Historical",
    "behavioral_momentum": "Historical",
    "hill_climbing": "Historical",
    "q_learning": "RL",
    "q_dual_alpha": "RL",
    "q_forgetting": "RL",
    "hmm_2state": "HMM",
    "hmm_3state": "HMM",
    "hmm_4state": "HMM",
}

_MODEL_LABELS = {
    "random": "Random",
    "bias": "Bias Only",
    "wsls": "WSLS",
    "generalized_matching_law": "Gen. Matching Law",
    "melioration": "Melioration",
    "kinetic": "Kinetic",
    "behavioral_momentum": "Behav. Momentum",
    "hill_climbing": "Hill-Climbing",
    "q_learning": "Q-Learning",
    "q_dual_alpha": "Q-Learning (Dual \u03b1)",
    "q_forgetting": "Q-Learning (Forgetting)",
    "hmm_2state": "HMM (2-state)",
    "hmm_3state": "HMM (3-state)",
    "hmm_4state": "HMM (4-state)",
}

_FAMILY_COLORS = {
    "Baseline": "#bdbdbd",
    "Historical": "#fdae61",
    "RL": "#abd9e9",
    "HMM": "#2c7bb6",
}


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
            ax.axvline(pb["start_ms"] / 1000, color="red", ls="--",
                       alpha=0.6, lw=1.0)


def _save(fig, path, config):
    dpi = config.get("dpi", 300)
    fig.savefig(path, dpi=dpi)
    pdf_path = path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)


def _bootstrap_ci(data_by_bin, n_boot=1000, ci=95, rng=None):
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
    """Generate all 9 empirical paper figures."""
    _style()
    fig_dir = os.path.join(output_dir, "empirical_paper", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fmt = "png"
    n_boot = config.get("n_bootstrap", 1000)

    figure1_phase_transitions(events_df, fig_dir, fmt, config, n_boot)
    figure2_model_comparison(analysis_results, fig_dir, fmt, config)
    figure3_adaptation_lags(events_df, metrics_df, fig_dir, fmt, config)
    figure4_lag_distributions(events_df, metrics_df, fig_dir, fmt, config)
    figure5_hysteresis(events_df, fig_dir, fmt, config)
    figure6_hmm_state_characteristics(analysis_results, fig_dir, fmt, config)
    figure7_hmm_phase_alignment(analysis_results, events_df, fig_dir, fmt,
                                config)
    figure8_ccm(analysis_results, fig_dir, fmt, config)
    figure9_rqa(analysis_results, events_df, fig_dir, fmt, config)

    print(f"  Empirical paper figures saved to {fig_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 – Phase Transitions in Choice Behavior
# ═══════════════════════════════════════════════════════════════════════════

def figure1_phase_transitions(events_df, fig_dir, fmt, config, n_boot):
    """Rolling choice proportion across the four phases, group mean + CI."""
    rng = np.random.default_rng(0)
    bin_width = 2  # seconds

    fig, ax = plt.subplots(figsize=(10, 4.5))

    bins_data = {}
    for sid, sdf in events_df.groupby("session_id"):
        sdf = sdf.sort_values("timestamp_ms")
        if "rolling_choice_prop_a_clicks" not in sdf.columns:
            continue
        sdf = sdf.dropna(subset=["rolling_choice_prop_a_clicks"])
        sdf["tb"] = (sdf["elapsed_time_s"] // bin_width * bin_width).astype(int)
        for tb, grp in sdf.groupby("tb"):
            bins_data.setdefault(tb, []).append(
                grp["rolling_choice_prop_a_clicks"].mean())

        # Individual traces (thin, transparent)
        indiv = sdf.groupby("tb")["rolling_choice_prop_a_clicks"].mean()
        ax.plot(indiv.index, indiv.values, color="gray", alpha=0.08, lw=0.4)

    # Group mean + CI
    x, m, lo, hi = _bootstrap_ci(bins_data, n_boot=n_boot, rng=rng)
    ax.plot(x, m, color="#2166AC", lw=2.5, zorder=5)
    ax.fill_between(x, lo, hi, alpha=0.20, color="#2166AC", zorder=4)

    # Phase boundaries and labels
    _phase_lines(ax, config)
    for pb in config.get("phase_boundaries", []):
        mid = (pb["start_ms"] + pb["end_ms"]) / 2000
        label = _PHASE_LABELS.get(pb["id"], f"Phase {pb['id']}")
        ax.text(mid, 0.97, label, ha="center", va="top",
                fontsize=8, color=_PHASE_COLORS.get(pb["id"], "gray"),
                fontweight="bold", transform=ax.get_xaxis_transform())

    ax.axhline(0.5, color="gray", ls=":", alpha=0.3, lw=0.6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rolling P(Choose A)")
    sns.despine(ax=ax)

    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure1_phase_transitions.{fmt}"),
          config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 – Model Comparison
# ═══════════════════════════════════════════════════════════════════════════

def figure2_model_comparison(analysis_results, fig_dir, fmt, config):
    """Mean AIC for all 14 models, organized by family, with broken axis."""
    mc = analysis_results.get("model_comparison")
    if mc is None or len(mc) == 0:
        print("  Figure 2 skipped: no model comparison results.")
        return

    mc = mc.copy()
    mc["family"] = mc["model"].map(_MODEL_FAMILIES)
    mc["label"] = mc["model"].map(
        lambda m: _MODEL_LABELS.get(m, m.replace("_", " ").title()))

    # Keep only the 14 empirical paper models
    keep = set(_MODEL_FAMILIES.keys())
    mc = mc[mc["model"].isin(keep)]

    agg = (mc.groupby(["label", "family"])["aic"]
           .agg(["mean", "sem"])
           .reset_index()
           .sort_values("mean"))

    # Broken y-axis: HMMs have negative AIC far from process models
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 2], "hspace": 0.08})

    for ax in (ax_top, ax_bot):
        bars = ax.barh(
            range(len(agg)), agg["mean"].values, xerr=agg["sem"].values,
            color=[_FAMILY_COLORS.get(f, "gray") for f in agg["family"]],
            edgecolor="white", height=0.7, capsize=3)
        ax.set_yticks(range(len(agg)))
        ax.set_yticklabels(agg["label"].values, fontsize=9)

    # Set axis limits for the break
    hmm_min = agg[agg["family"] == "HMM"]["mean"].min()
    hmm_max = agg[agg["family"] == "HMM"]["mean"].max()
    process_min = agg[agg["family"] != "HMM"]["mean"].min()
    process_max = agg[agg["family"] != "HMM"]["mean"].max()

    if hmm_min < 0:
        ax_top.set_xlim(process_min - 100, process_max + 100)
        ax_bot.set_xlim(hmm_min - 500, hmm_max + 500)
    else:
        # No break needed if all positive
        ax_top.set_xlim(agg["mean"].min() - 100, agg["mean"].max() + 100)
        ax_bot.set_xlim(agg["mean"].min() - 100, agg["mean"].max() + 100)

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False)

    # Break indicators
    d = 0.015
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, lw=1)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs["transform"] = ax_bot.transAxes
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    ax_bot.set_xlabel("Mean AIC (lower = better fit)")

    # Family legend
    handles = [mpatches.Patch(color=c, label=f)
               for f, c in _FAMILY_COLORS.items()]
    ax_top.legend(handles=handles, loc="upper right", fontsize=8,
                  frameon=True, edgecolor="lightgray")

    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure2_model_comparison.{fmt}"),
          config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 – Phase-Transition Dynamics (Adaptation Lags)
# ═══════════════════════════════════════════════════════════════════════════

def figure3_adaptation_lags(events_df, metrics_df, fig_dir, fmt, config):
    """Adaptation lags at each phase boundary: time to criterion crossing."""
    pbs = config.get("phase_boundaries", [])
    if len(pbs) < 2:
        return

    window = config.get("rolling_window_clicks", 20)
    threshold = 0.5  # criterion: when rolling P(A) crosses 0.5

    transitions = []
    for i in range(1, len(pbs)):
        transitions.append({
            "from": pbs[i-1]["id"],
            "to": pbs[i]["id"],
            "t_s": pbs[i]["start_ms"] / 1000,
            "label": (f"{_PHASE_LABELS.get(pbs[i-1]['id'], '')} \u2192 "
                      f"{_PHASE_LABELS.get(pbs[i]['id'], '')}"),
        })

    fig, axes = plt.subplots(1, len(transitions),
                             figsize=(4.5 * len(transitions), 4),
                             sharey=True)
    if len(transitions) == 1:
        axes = [axes]

    for col, tr in enumerate(transitions):
        ax = axes[col]
        lags = []
        for sid, sdf in events_df.groupby("session_id"):
            sdf = sdf.sort_values("timestamp_ms")
            if "rolling_choice_prop_a_clicks" not in sdf.columns:
                continue

            post = sdf[sdf["elapsed_time_s"] >= tr["t_s"]].copy()
            if len(post) < 5:
                continue

            signal = post["rolling_choice_prop_a_clicks"].values
            times = post["elapsed_time_s"].values

            # Determine expected direction
            if tr["to"] in (2,):  # A-advantage: expect P(A) > 0.5
                crossed = np.where(signal > threshold)[0]
            elif tr["to"] in (3,):  # B-advantage: expect P(A) < 0.5
                crossed = np.where(signal < threshold)[0]
            else:  # Phase 4 scarcity: expect return toward 0.5
                crossed = np.where(np.abs(signal - 0.5) < 0.15)[0]

            if len(crossed) > 0:
                lag = times[crossed[0]] - tr["t_s"]
                lags.append(lag)

        if lags:
            lags = np.array(lags)
            parts = ax.violinplot([lags], positions=[0], showmeans=True,
                                  showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(_PHASE_COLORS.get(tr["to"], "steelblue"))
                pc.set_alpha(0.6)
            parts["cmeans"].set_color("black")

            # Overlay individual points
            jitter = np.random.default_rng(col).uniform(-0.15, 0.15,
                                                         len(lags))
            ax.scatter(jitter, lags, s=20, alpha=0.5,
                       color=_PHASE_COLORS.get(tr["to"], "steelblue"),
                       edgecolors="white", linewidth=0.5, zorder=3)

            ax.text(0, ax.get_ylim()[1] * 0.95,
                    f"M = {np.mean(lags):.1f}s\nMd = {np.median(lags):.1f}s",
                    ha="center", va="top", fontsize=8)

        ax.set_title(tr["label"], fontsize=9, fontweight="bold")
        ax.set_xticks([])
        if col == 0:
            ax.set_ylabel("Adaptation Lag (s)")
        sns.despine(ax=ax, bottom=True)

    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure3_adaptation_lags.{fmt}"),
          config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 – Adaptation Lag Distributions by Phase
# ═══════════════════════════════════════════════════════════════════════════

def figure4_lag_distributions(events_df, metrics_df, fig_dir, fmt, config):
    """Histograms of adaptation lags showing decrease across transitions."""
    pbs = config.get("phase_boundaries", [])
    if len(pbs) < 2:
        return

    threshold = 0.5
    transition_lags = {}

    for i in range(1, len(pbs)):
        from_id = pbs[i-1]["id"]
        to_id = pbs[i]["id"]
        t_s = pbs[i]["start_ms"] / 1000
        label = (f"Phase {from_id} \u2192 {to_id}")
        lags = []

        for sid, sdf in events_df.groupby("session_id"):
            sdf = sdf.sort_values("timestamp_ms")
            if "rolling_choice_prop_a_clicks" not in sdf.columns:
                continue

            post = sdf[sdf["elapsed_time_s"] >= t_s]
            if len(post) < 5:
                continue

            signal = post["rolling_choice_prop_a_clicks"].values
            times = post["elapsed_time_s"].values

            if to_id == 2:
                crossed = np.where(signal > threshold)[0]
            elif to_id == 3:
                crossed = np.where(signal < threshold)[0]
            else:
                crossed = np.where(np.abs(signal - 0.5) < 0.15)[0]

            if len(crossed) > 0:
                lags.append(times[crossed[0]] - t_s)

        transition_lags[label] = np.array(lags) if lags else np.array([])

    n_trans = len(transition_lags)
    fig, axes = plt.subplots(1, n_trans, figsize=(4 * n_trans, 3.5),
                             sharey=True, sharex=True)
    if n_trans == 1:
        axes = [axes]

    colors = ["#999999", "#4393C3", "#D6604D", "#5AAE61"]
    for idx, (label, lags) in enumerate(transition_lags.items()):
        ax = axes[idx]
        if len(lags) > 0:
            ax.hist(lags, bins=12, color=colors[idx % len(colors)],
                    edgecolor="white", alpha=0.8)
            ax.axvline(np.median(lags), color="black", ls="--", lw=1.5,
                       label=f"Median = {np.median(lags):.1f}s")
            ax.legend(fontsize=7)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Adaptation Lag (s)")
        if idx == 0:
            ax.set_ylabel("Count")
        sns.despine(ax=ax)

    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure4_lag_distributions.{fmt}"),
          config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 – Hysteresis in Choice Allocation
# ═══════════════════════════════════════════════════════════════════════════

def figure5_hysteresis(events_df, fig_dir, fmt, config):
    """Choice proportion vs latent advantage, showing path dependence."""
    pbs = config.get("phase_boundaries", [])
    if len(pbs) < 3:
        return

    # Phase 2: A advantage rising; Phase 3: B advantage (A falling)
    phase2_start = pbs[1]["start_ms"] / 1000
    phase2_end = pbs[1]["end_ms"] / 1000
    phase3_start = pbs[2]["start_ms"] / 1000
    phase3_end = pbs[2]["end_ms"] / 1000

    bin_width = 3  # seconds

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Collect per-bin data across sessions
    phase2_bins = {}
    phase3_bins = {}

    for sid, sdf in events_df.groupby("session_id"):
        sdf = sdf.sort_values("timestamp_ms")
        if "rolling_choice_prop_a_clicks" not in sdf.columns:
            continue

        # Phase 2
        p2 = sdf[(sdf["elapsed_time_s"] >= phase2_start) &
                  (sdf["elapsed_time_s"] < phase2_end)].copy()
        if len(p2) > 5:
            p2["tb"] = ((p2["elapsed_time_s"] - phase2_start) // bin_width
                        ).astype(int)
            for tb, grp in p2.groupby("tb"):
                # Normalize time bin to [0, 1] range within phase
                norm_t = tb * bin_width / (phase2_end - phase2_start)
                phase2_bins.setdefault(tb, []).append(
                    grp["rolling_choice_prop_a_clicks"].mean())

        # Phase 3
        p3 = sdf[(sdf["elapsed_time_s"] >= phase3_start) &
                  (sdf["elapsed_time_s"] < phase3_end)].copy()
        if len(p3) > 5:
            p3["tb"] = ((p3["elapsed_time_s"] - phase3_start) // bin_width
                        ).astype(int)
            for tb, grp in p3.groupby("tb"):
                phase3_bins.setdefault(tb, []).append(
                    grp["rolling_choice_prop_a_clicks"].mean())

    # Compute latent advantage proxy (time within phase as proxy)
    # Phase 2: advantage grows from 0 to max; Phase 3: advantage reverses
    if phase2_bins and phase3_bins:
        max_bin = max(max(phase2_bins.keys()), max(phase3_bins.keys()))

        # Phase 2: A-advantage increasing (plot left to right)
        p2_t = sorted(phase2_bins.keys())
        p2_advantage = [b * bin_width / (phase2_end - phase2_start)
                        for b in p2_t]
        p2_choice = [np.mean(phase2_bins[b]) for b in p2_t]

        # Phase 3: A-advantage decreasing (B takes over)
        p3_t = sorted(phase3_bins.keys())
        # Reverse the x-axis for phase 3 to show returning path
        p3_advantage = [1.0 - b * bin_width / (phase3_end - phase3_start)
                        for b in p3_t]
        p3_choice = [np.mean(phase3_bins[b]) for b in p3_t]

        ax.plot(p2_advantage, p2_choice, "o-", color="#4393C3", lw=2,
                markersize=5, label="Phase 2 (A rising)", zorder=5)
        ax.plot(p3_advantage, p3_choice, "s-", color="#D6604D", lw=2,
                markersize=5, label="Phase 3 (B rising)", zorder=5)

        # Arrows to show direction
        if len(p2_advantage) > 2:
            mid = len(p2_advantage) // 2
            ax.annotate("", xy=(p2_advantage[mid+1], p2_choice[mid+1]),
                        xytext=(p2_advantage[mid], p2_choice[mid]),
                        arrowprops=dict(arrowstyle="->", color="#4393C3",
                                        lw=2))
        if len(p3_advantage) > 2:
            mid = len(p3_advantage) // 2
            ax.annotate("", xy=(p3_advantage[mid+1], p3_choice[mid+1]),
                        xytext=(p3_advantage[mid], p3_choice[mid]),
                        arrowprops=dict(arrowstyle="->", color="#D6604D",
                                        lw=2))

    ax.axhline(0.5, color="gray", ls=":", alpha=0.3)
    ax.set_xlabel("Relative Advantage for Option A")
    ax.set_ylabel("P(Choose A)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure5_hysteresis.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6 – HMM State Characteristics
# ═══════════════════════════════════════════════════════════════════════════

def figure6_hmm_state_characteristics(analysis_results, fig_dir, fmt, config):
    """State-space scatter of HMM states in choice x ICI space."""
    hmm_seqs = analysis_results.get("hmm_state_sequences", {})
    hmm_summary = analysis_results.get("hmm_state_summary")

    if not hmm_seqs and (hmm_summary is None or len(hmm_summary) == 0):
        print("  Figure 6 skipped: no HMM results.")
        return

    state_colors = ["#4393C3", "#D6604D", "#5AAE61", "#984EA3"]
    state_labels_map = {0: "Exploiting B", 1: "Exploring/Switching",
                        2: "Exploiting A", 3: "Moderate Bias"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (A) Scatter plot of mean_choice_a vs mean_reward by state label
    ax = axes[0]
    if hmm_summary is not None and len(hmm_summary) > 0:
        per_session = hmm_summary[hmm_summary["session_id"] != "GROUP_MEAN"]
        for label_name in per_session["label"].unique():
            subset = per_session[per_session["label"] == label_name]
            color_idx = (0 if "B" in label_name else
                         2 if "A" in label_name else 1)
            ax.scatter(subset["mean_choice_a"], subset["mean_reward"],
                       s=30, alpha=0.5,
                       color=state_colors[color_idx % len(state_colors)],
                       label=label_name, edgecolors="white", linewidth=0.5)

        ax.set_xlabel("Mean P(Choose A)")
        ax.set_ylabel("Mean Reward Rate")
        ax.legend(fontsize=8)
    sns.despine(ax=ax)
    ax.set_title("A. HMM States in Choice \u00d7 Reward Space",
                 fontsize=10, fontweight="bold", loc="left")

    # (B) Group-average emission means per state label
    ax = axes[1]
    if hmm_summary is not None and len(hmm_summary) > 0:
        group_rows = hmm_summary[hmm_summary["session_id"] == "GROUP_MEAN"]
        if len(group_rows) > 0:
            labels = group_rows["label"].values
            choice_means = group_rows["mean_choice_a"].values
            x = np.arange(len(labels))
            bar_colors = []
            for lbl in labels:
                if "B" in lbl:
                    bar_colors.append(state_colors[0])
                elif "A" in lbl:
                    bar_colors.append(state_colors[2])
                else:
                    bar_colors.append(state_colors[1])

            ax.bar(x, choice_means, color=bar_colors, edgecolor="white",
                   width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
            ax.set_ylabel("Mean P(Choose A)")
            ax.axhline(0.5, color="gray", ls=":", alpha=0.3)
    ax.set_title("B. Group-Average State Profiles",
                 fontsize=10, fontweight="bold", loc="left")
    sns.despine(ax=ax)

    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure6_hmm_states.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7 – HMM Phase Alignment
# ═══════════════════════════════════════════════════════════════════════════

def figure7_hmm_phase_alignment(analysis_results, events_df, fig_dir, fmt,
                                config):
    """State assignments as colored bands aligned with phase boundaries."""
    hmm_seqs = analysis_results.get("hmm_state_sequences", {})
    if not hmm_seqs:
        print("  Figure 7 skipped: no HMM state sequences.")
        return

    state_colors = ["#4393C3", "#D6604D", "#5AAE61", "#984EA3"]
    sids = list(hmm_seqs.keys())

    # Show up to 6 representative sessions
    n_show = min(6, len(sids))
    step = max(1, len(sids) // n_show)
    show_sids = sids[::step][:n_show]

    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.5 * n_show),
                             sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, sid in enumerate(show_sids):
        ax = axes[i]
        sdf = events_df[events_df["session_id"] == sid].sort_values(
            "timestamp_ms")
        seq = hmm_seqs[sid]
        states = seq["states"]
        n_states = seq["n_states"]

        times = sdf["elapsed_time_s"].values[:len(states)]

        # Plot rolling choice proportion
        if "rolling_choice_prop_a_clicks" in sdf.columns:
            signal = sdf["rolling_choice_prop_a_clicks"].values[:len(states)]
            ax.plot(times, signal, color="black", lw=0.8, alpha=0.6)

        # Color background by state
        for t_idx in range(len(times) - 1):
            s = states[t_idx]
            ax.axvspan(times[t_idx], times[t_idx + 1],
                       color=state_colors[s % len(state_colors)],
                       alpha=0.25, linewidth=0)

        _phase_lines(ax, config)
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(A)", fontsize=8)
        label = f"{sid[:10]}..." if len(str(sid)) > 10 else sid
        ax.text(0.01, 0.95, label, transform=ax.transAxes, fontsize=7,
                va="top", color="gray")
        sns.despine(ax=ax)

    axes[-1].set_xlabel("Time (s)")

    # Legend
    handles = [mpatches.Patch(color=state_colors[s], alpha=0.5,
                              label=f"State {s}")
               for s in range(min(4, max(seq["n_states"]
                                         for seq in hmm_seqs.values())))]
    handles.append(Line2D([0], [0], color="red", ls="--", lw=1,
                          label="Phase boundary"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=True,
               edgecolor="lightgray")

    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure7_hmm_phase_alignment.{fmt}"),
          config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8 – Convergent Cross-Mapping
# ═══════════════════════════════════════════════════════════════════════════

def figure8_ccm(analysis_results, fig_dir, fmt, config):
    """CCM convergence: reward -> choice vs choice -> reward."""
    ccm_df = analysis_results.get("ccm")
    if ccm_df is None or len(ccm_df) == 0:
        print("  Figure 8 skipped: no CCM results.")
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
            lw=2, markersize=5, label="Reward \u2192 Choice")
    ax.fill_between(group["lib_size"],
                    group["rc_mean"] - group["rc_sem"],
                    group["rc_mean"] + group["rc_sem"],
                    alpha=0.15, color="#4393C3")
    ax.plot(group["lib_size"], group["cr_mean"], "s-", color="#D6604D",
            lw=2, markersize=5, label="Choice \u2192 Reward")
    ax.fill_between(group["lib_size"],
                    group["cr_mean"] - group["cr_sem"],
                    group["cr_mean"] + group["cr_sem"],
                    alpha=0.15, color="#D6604D")

    ax.set_xlabel("Library Size")
    ax.set_ylabel("Cross-Mapping \u03c1")
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
        ax.scatter(cr, rc, s=40, alpha=0.7, color="steelblue",
                   edgecolors="white")
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "--", color="gray", alpha=0.5)
        ax.set_xlabel("Choice \u2192 Reward \u03c1")
        ax.set_ylabel("Reward \u2192 Choice \u03c1")
    ax.set_title("B. Per-Session Asymmetry", fontsize=10,
                 fontweight="bold", loc="left")
    sns.despine(ax=ax)

    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure8_ccm.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 9 – Recurrence Quantification Analysis
# ═══════════════════════════════════════════════════════════════════════════

def figure9_rqa(analysis_results, events_df, fig_dir, fmt, config):
    """RQA: (A) example recurrence plot, (B) metric distributions,
    (C) determinism by phase."""
    rqa_df = analysis_results.get("rqa")
    if rqa_df is None or len(rqa_df) == 0:
        print("  Figure 9 skipped: no RQA results.")
        return

    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.2, 1])

    # (A) Example recurrence plot
    ax_rp = fig.add_subplot(gs[0])
    sids = events_df["session_id"].unique()
    if len(sids) > 0 and "rolling_choice_prop_a_clicks" in events_df.columns:
        sid = sids[len(sids) // 2]
        sdf = events_df[events_df["session_id"] == sid].sort_values(
            "timestamp_ms")
        signal = sdf["rolling_choice_prop_a_clicks"].dropna().values
        if len(signal) > 300:
            signal = signal[::len(signal) // 300]
        dist = np.abs(signal[:, None] - signal[None, :])
        recurrence = (dist < 0.1).astype(int)
        np.fill_diagonal(recurrence, 0)
        ax_rp.imshow(recurrence, cmap="Greys", origin="lower",
                     aspect="equal")
        ax_rp.set_xlabel("Click index")
        ax_rp.set_ylabel("Click index")
    ax_rp.set_title("A. Example Recurrence Plot", fontsize=10,
                    fontweight="bold", loc="left")

    # (B) Metric distributions
    ax_dist = fig.add_subplot(gs[1])
    full = rqa_df[rqa_df["scope"] == "full_session"] if "scope" in rqa_df.columns else rqa_df
    metrics = ["determinism", "recurrence_rate", "laminarity"]
    avail = [m for m in metrics if m in full.columns]
    labels = {"determinism": "Determinism", "recurrence_rate": "Recurrence Rate",
              "laminarity": "Laminarity"}
    colors = ["#4393C3", "#D6604D", "#5AAE61"]

    if avail:
        parts = ax_dist.violinplot(
            [full[m].dropna().values for m in avail],
            positions=np.arange(len(avail)), showmeans=True,
            showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.6)
        parts["cmeans"].set_color("black")
        ax_dist.set_xticks(np.arange(len(avail)))
        ax_dist.set_xticklabels([labels.get(m, m) for m in avail], fontsize=9)
    ax_dist.set_ylabel("Value")
    ax_dist.set_title("B. RQA Metric Distributions", fontsize=10,
                      fontweight="bold", loc="left")
    sns.despine(ax=ax_dist)

    # (C) Determinism by phase
    ax_phase = fig.add_subplot(gs[2])
    phase_col = "phase_id" if "phase_id" in rqa_df.columns else None
    if phase_col and "determinism" in rqa_df.columns:
        phase_data = rqa_df[rqa_df[phase_col] > 0].copy()
        if len(phase_data) > 0:
            for pid in sorted(phase_data[phase_col].unique()):
                color = _PHASE_COLORS.get(pid, "gray")
                vals = phase_data.loc[phase_data[phase_col] == pid,
                                      "determinism"].dropna()
                if len(vals) == 0:
                    continue
                ax_phase.boxplot(
                    [vals.values], positions=[pid], widths=0.5,
                    patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.5),
                    medianprops=dict(color="black"),
                    showfliers=False)
                jitter = np.random.default_rng(pid).uniform(
                    -0.12, 0.12, len(vals))
                ax_phase.scatter(pid + jitter, vals.values, s=15, alpha=0.4,
                                 color=color, zorder=3)
            ax_phase.set_xticks([1, 2, 3, 4])
            ax_phase.set_xticklabels(
                ["Ph 1\nSymmetric", "Ph 2\nA-adv",
                 "Ph 3\nB-adv", "Ph 4\nScarcity"], fontsize=8)
    ax_phase.set_ylabel("Determinism")
    ax_phase.set_title("C. Determinism by Phase", fontsize=10,
                       fontweight="bold", loc="left")
    sns.despine(ax=ax_phase)

    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure9_rqa.{fmt}"), config)
