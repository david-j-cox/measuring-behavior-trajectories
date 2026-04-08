"""Generate figures for the empirical paper:
'Testing Behavior Dynamic Models to Account for Choice Trajectories
 in Non-Stationary Environments'

Produces 8 primary figures:
  1. Phase transitions in choice behavior (group rolling P(A))
  2. Model comparison (mean AIC, 14 models by family)
  3. Phase-transition dynamics (adaptation lags at boundaries)
  4. Hysteresis in choice allocation (choice vs latent advantage)
  5. HMM state characteristics (choice x ICI state space)
  6. HMM phase alignment (state bands + phase boundaries)
  7. Convergent cross-mapping (CCM convergence)
  8. Recurrence quantification analysis (RQA summary)
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
# Shared helpers & constants
# ═══════════════════════════════════════════════════════════════════════════

# Grayscale palette for JEAB black-and-white publication
_PHASE_GRAYS = {1: "#AAAAAA", 2: "#555555", 3: "#888888", 4: "#333333"}
_PHASE_HATCHES = {1: "", 2: "//", 3: "\\\\", 4: "xx"}
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

_FAMILY_GRAYS = {
    "Baseline": "#CCCCCC",
    "Historical": "#999999",
    "RL": "#666666",
    "HMM": "#333333",
}
_FAMILY_HATCHES = {
    "Baseline": "",
    "Historical": "//",
    "RL": "\\\\",
    "HMM": "xx",
}

# Font sizes — sized for JEAB publication (figures may be scaled down)
_LABEL_FS = 26       # axis labels
_TICK_FS = 14        # tick labels
_TITLE_FS = 18       # subplot titles
_CONDITION_FS = 14   # phase/condition annotations (= tick size)
_LEGEND_FS = 12      # legend text
_ANNOTATION_FS = 14  # in-plot annotations
_LABELPAD = 12       # padding between axis and label

# Color palette for Figure 2 (model comparison) where color aids readability
_FAMILY_COLORS = {
    "Baseline": "#bdbdbd",
    "Historical": "#fdae61",
    "RL": "#abd9e9",
    "HMM": "#2c7bb6",
}

# HMM state grays + markers for grayscale differentiation
_STATE_GRAYS = ["#333333", "#999999", "#666666", "#BBBBBB"]
_STATE_MARKERS = ["o", "s", "^", "D"]
_STATE_HATCHES = ["", "//", "\\\\", "xx"]


def _style():
    sns.set_theme(style="ticks", font_scale=1.15)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "xtick.labelsize": _TICK_FS,
        "ytick.labelsize": _TICK_FS,
    })


def _phase_lines(ax, config):
    for pb in config.get("phase_boundaries", []):
        if pb["start_ms"] > 0:
            ax.axvline(pb["start_ms"] / 1000, color="black", ls="--",
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
                         output_dir: str, only: list = None,
                         figures_dir: str = None):
    """Generate empirical paper figures.

    Parameters
    ----------
    only : list of int, optional
        If provided, generate only these figure numbers (e.g. [5, 7]).
        If None, generate all figures.
    figures_dir : str, optional
        Direct path to save figures. If None, falls back to
        output_dir/figures/figures for backward compatibility.
    """
    _style()
    if figures_dir is not None:
        fig_dir = figures_dir
    else:
        fig_dir = os.path.join(output_dir, "figures", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fmt = "png"
    n_boot = config.get("n_bootstrap", 1000)

    def _run(n):
        return only is None or n in only

    if _run(1):
        figure1_phase_transitions(events_df, fig_dir, fmt, config, n_boot)
    if _run(2):
        figure2_model_comparison(analysis_results, fig_dir, fmt, config)
    if _run(3):
        figure3_adaptation_lags(events_df, metrics_df, fig_dir, fmt, config)
    if _run(4):
        figure4_hysteresis(events_df, fig_dir, fmt, config)
        figure4_hysteresis_supplement(events_df, fig_dir, fmt, config)
    if _run(5):
        figure5_hmm_state_characteristics(analysis_results, fig_dir, fmt,
                                          config)
    if _run(6):
        figure6_hmm_phase_alignment(analysis_results, events_df, fig_dir, fmt,
                                    config)
    if _run(7):
        figure7_ccm(analysis_results, fig_dir, fmt, config)
    if _run(8):
        figure8_rqa(analysis_results, events_df, fig_dir, fmt, config)

    which = f"figure(s) {only}" if only else "all figures"
    print(f"  {which} saved to {fig_dir}")


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
    ax.plot(x, m, color="black", lw=2.5, zorder=5)
    ax.fill_between(x, lo, hi, alpha=0.20, color="gray", zorder=4)

    # Phase boundaries and labels
    _phase_lines(ax, config)
    for pb in config.get("phase_boundaries", []):
        mid = (pb["start_ms"] + pb["end_ms"]) / 2000
        label = _PHASE_LABELS.get(pb["id"], f"Phase {pb['id']}")
        ax.text(mid, 0.97, label, ha="center", va="top",
                fontsize=_CONDITION_FS, color="black",
                fontweight="bold", transform=ax.get_xaxis_transform())

    ax.axhline(0.5, color="gray", ls=":", alpha=0.3, lw=0.6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)", fontsize=_LABEL_FS, labelpad=_LABELPAD)
    ax.set_ylabel("Rolling P(Choose A)", fontsize=_LABEL_FS,
                  labelpad=_LABELPAD)
    sns.despine(ax=ax)

    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure1_phase_transitions.{fmt}"),
          config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 – Model Comparison
# ═══════════════════════════════════════════════════════════════════════════

def figure2_model_comparison(analysis_results, fig_dir, fmt, config):
    """Mean AIC for all 14 models, sorted best (top) to worst (bottom)."""
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

    # Build label order mapping for swarmplot alignment
    label_order = list(agg["label"].values)

    # Single axis, sorted best (top) to worst (bottom)
    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = range(len(agg))
    bars = ax.barh(
        y_pos, agg["mean"].values, xerr=agg["sem"].values,
        color=[_FAMILY_COLORS.get(f, "gray") for f in agg["family"]],
        edgecolor="white", height=0.7, capsize=3,
        error_kw={"lw": 1.2, "capthick": 1.2})

    # Overlay individual-participant AIC values
    for idx, row in agg.iterrows():
        model_label = row["label"]
        y = label_order.index(model_label)
        indiv_aic = mc.loc[mc["label"] == model_label, "aic"].values
        if len(indiv_aic) > 0:
            jitter = np.random.default_rng(y).uniform(
                -0.25, 0.25, len(indiv_aic))
            ax.scatter(indiv_aic, y + jitter, s=8, alpha=0.4,
                       color="black", edgecolors="none", zorder=4)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(agg["label"].values, fontsize=_TICK_FS)
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("Mean AIC (lower = better fit)", fontsize=_LABEL_FS,
                  labelpad=_LABELPAD)
    sns.despine(ax=ax, right=True, top=True)

    # Family legend
    handles = [mpatches.Patch(facecolor=_FAMILY_COLORS[f],
                              edgecolor="gray", label=f)
               for f in _FAMILY_COLORS]
    handles.append(Line2D([0], [0], marker="o", color="none",
                          markerfacecolor="black", markersize=4,
                          alpha=0.5, label="Individual"))
    ax.legend(handles=handles, loc="lower right", fontsize=_LEGEND_FS,
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
                             figsize=(4.5 * len(transitions), 6),
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
            gray = _PHASE_GRAYS.get(tr["to"], "#888888")
            parts = ax.violinplot([lags], positions=[0], showmeans=True,
                                  showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(gray)
                pc.set_alpha(0.6)
            parts["cmeans"].set_color("black")

            # Overlay individual points
            jitter = np.random.default_rng(col).uniform(-0.15, 0.15,
                                                         len(lags))
            ax.scatter(jitter, lags, s=20, alpha=0.5, color=gray,
                       edgecolors="white", linewidth=0.5, zorder=3)

        ax.set_title(tr["label"], fontsize=_TITLE_FS, fontweight="bold")
        ax.set_xticks([])
        if col == 0:
            ax.set_ylabel("Adaptation Lag (s)", fontsize=_LABEL_FS + 4,
                          labelpad=_LABELPAD)
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

    phase_ids = [pbs[i]["id"] for i in range(1, len(pbs))]
    for idx, (label, lags) in enumerate(transition_lags.items()):
        ax = axes[idx]
        pid = phase_ids[idx] if idx < len(phase_ids) else 1
        if len(lags) > 0:
            ax.hist(lags, bins=12, color="black", edgecolor="white",
                    alpha=0.8)
            ax.axvline(np.median(lags), color="black", ls="--", lw=1.5,
                       label=f"Median = {np.median(lags):.1f}s")
            ax.legend(fontsize=_LEGEND_FS)
        ax.set_title(label, fontsize=_TITLE_FS, fontweight="bold")
        ax.set_xlabel("Adaptation Lag (s)", fontsize=_TICK_FS,
                      labelpad=_LABELPAD)
        if idx == 0:
            ax.set_ylabel("Count", fontsize=_LABEL_FS, labelpad=_LABELPAD)
        sns.despine(ax=ax)

    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure4_lag_distributions.{fmt}"),
          config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 – Hysteresis in Choice Allocation
# ═══════════════════════════════════════════════════════════════════════════

def figure4_hysteresis(events_df, fig_dir, fmt, config):
    """Continuous trajectory through all phases showing hysteresis loops.

    Two subplots:
      A. x = latent advantage (V_A - V_B)
      B. x = local reward-rate difference (R_A - R_B)
    The trajectory is binned by time, producing a continuous path with
    arrows showing temporal direction. If equilibrium-seeking models are
    correct, overlapping x-regions should produce identical y-values
    regardless of phase; vertical separation at the same x indicates
    path dependence (hysteresis).
    """
    pbs = config.get("phase_boundaries", [])
    if len(pbs) < 2:
        return

    bin_width = 10   # seconds
    rolling = True   # True = rolling window, False = non-overlapping bins
    bin_step = 1     # step size in seconds for rolling window

    x_specs = [
        {
            "col": "latent_advantage_pre",
            "xlabel": "$V_A - V_B$",
            "title": "A. Latent Value",
        },
        {
            "col_a": "local_reward_rate_a",
            "col_b": "local_reward_rate_b",
            "xlabel": "$R_A - R_B$",
            "title": "B. Local Reinforcement Rate",
        },
    ]

    # Colored phase styles with arrow-line rendering
    _phase_styles = {
        1: {"color": "#999999", "label": "Phase 1 (Symmetric)"},
        2: {"color": "#4393C3", "label": "Phase 2 (A rising)"},
        3: {"color": "#D6604D", "label": "Phase 3 (B rising)"},
        4: {"color": "#5AAE61", "label": "Phase 4 (Scarcity)"},
    }

    # Right-column specs: normalized relative to the advantaged option
    x_specs_norm = [
        {"col": "latent_advantage_pre",
         "xlabel": "$V_{Higher} - V_{Lower}$",
         "title": "C. Latent Value (Normalized)"},
        {"col_a": "local_reward_rate_a", "col_b": "local_reward_rate_b",
         "xlabel": "$R_{Higher} - R_{Lower}$",
         "title": "D. Local Reinforcement Rate (Normalized)"},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.2)

    # --- Helper to compute time-binned trajectories for one phase ---
    def _phase_trajectory(sdf, spec, t_start, t_end, normalize_phase=None):
        """Return (mean_x, mean_y) arrays for one session-phase.

        If normalize_phase is set (2 or 3), flip axes so that:
          x = advantage for the higher-payout option (always positive)
          y = P(choose higher-payout option)
        Phase 2: A is higher → no flip needed
        Phase 3: B is higher → flip both
        """
        if "rolling_choice_prop_a_clicks" not in sdf.columns:
            return None, None

        if "col" in spec:
            if spec["col"] not in sdf.columns:
                return None, None
            adv = sdf[spec["col"]]
        else:
            if (spec["col_a"] not in sdf.columns or
                    spec["col_b"] not in sdf.columns):
                return None, None
            adv = sdf[spec["col_a"]] - sdf[spec["col_b"]]

        choice = sdf["rolling_choice_prop_a_clicks"]
        mask = ((sdf["elapsed_time_s"] >= t_start) &
                (sdf["elapsed_time_s"] < t_end) &
                adv.notna() & choice.notna())
        phase_df = sdf[mask]
        if len(phase_df) < 5:
            return None, None

        phase_df = phase_df.copy()
        elapsed = phase_df["elapsed_time_s"].values

        adv_vals = adv.copy()
        choice_vals = choice.copy()

        # Normalize: flip so x = higher - lower, y = P(choose higher)
        if normalize_phase == 3:
            adv_vals = -adv_vals          # B is higher, so flip sign
            choice_vals = 1 - choice_vals  # P(choose B) = 1 - P(choose A)

        adv_arr = adv_vals.loc[phase_df.index].values
        choice_arr = choice_vals.loc[phase_df.index].values

        bin_x, bin_y = {}, {}
        if rolling:
            # Rolling window: step through by bin_step
            t_min, t_max = elapsed.min(), elapsed.max()
            tb_idx = 0
            t = t_min
            while t + bin_width <= t_max:
                mask = (elapsed >= t) & (elapsed < t + bin_width)
                if mask.sum() >= 3:
                    bin_x.setdefault(tb_idx, []).append(
                        np.mean(adv_arr[mask]))
                    bin_y.setdefault(tb_idx, []).append(
                        np.mean(choice_arr[mask]))
                tb_idx += 1
                t += bin_step
        else:
            # Non-overlapping bins
            phase_df["tb"] = (
                (elapsed - t_start) // bin_width
            ).astype(int)
            for tb, grp in phase_df.groupby("tb"):
                idx_in_arr = np.isin(elapsed,
                    grp["elapsed_time_s"].values)
                bin_x.setdefault(tb, []).append(np.mean(adv_arr[idx_in_arr]))
                bin_y.setdefault(tb, []).append(
                    np.mean(choice_arr[idx_in_arr]))

        return bin_x, bin_y

    # --- Left column: raw trajectories (all 4 phases) ---
    for row_idx, spec in enumerate(x_specs):
        ax = axes[row_idx, 0]
        for pb in pbs:
            pid = pb["id"]
            t_start = pb["start_ms"] / 1000
            t_end = pb["end_ms"] / 1000
            style = _phase_styles[pid]

            agg_x, agg_y = {}, {}
            for sid, sdf in events_df.groupby("session_id"):
                sdf = sdf.sort_values("timestamp_ms")
                bx, by = _phase_trajectory(sdf, spec, t_start, t_end)
                if bx is None:
                    continue
                for tb in bx:
                    agg_x.setdefault(tb, []).extend(bx[tb])
                    agg_y.setdefault(tb, []).extend(by[tb])

            if not agg_x:
                continue
            tbs = sorted(agg_x.keys())
            mx = np.array([np.mean(agg_x[t]) for t in tbs])
            my = np.array([np.mean(agg_y[t]) for t in tbs])

            dx = mx[1:] - mx[:-1]
            dy = my[1:] - my[:-1]
            ax.quiver(mx[:-1], my[:-1], dx, dy,
                      angles="xy", scale_units="xy", scale=1,
                      color=style["color"], alpha=0.3, width=0.005,
                      headwidth=5, headlength=4, headaxislength=3,
                      zorder=5, label=style["label"])

        ax.axhline(0.5, color="gray", ls=":", alpha=0.3)
        ax.axvline(0.0, color="gray", ls=":", alpha=0.3)
        ax.set_xlabel(spec["xlabel"], fontsize=_LABEL_FS,
                      labelpad=_LABELPAD)
        ax.set_ylabel("P(Choose A)", fontsize=_LABEL_FS,
                      labelpad=_LABELPAD)
        ax.set_ylim(0, 1)
        ax.set_title(spec["title"], fontsize=_TITLE_FS, fontweight="bold",
                     loc="left")
        ax.legend(fontsize=_LEGEND_FS, loc="best")
        sns.despine(ax=ax)

    # --- Right column: normalized (Phase 2 & 3 only) ---
    for row_idx, spec in enumerate(x_specs_norm):
        ax = axes[row_idx, 1]
        base_spec = x_specs[row_idx]  # use same data columns

        for pid in (2, 3):
            pb = pbs[pid - 1]
            t_start = pb["start_ms"] / 1000
            t_end = pb["end_ms"] / 1000
            style = _phase_styles[pid]

            agg_x, agg_y = {}, {}
            for sid, sdf in events_df.groupby("session_id"):
                sdf = sdf.sort_values("timestamp_ms")
                bx, by = _phase_trajectory(sdf, base_spec, t_start, t_end,
                                           normalize_phase=pid)
                if bx is None:
                    continue
                for tb in bx:
                    agg_x.setdefault(tb, []).extend(bx[tb])
                    agg_y.setdefault(tb, []).extend(by[tb])

            if not agg_x:
                continue
            tbs = sorted(agg_x.keys())
            mx = np.array([np.mean(agg_x[t]) for t in tbs])
            my = np.array([np.mean(agg_y[t]) for t in tbs])

            dx = mx[1:] - mx[:-1]
            dy = my[1:] - my[:-1]
            ax.quiver(mx[:-1], my[:-1], dx, dy,
                      angles="xy", scale_units="xy", scale=1,
                      color=style["color"], alpha=0.3, width=0.005,
                      headwidth=5, headlength=4, headaxislength=3,
                      zorder=5, label=style["label"])

        ax.axhline(0.5, color="gray", ls=":", alpha=0.3)
        ax.axvline(0.0, color="gray", ls=":", alpha=0.3)
        ax.set_xlabel(spec["xlabel"], fontsize=_LABEL_FS,
                      labelpad=_LABELPAD)
        ax.set_ylabel("P(Choose Higher-Payout)", fontsize=_LABEL_FS,
                      labelpad=_LABELPAD)
        ax.set_ylim(0, 1)
        ax.set_title(spec["title"], fontsize=_TITLE_FS, fontweight="bold",
                     loc="left")
        ax.legend(fontsize=_LEGEND_FS, loc="best")
        sns.despine(ax=ax)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    _save(fig, os.path.join(fig_dir, f"figure4_hysteresis.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 Supplement – Individual Participant Hysteresis
# ═══════════════════════════════════════════════════════════════════════════

def figure4_hysteresis_supplement(events_df, fig_dir, fmt, config):
    """Individual participant hysteresis trajectories.

    6 figures, each with 5 rows x 4 cols (2 participants per row,
    latent value + reward rate for each).
    """
    pbs = config.get("phase_boundaries", [])
    if len(pbs) < 2:
        return

    bin_width = 10
    bin_step = 1  # rolling window step

    _phase_styles = {
        1: {"color": "#999999", "label": "Ph 1"},
        2: {"color": "#4393C3", "label": "Ph 2"},
        3: {"color": "#D6604D", "label": "Ph 3"},
        4: {"color": "#5AAE61", "label": "Ph 4"},
    }

    x_specs = [
        {"col": "latent_advantage_pre",
         "xlabel": "$V_A - V_B$"},
        {"col_a": "local_reward_rate_a", "col_b": "local_reward_rate_b",
         "xlabel": "$R_A - R_B$"},
    ]

    sids = sorted(events_df["session_id"].unique())
    rows_per_fig = 5
    n_figs = int(np.ceil(len(sids) / rows_per_fig))

    # Column specs: raw latent, raw reward, normalized latent, normalized reward
    col_specs = [
        {"base": x_specs[0], "normalize": False,
         "xlabel": "$V_A - V_B$", "ylabel": "P(A)"},
        {"base": x_specs[1], "normalize": False,
         "xlabel": "$R_A - R_B$", "ylabel": "P(A)"},
        {"base": x_specs[0], "normalize": True,
         "xlabel": "$V_{Higher} - V_{Lower}$", "ylabel": "P(Higher)"},
        {"base": x_specs[1], "normalize": True,
         "xlabel": "$R_{Higher} - R_{Lower}$", "ylabel": "P(Higher)"},
    ]

    supp_dir = os.path.join(fig_dir, "supplements")
    os.makedirs(supp_dir, exist_ok=True)

    for fig_idx in range(n_figs):
        start = fig_idx * rows_per_fig
        end = min(start + rows_per_fig, len(sids))
        batch_sids = sids[start:end]
        n_rows = len(batch_sids)

        fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for row, sid in enumerate(batch_sids):
            sdf = events_df[events_df["session_id"] == sid].sort_values(
                "timestamp_ms")
            if "rolling_choice_prop_a_clicks" not in sdf.columns:
                continue

            for col_idx, cspec in enumerate(col_specs):
                ax = axes[row, col_idx]
                base = cspec["base"]
                normalize = cspec["normalize"]

                # Which phases to plot
                phases_to_plot = [2, 3] if normalize else list(range(1, 5))

                for phase_id in phases_to_plot:
                    pb = pbs[phase_id - 1]
                    t_start = pb["start_ms"] / 1000
                    t_end = pb["end_ms"] / 1000
                    style = _phase_styles.get(phase_id, _phase_styles[1])

                    # Compute advantage
                    if "col" in base:
                        if base["col"] not in sdf.columns:
                            continue
                        adv = sdf[base["col"]].copy()
                    else:
                        if (base["col_a"] not in sdf.columns or
                                base["col_b"] not in sdf.columns):
                            continue
                        adv = (sdf[base["col_a"]] - sdf[base["col_b"]]).copy()

                    choice = sdf["rolling_choice_prop_a_clicks"].copy()

                    mask = ((sdf["elapsed_time_s"] >= t_start) &
                            (sdf["elapsed_time_s"] < t_end) &
                            adv.notna() & choice.notna())
                    phase_df = sdf[mask]
                    if len(phase_df) < 5:
                        continue

                    # Normalize for Phase 3: flip so x = higher-lower,
                    # y = P(choose higher)
                    adv_plot = adv.loc[phase_df.index]
                    choice_plot = choice.loc[phase_df.index]
                    if normalize and phase_id == 3:
                        adv_plot = -adv_plot
                        choice_plot = 1 - choice_plot

                    elapsed = phase_df["elapsed_time_s"].values
                    adv_arr = adv_plot.values
                    choice_arr = choice_plot.values

                    tb_x, tb_y = [], []
                    t = elapsed.min()
                    t_max = elapsed.max()
                    while t + bin_width <= t_max:
                        wmask = (elapsed >= t) & (elapsed < t + bin_width)
                        if wmask.sum() >= 3:
                            tb_x.append(np.mean(adv_arr[wmask]))
                            tb_y.append(np.mean(choice_arr[wmask]))
                        t += bin_step

                    mean_x = np.array(tb_x)
                    mean_y = np.array(tb_y)
                    if len(mean_x) < 2:
                        continue

                    dx = mean_x[1:] - mean_x[:-1]
                    dy = mean_y[1:] - mean_y[:-1]
                    ax.quiver(mean_x[:-1], mean_y[:-1], dx, dy,
                              angles="xy", scale_units="xy", scale=1,
                              color=style["color"], alpha=0.3,
                              width=0.006, headwidth=4, headlength=3,
                              headaxislength=2.5)

                ax.axhline(0.5, color="gray", ls=":", alpha=0.3)
                ax.axvline(0.0, color="gray", ls=":", alpha=0.3)
                ax.set_ylim(0, 1)
                ax.tick_params(labelsize=8)
                sns.despine(ax=ax)

                # Bottom row gets x-labels
                if row == n_rows - 1:
                    ax.set_xlabel(cspec["xlabel"], fontsize=12)
                # Left-most raw and normalized columns get y-labels
                if col_idx == 0:
                    ax.set_ylabel(cspec["ylabel"], fontsize=12)
                elif col_idx == 2:
                    ax.set_ylabel(cspec["ylabel"], fontsize=12)

                # Participant ID on first column only
                if col_idx == 0:
                    short_id = (f"{sid[:14]}..." if len(str(sid)) > 14
                                else sid)
                    ax.set_title(short_id, fontsize=9, loc="left")

        # Column headers on top row
        col_headers = ["Raw: Latent Value", "Raw: Reinforcement Rate",
                       "Normalized: Latent Value", "Normalized: Reinforcement Rate"]
        for c, header in enumerate(col_headers):
            axes[0, c].text(0.5, 1.15, header, transform=axes[0, c].transAxes,
                            ha="center", fontsize=11, fontweight="bold")

        # Legend on first subplot
        handles = [mpatches.Patch(color=s["color"], alpha=0.5,
                                  label=s["label"])
                   for s in _phase_styles.values()]
        axes[0, 0].legend(handles=handles, fontsize=7, loc="best")
        # Normalized legend (Phase 2 & 3 only)
        norm_handles = [mpatches.Patch(color=_phase_styles[p]["color"],
                                       alpha=0.5,
                                       label=_phase_styles[p]["label"])
                        for p in (2, 3)]
        axes[0, 2].legend(handles=norm_handles, fontsize=7, loc="best")

        plt.tight_layout()
        fig.subplots_adjust(top=0.95)
        _save(fig, os.path.join(supp_dir,
              f"figure4_supp_{fig_idx + 1}.{fmt}"), config)

    print(f"  Hysteresis supplements ({n_figs} pages) saved to {supp_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 – HMM State Characteristics
# ═══════════════════════════════════════════════════════════════════════════

def figure5_hmm_state_characteristics(analysis_results, fig_dir, fmt, config):
    """State-space scatter of HMM states in choice x ICI space."""
    hmm_seqs = analysis_results.get("hmm_state_sequences", {})
    hmm_summary = analysis_results.get("hmm_state_summary")

    if not hmm_seqs and (hmm_summary is None or len(hmm_summary) == 0):
        print("  Figure 6 skipped: no HMM results.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    fig.subplots_adjust(hspace=0.3)

    # (A) Scatter plot of mean_choice_a vs mean_reward by state label
    ax = axes[0]
    if hmm_summary is not None and len(hmm_summary) > 0:
        per_session = hmm_summary[hmm_summary["session_id"] != "GROUP_MEAN"]
        for i, label_name in enumerate(
                sorted(per_session["label"].unique())):
            subset = per_session[per_session["label"] == label_name]
            ax.scatter(subset["mean_choice_a"], subset["mean_reward"],
                       s=30, alpha=0.5,
                       color=_STATE_GRAYS[i % len(_STATE_GRAYS)],
                       marker=_STATE_MARKERS[i % len(_STATE_MARKERS)],
                       label=label_name, edgecolors="white", linewidth=0.5)

        ax.set_xlabel("Mean P(Choose A)", fontsize=_LABEL_FS,
                      labelpad=_LABELPAD)
        ax.set_ylabel("Mean Reinforcement Rate", fontsize=_LABEL_FS,
                      labelpad=_LABELPAD)
        ax.legend(fontsize=_LEGEND_FS, loc="center left",
                  bbox_to_anchor=(1.05, 0.4), borderaxespad=0)
    sns.despine(ax=ax)
    ax.set_title("A. HMM States in\nChoice \u00d7 Reinforcement Space",
                 fontsize=_TITLE_FS, fontweight="bold", loc="left")

    # (B) Group-average emission means per state label with 95% CI
    ax = axes[1]
    if hmm_summary is not None and len(hmm_summary) > 0:
        per_session = hmm_summary[hmm_summary["session_id"] != "GROUP_MEAN"]
        state_labels = sorted(per_session["label"].unique())
        x = np.arange(len(state_labels))
        means, cis = [], []
        for lbl in state_labels:
            vals = per_session.loc[per_session["label"] == lbl,
                                   "mean_choice_a"].dropna().values
            means.append(np.mean(vals))
            sem = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            cis.append(1.96 * sem)

        ax.bar(x, means, yerr=cis, color="white", edgecolor="black",
               width=0.6, capsize=4, error_kw={"lw": 1.2, "capthick": 1.2})

        # Overlay individual participant values
        for i, lbl in enumerate(state_labels):
            vals = per_session.loc[per_session["label"] == lbl,
                                   "mean_choice_a"].dropna().values
            jitter = np.random.default_rng(i).uniform(
                -0.2, 0.2, len(vals))
            ax.scatter(x[i] + jitter, vals, s=10, alpha=0.4,
                       color="black", edgecolors="none", zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(state_labels, fontsize=_TICK_FS, rotation=15,
                           ha="right")
        ax.set_ylabel("Mean P(Choose A)", fontsize=_LABEL_FS,
                      labelpad=_LABELPAD)
        ax.axhline(0.5, color="gray", ls=":", alpha=0.3)
    ax.set_title("B. Group-Average State Profiles",
                 fontsize=_TITLE_FS, fontweight="bold", loc="left")
    sns.despine(ax=ax)

    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure5_hmm_states.{fmt}"), config)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6 – HMM Phase Alignment
# ═══════════════════════════════════════════════════════════════════════════

def figure6_hmm_phase_alignment(analysis_results, events_df, fig_dir, fmt,
                                config):
    """State assignments as colored bands aligned with phase boundaries."""
    hmm_seqs = analysis_results.get("hmm_state_sequences", {})
    if not hmm_seqs:
        print("  Figure 6 skipped: no HMM state sequences.")
        return

    # State colors for background bands
    _state_colors = ["#4393C3", "#D6604D", "#5AAE61", "#984EA3"]

    sids = list(hmm_seqs.keys())

    # Show 6 representative sessions in main figure
    n_show = min(6, len(sids))
    step = max(1, len(sids) // n_show)
    show_sids = sids[::step][:n_show]

    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.5 * n_show),
                             sharex=True)
    if n_show == 1:
        axes = [axes]

    _plot_hmm_panels(axes, show_sids, hmm_seqs, events_df, config,
                     _state_colors)

    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure6_hmm_phase_alignment.{fmt}"),
          config)

    # --- Supplemental: all participants, 6 per page ---
    supp_dir = os.path.join(fig_dir, "supplements")
    os.makedirs(supp_dir, exist_ok=True)
    all_sids = sorted(hmm_seqs.keys())
    rows_per_fig = 6
    n_figs = int(np.ceil(len(all_sids) / rows_per_fig))

    for fig_idx in range(n_figs):
        start = fig_idx * rows_per_fig
        end = min(start + rows_per_fig, len(all_sids))
        batch = all_sids[start:end]
        n_rows = len(batch)

        sfig, saxes = plt.subplots(n_rows, 1, figsize=(10, 2.5 * n_rows),
                                   sharex=True)
        if n_rows == 1:
            saxes = [saxes]

        _plot_hmm_panels(saxes, batch, hmm_seqs, events_df, config,
                         _state_colors)

        plt.tight_layout()
        _save(sfig, os.path.join(supp_dir,
              f"figure6_supp_{fig_idx + 1}.{fmt}"), config)

    print(f"  HMM alignment supplements ({n_figs} pages) saved to {supp_dir}")


def _plot_hmm_panels(axes, sids, hmm_seqs, events_df, config, state_colors):
    """Shared plotting logic for HMM phase alignment panels."""
    for i, sid in enumerate(sids):
        ax = axes[i]
        sdf = events_df[events_df["session_id"] == sid].sort_values(
            "timestamp_ms")
        seq = hmm_seqs[sid]
        states = seq["states"]

        times = sdf["elapsed_time_s"].values[:len(states)]

        # Color background bands by HMM state
        for t_idx in range(len(times) - 1):
            s = states[t_idx]
            ax.axvspan(times[t_idx], times[t_idx + 1],
                       color=state_colors[s % len(state_colors)],
                       alpha=0.20, linewidth=0)

        # Plot rolling choice proportion
        if "rolling_choice_prop_a_clicks" in sdf.columns:
            signal = sdf["rolling_choice_prop_a_clicks"].values[:len(states)]
            ax.plot(times, signal, color="black", lw=0.8, alpha=0.6)

        _phase_lines(ax, config)
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(A)", fontsize=_LABEL_FS, labelpad=_LABELPAD)
        label = f"{sid[:10]}..." if len(str(sid)) > 10 else sid
        ax.text(0.01, 0.95, label, transform=ax.transAxes,
                fontsize=_ANNOTATION_FS, va="top", color="gray")
        sns.despine(ax=ax)

    axes[-1].set_xlabel("Time (s)", fontsize=_LABEL_FS, labelpad=_LABELPAD)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7 – Convergent Cross-Mapping
# ═══════════════════════════════════════════════════════════════════════════

def figure7_ccm(analysis_results, fig_dir, fmt, config):
    """CCM convergence per phase: reward -> choice vs choice -> reward."""
    ccm_df = analysis_results.get("ccm")
    if ccm_df is None or len(ccm_df) == 0:
        print("  Figure 7 skipped: no CCM results.")
        return

    # Determine phases present in data
    if "phase_id" in ccm_df.columns:
        phases = sorted(ccm_df["phase_id"].unique())
        phases = [p for p in phases if p > 0]  # exclude 0 (full-session)
    else:
        phases = [0]

    n_phases = len(phases)
    # Layout: 4 rows x 2 cols (one row per phase)
    fig, axes = plt.subplots(n_phases, 2, figsize=(12, 5 * n_phases))
    if n_phases == 1:
        axes = axes[np.newaxis, :]

    phase_labels = {0: "Full Session", **_PHASE_LABELS}

    for row, pid in enumerate(phases):
        if "phase_id" in ccm_df.columns:
            pdf = ccm_df[ccm_df["phase_id"] == pid]
        else:
            pdf = ccm_df

        label = phase_labels.get(pid, f"Phase {pid}")

        # Left column: convergence curves (normalized library proportion)
        ax = axes[row, 0]

        # Normalize lib_size to proportion of each participant's max
        pdf = pdf.copy()
        max_libs = pdf.groupby("session_id")["lib_size"].transform("max")
        pdf["lib_prop"] = pdf["lib_size"] / max_libs

        # Bin into 5 equal-width bins so all participants contribute to each
        pdf["lib_bin"] = pd.cut(pdf["lib_prop"], bins=5, labels=False)
        bin_centers = pdf.groupby("lib_bin")["lib_prop"].mean()

        group = pdf.groupby("lib_bin").agg(
            rc_mean=("rho_reward_causes_choice", "mean"),
            rc_sem=("rho_reward_causes_choice", "sem"),
            cr_mean=("rho_choice_causes_reward", "mean"),
            cr_sem=("rho_choice_causes_reward", "sem"),
        )
        group["lib_prop"] = bin_centers

        ax.errorbar(group["lib_prop"], group["rc_mean"],
                    yerr=1.96 * group["rc_sem"], fmt="o-", color="black",
                    lw=2, markersize=8, alpha=0.5, capsize=3,
                    label="Reinforcement \u2192 Choice")
        ax.errorbar(group["lib_prop"], group["cr_mean"],
                    yerr=1.96 * group["cr_sem"], fmt="s--", color="#666666",
                    lw=2, markersize=8, alpha=0.5, capsize=3,
                    label="Choice \u2192 Reinforcement")

        ax.set_title(label, fontsize=_TITLE_FS, fontweight="bold",
                     loc="left")
        ax.set_ylabel("Cross-Mapping $\\it{\\rho}$",
                      fontsize=_LABEL_FS, labelpad=_LABELPAD)
        ax.set_xlim(0, 1.1)
        if row == n_phases - 1:
            ax.set_xlabel("Library Proportion", fontsize=_LABEL_FS,
                          labelpad=_LABELPAD)
        if row == 0:
            ax.legend(fontsize=14)
        sns.despine(ax=ax)

        # Right column: per-session asymmetry scatter
        ax = axes[row, 1]
        group_keys = ["session_id"]
        if "phase_id" in pdf.columns:
            group_keys.append("phase_id")
        max_lib = pdf.loc[pdf.groupby(group_keys)["lib_size"].idxmax()]
        if "rho_reward_causes_choice" in max_lib.columns:
            rc = max_lib["rho_reward_causes_choice"].values
            cr = max_lib["rho_choice_causes_reward"].values
            ax.scatter(cr, rc, s=30, alpha=0.5, color="#555555",
                       edgecolors="white")
            all_vals = np.concatenate([cr, rc])
            lo = min(0, np.nanmin(all_vals) - 0.05)
            hi = max(1, np.nanmax(all_vals) + 0.05)
            ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.5)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_title(f"Asymmetry: {label}", fontsize=_TITLE_FS,
                     fontweight="bold", loc="left")
        ax.set_ylabel("Reinforcement \u2192 Choice $\\it{\\rho}$",
                      fontsize=_LABEL_FS, labelpad=_LABELPAD)
        if row == n_phases - 1:
            ax.set_xlabel("Choice \u2192 Reinforcement $\\it{\\rho}$",
                          fontsize=_LABEL_FS, labelpad=_LABELPAD)
        sns.despine(ax=ax)

    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, f"figure7_ccm.{fmt}"), config)

    # --- Supplemental: individual CCM convergence curves ---
    supp_dir = os.path.join(fig_dir, "supplements")
    os.makedirs(supp_dir, exist_ok=True)

    all_sids = sorted(ccm_df["session_id"].unique())
    per_page = 30  # 6 rows x 5 cols
    n_figs = int(np.ceil(len(all_sids) / per_page))
    n_cols = 5
    phase_markers = {1: "o", 2: "s", 3: "^", 4: "D"}

    for fig_idx in range(n_figs):
        start = fig_idx * per_page
        end = min(start + per_page, len(all_sids))
        batch = all_sids[start:end]
        n_rows = int(np.ceil(len(batch) / n_cols))

        sfig, saxes = plt.subplots(n_rows, n_cols,
                                   figsize=(4 * n_cols, 3 * n_rows))
        if n_rows == 1:
            saxes = saxes[np.newaxis, :]

        for p_idx, sid in enumerate(batch):
            row = p_idx // n_cols
            col = p_idx % n_cols
            ax = saxes[row, col]

            sid_data = ccm_df[ccm_df["session_id"] == sid]

            for pid in phases:
                if "phase_id" in sid_data.columns:
                    pdata = sid_data[sid_data["phase_id"] == pid]
                else:
                    pdata = sid_data
                if len(pdata) == 0:
                    continue

                max_lib = pdata["lib_size"].max()
                if max_lib <= 0:
                    continue
                lib_prop = pdata["lib_size"] / max_lib
                marker = phase_markers.get(pid, "o")

                # R->C: black filled
                ax.plot(lib_prop, pdata["rho_reward_causes_choice"],
                        marker=marker, ls="-", color="black",
                        markersize=3, lw=1, alpha=0.5)
                # C->R: gray open
                ax.plot(lib_prop, pdata["rho_choice_causes_reward"],
                        marker=marker, ls="--", color="gray",
                        markersize=3, lw=1, alpha=0.5,
                        markerfacecolor="white", markeredgecolor="gray")

            ax.set_xlim(0, 1.1)
            ax.set_ylim(-0.2, 1.0)
            short_id = f"{sid[:10]}..." if len(str(sid)) > 10 else sid
            ax.set_title(short_id, fontsize=7, loc="left")
            ax.tick_params(labelsize=7)
            sns.despine(ax=ax)

            if row == n_rows - 1:
                ax.set_xlabel("Lib. Prop.", fontsize=8)
            if col == 0:
                ax.set_ylabel("$\\it{\\rho}$", fontsize=10)

        # Hide unused axes
        for p_idx in range(len(batch), n_rows * n_cols):
            saxes[p_idx // n_cols, p_idx % n_cols].set_visible(False)

        # Legend on first subplot
        phase_names = {1: "Sym", 2: "A-adv", 3: "B-adv", 4: "Scar"}
        handles = []
        for p in phases:
            m = phase_markers.get(p, "o")
            handles.append(Line2D([0], [0], marker=m, color="black",
                                  markersize=4, lw=1,
                                  label=f"{phase_names.get(p, '')} R\u2192C"))
            handles.append(Line2D([0], [0], marker=m, color="gray",
                                  markersize=4, lw=1, ls="--",
                                  markerfacecolor="white",
                                  markeredgecolor="gray",
                                  label=f"{phase_names.get(p, '')} C\u2192R"))
        saxes[0, 0].legend(handles=handles, fontsize=4, loc="upper left",
                           ncol=2)

        plt.tight_layout()
        _save(sfig, os.path.join(supp_dir,
              f"figure7_supp_{fig_idx + 1}.{fmt}"), config)

    print(f"  CCM supplements ({n_figs} pages) saved to {supp_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8 – Recurrence Quantification Analysis
# ═══════════════════════════════════════════════════════════════════════════

def figure8_rqa(analysis_results, events_df, fig_dir, fmt, config):
    """Individual recurrence plots in a 7 col x 9 row grid.

    First two cells are reference plots (Lorenz attractor for chaotic
    structure, white noise for no structure). Remaining 61 cells hold
    the 60 participants plus one unused cell.
    """
    if "rolling_choice_prop_a_clicks" not in events_df.columns:
        print("  Figure 8 skipped: no rolling choice proportion.")
        return

    pbs = config.get("phase_boundaries", [])
    sids = sorted(events_df["session_id"].unique())

    n_cols = 5
    n_rows = 7
    threshold = 0.05
    n_ref = 300

    # Number of reference cells (3 examples + 1 spacer)
    n_ref_cells = 4
    slots_per_page = n_cols * n_rows - n_ref_cells
    n_figs = int(np.ceil(len(sids) / slots_per_page))

    def _plot_references(axes_row):
        """Plot 3 reference recurrence plots + 1 spacer in first row."""
        # 1. Damped oscillation
        ax = axes_row[0]
        t_ref = np.linspace(0, 6 * np.pi, n_ref)
        damped = 0.5 + 0.5 * np.exp(-t_ref / (4 * np.pi)) * np.sin(t_ref)
        dist = np.abs(damped[:, None] - damped[None, :])
        rec = (dist < threshold).astype(int)
        np.fill_diagonal(rec, 0)
        ax.imshow(rec, cmap="Greys", origin="lower", aspect="equal")
        ax.set_title("Damped Oscillation", fontsize=12, fontweight="bold",
                     loc="left", pad=2)
        ax.set_xticks([]); ax.set_yticks([])

        # 2. Chirp (structured but changing frequency)
        ax = axes_row[1]
        t_ref2 = np.linspace(0, 1, n_ref)
        chirp = 0.5 + 0.5 * np.sin(2 * np.pi * (2 + 15 * t_ref2) * t_ref2)
        dist = np.abs(chirp[:, None] - chirp[None, :])
        rec = (dist < threshold).astype(int)
        np.fill_diagonal(rec, 0)
        ax.imshow(rec, cmap="Greys", origin="lower", aspect="equal")
        ax.set_title("Chirp (Non-stationary)", fontsize=12,
                     fontweight="bold", loc="left", pad=2)
        ax.set_xticks([]); ax.set_yticks([])

        # 3. White noise (no structure)
        ax = axes_row[2]
        rng = np.random.default_rng(42)
        noise = rng.uniform(0, 1, n_ref)
        dist = np.abs(noise[:, None] - noise[None, :])
        rec = (dist < threshold).astype(int)
        np.fill_diagonal(rec, 0)
        ax.imshow(rec, cmap="Greys", origin="lower", aspect="equal")
        ax.set_title("Random (White Noise)", fontsize=12,
                     fontweight="bold", loc="left", pad=2)
        ax.set_xticks([]); ax.set_yticks([])

        # 4. Spacer
        ax = axes_row[3]
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_visible(False)

    for fig_idx in range(n_figs):
        start = fig_idx * slots_per_page
        end = min(start + slots_per_page, len(sids))
        batch = sids[start:end]

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(3 * n_cols, 3 * n_rows))

        # Reference plots in first row
        _plot_references(axes[0])

        # Participants start at cell index 4 (after 3 refs + 1 spacer)
        cell_idx = n_ref_cells
        p_num = start + 1  # P1, P2, ...

        for sid in batch:
            row = cell_idx // n_cols
            col = cell_idx % n_cols
            if row >= n_rows:
                break
            ax = axes[row, col]

            sdf = events_df[events_df["session_id"] == sid].sort_values(
                "timestamp_ms")
            signal = sdf["rolling_choice_prop_a_clicks"].dropna().values
            elapsed = sdf["elapsed_time_s"].dropna().values[:len(signal)]

            if len(signal) < 20:
                ax.set_visible(False)
                cell_idx += 1
                continue

            if len(signal) > 300:
                step = len(signal) // 300
                signal = signal[::step]
                elapsed = elapsed[::step]

            n = len(signal)
            dist = np.abs(signal[:, None] - signal[None, :])
            recurrence = (dist < threshold).astype(int)
            np.fill_diagonal(recurrence, 0)

            ax.imshow(recurrence, cmap="Greys", origin="lower",
                      aspect="equal", extent=[0, n, 0, n],
                      zorder=1, interpolation="nearest")

            # Phase boundary pips outside the plot area
            for pb in pbs:
                t_boundary = pb["start_ms"] / 1000
                if t_boundary <= 0:
                    continue
                idx = np.searchsorted(elapsed, t_boundary)
                if 0 < idx < n:
                    ax.plot([idx, idx], [-n * 0.04, 0],
                            color="black", lw=1.5, clip_on=False)
                    ax.plot([-n * 0.04, 0], [idx, idx],
                            color="black", lw=1.5, clip_on=False)

            ax.set_title(f"P{p_num}", fontsize=8, loc="left", pad=2)
            ax.set_xticks([])
            ax.set_yticks([])

            cell_idx += 1
            p_num += 1

        # Hide unused cells
        for ci in range(cell_idx, n_rows * n_cols):
            axes[ci // n_cols, ci % n_cols].set_visible(False)

        fig.supxlabel("Click Index", fontsize=42, y=-0.01)
        fig.supylabel("Click Index", fontsize=42, x=-0.01)
        plt.tight_layout(pad=0.5)

        suffix = f"_{fig_idx + 1}" if n_figs > 1 else ""
        _save(fig, os.path.join(fig_dir,
              f"figure8_rqa{suffix}.{fmt}"), config)
