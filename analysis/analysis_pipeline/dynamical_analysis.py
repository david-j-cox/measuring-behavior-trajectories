"""Dynamical systems analysis: state-space trajectories, hysteresis, autocorrelation, EDM, RQA, CCM."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import correlate
import pyEDM


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

    # Recurrence analysis with RQA metrics
    rqa_results = _compute_rqa_all(events_df, config)
    results["rqa"] = rqa_results
    _plot_recurrence(events_df, config, fig_dir, fmt)
    if len(rqa_results) > 0:
        _plot_rqa_summary(rqa_results, config, fig_dir, fmt)

    # EDM simplex projection (all participants)
    edm_results = _run_edm_simplex(events_df, config, fig_dir, fmt)
    results["edm_simplex"] = edm_results

    # CCM: causal coupling between reward and choice
    ccm_results = _run_ccm(events_df, config, fig_dir, fmt)
    results["ccm"] = ccm_results

    # S-Map: state-dependent nonlinearity
    smap_results = _run_smap(events_df, config, fig_dir, fmt)
    results["smap"] = smap_results

    return results


def _plot_state_space(events_df: pd.DataFrame, config: dict,
                      fig_dir: str, fmt: str):
    """Plot 2D state-space KDE (choice prop vs reward rate) for all participants."""
    if "rolling_choice_prop_a_clicks" not in events_df.columns:
        return
    if "rolling_reward_rate_clicks" not in events_df.columns:
        return

    from scipy.stats import gaussian_kde

    phase_colors = {1: "Greys", 2: "Blues", 3: "Reds", 4: "Greens"}
    phase_line_colors = {1: "gray", 2: "dodgerblue", 3: "orangered", 4: "green"}
    session_ids = events_df["session_id"].unique()

    n_cols = min(4, len(session_ids))
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

        # Combined KDE across all phases
        x_all = sdf["rolling_choice_prop_a_clicks"].dropna().values
        y_all = sdf["rolling_reward_rate_clicks"].dropna().values
        if len(x_all) < 10:
            ax.set_visible(False)
            continue

        # Compute and plot KDE per phase
        xi = np.linspace(0, 1, 80)
        yi = np.linspace(0, 1, 80)
        Xi, Yi = np.meshgrid(xi, yi)
        positions = np.vstack([Xi.ravel(), Yi.ravel()])

        for pid in sorted(phase_colors.keys()):
            phase_df = sdf[sdf["phase_id"] == pid]
            xp = phase_df["rolling_choice_prop_a_clicks"].dropna().values
            yp = phase_df["rolling_reward_rate_clicks"].dropna().values
            if len(xp) < 10:
                continue
            try:
                kde = gaussian_kde(np.vstack([xp, yp]), bw_method=0.15)
                Z = kde(positions).reshape(Xi.shape)
                ax.contourf(Xi, Yi, Z, levels=6, cmap=phase_colors[pid], alpha=0.4)
                ax.contour(Xi, Yi, Z, levels=3, colors=phase_line_colors[pid],
                           linewidths=0.5, alpha=0.6)
            except np.linalg.LinAlgError:
                ax.scatter(xp, yp, s=2, alpha=0.3, color=phase_line_colors[pid])

        ax.set_xlabel("P(Choose A)")
        ax.set_ylabel("Reward Rate")
        label = f"{sid[:8]}..." if len(str(sid)) > 8 else sid
        ax.set_title(label, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Manual legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=phase_line_colors[p], alpha=0.5,
                             label=f"Phase {p}") for p in sorted(phase_colors.keys())]
    if len(axes) > 0:
        fig.legend(handles=legend_elements, loc="upper right", fontsize=8)

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
    """Recurrence plots for all participants."""
    if "rolling_choice_prop_a_clicks" not in events_df.columns:
        return

    session_ids = events_df["session_id"].unique()
    threshold = 0.1

    n_cols = min(4, len(session_ids))
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
        signal = sdf["rolling_choice_prop_a_clicks"].values

        if len(signal) < 20:
            ax.set_visible(False)
            continue

        # Subsample if too long
        if len(signal) > 300:
            step = len(signal) // 300
            signal = signal[::step]

        dist = np.abs(signal[:, None] - signal[None, :])
        recurrence = (dist < threshold).astype(int)

        ax.imshow(recurrence, cmap="Greys", origin="lower", aspect="equal")
        ax.set_xlabel("Click index")
        ax.set_ylabel("Click index")
        ax.set_title(f"{sid[:8]}..." if len(str(sid)) > 8 else f"{sid}")

    for j in range(len(session_ids), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Recurrence Plots (All Participants)", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"recurrence_all.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)


def _run_edm_simplex(events_df: pd.DataFrame, config: dict,
                     fig_dir: str, fmt: str) -> pd.DataFrame:
    """Run simplex projection (EDM) for each participant and plot observed vs predicted."""
    if "rolling_choice_prop_a_clicks" not in events_df.columns:
        return pd.DataFrame()

    session_ids = events_df["session_id"].unique()
    edm_rows = []

    n_cols = min(4, len(session_ids))
    n_rows = int(np.ceil(len(session_ids) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for i, sid in enumerate(session_ids):
        sdf = events_df[events_df["session_id"] == sid].sort_values("timestamp_ms")
        signal = sdf["rolling_choice_prop_a_clicks"].dropna().values

        if len(signal) < 30:
            if i < len(axes):
                axes[i].set_visible(False)
            continue

        # Build a dataframe for pyEDM
        edm_df = pd.DataFrame({"time": np.arange(len(signal)), "x": signal})

        # Find optimal embedding dimension via simplex
        try:
            simplex_out = pyEDM.EmbedDimension(
                dataFrame=edm_df, columns="x", target="x",
                lib=f"1 {len(signal) // 2}",
                pred=f"{len(signal) // 2 + 1} {len(signal)}",
                maxE=10, showPlot=False
            )

            best_E = int(simplex_out.loc[simplex_out["rho"].idxmax(), "E"])
            if best_E < 1:
                best_E = 1

            # Run simplex projection at the best E
            pred_out = pyEDM.Simplex(
                dataFrame=edm_df, columns="x", target="x",
                lib=f"1 {len(signal) // 2}",
                pred=f"{len(signal) // 2 + 1} {len(signal)}",
                E=best_E, showPlot=False
            )

            best_rho = simplex_out["rho"].max()
            edm_rows.append({
                "session_id": sid,
                "best_E": best_E,
                "rho": best_rho,
                "n_points": len(signal),
            })

            # Plot observed vs predicted
            if i < len(axes):
                ax = axes[i]
                pred_df = pred_out.dropna(subset=["Predictions"])
                ax.plot(pred_df["Time"], pred_df["Observations"],
                        color="steelblue", linewidth=0.8, label="Observed", alpha=0.7)
                ax.plot(pred_df["Time"], pred_df["Predictions"],
                        color="orangered", linewidth=0.8, label="Predicted", alpha=0.7)
                ax.set_xlabel("Click index")
                ax.set_ylabel("P(Choose A)")
                label = f"{sid[:8]}..." if len(str(sid)) > 8 else sid
                ax.set_title(f"{label} (E={best_E}, ρ={best_rho:.2f})", fontsize=9)
                if i == 0:
                    ax.legend(fontsize=7)

        except Exception as e:
            print(f"  EDM failed for session {sid}: {e}")
            if i < len(axes):
                axes[i].set_visible(False)
            continue

    for j in range(len(session_ids), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("EDM Simplex Projection: Observed vs Predicted", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"edm_simplex_all.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)

    return pd.DataFrame(edm_rows)


# ============================================================
# Recurrence Quantification Analysis (RQA)
# ============================================================

def _compute_rqa_all(events_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute RQA metrics from recurrence matrices for all sessions."""
    if "rolling_choice_prop_a_clicks" not in events_df.columns:
        return pd.DataFrame()

    rows = []
    threshold = 0.1
    phase_boundaries = config.get("phase_boundaries", [])

    for sid, sdf in events_df.groupby("session_id"):
        sdf = sdf.sort_values("timestamp_ms")
        signal = sdf["rolling_choice_prop_a_clicks"].dropna().values

        if len(signal) < 30:
            continue

        # Full session RQA
        rqa = _compute_rqa_metrics(signal, threshold)
        rqa["session_id"] = sid
        rqa["scope"] = "full_session"
        rqa["phase_id"] = 0
        rows.append(rqa)

        # Per-phase RQA
        phases = sdf.loc[sdf["rolling_choice_prop_a_clicks"].notna(),
                         "phase_id"].values
        for pb in phase_boundaries:
            phase_mask = phases == pb["id"]
            phase_signal = signal[phase_mask]
            if len(phase_signal) < 20:
                continue
            rqa_phase = _compute_rqa_metrics(phase_signal, threshold)
            rqa_phase["session_id"] = sid
            rqa_phase["scope"] = f"phase_{pb['id']}"
            rqa_phase["phase_id"] = pb["id"]
            rows.append(rqa_phase)

    return pd.DataFrame(rows)


def _compute_rqa_metrics(signal, threshold=0.1):
    """
    Extract quantitative metrics from a recurrence plot.

    Metrics:
    - recurrence_rate (RR): fraction of recurrent points
    - determinism (DET): fraction of recurrent points forming diagonal lines (len >= 2)
    - mean_diagonal_length (L): mean length of diagonal lines
    - max_diagonal_length (Lmax): longest diagonal line
    - entropy_diagonal (ENTR): Shannon entropy of diagonal line length distribution
    - laminarity (LAM): fraction of recurrent points forming vertical lines (len >= 2)
    - trapping_time (TT): mean length of vertical lines
    """
    n = len(signal)
    # Subsample if too long (RQA on >500 points gets slow)
    if n > 500:
        step = n // 500
        signal = signal[::step]
        n = len(signal)

    # Distance matrix
    dist = np.abs(signal[:, None] - signal[None, :])
    recurrence = (dist < threshold).astype(int)

    # Remove main diagonal (self-recurrence)
    np.fill_diagonal(recurrence, 0)

    total_points = n * (n - 1)  # excluding diagonal
    recurrent_points = recurrence.sum()

    if total_points == 0:
        return _empty_rqa()

    rr = recurrent_points / total_points

    # Diagonal lines
    diag_lengths = _extract_line_lengths(recurrence, direction="diagonal")
    # Vertical lines
    vert_lengths = _extract_line_lengths(recurrence, direction="vertical")

    # Determinism
    diag_in_lines = sum(l for l in diag_lengths if l >= 2)
    det = diag_in_lines / recurrent_points if recurrent_points > 0 else 0

    # Mean and max diagonal length
    diag_lines = [l for l in diag_lengths if l >= 2]
    mean_diag = np.mean(diag_lines) if diag_lines else 0
    max_diag = max(diag_lines) if diag_lines else 0

    # Entropy of diagonal line distribution
    if diag_lines:
        line_counts = np.bincount(diag_lines)
        line_probs = line_counts[line_counts > 0] / sum(line_counts[line_counts > 0])
        entropy_diag = -np.sum(line_probs * np.log(line_probs + 1e-12))
    else:
        entropy_diag = 0

    # Laminarity
    vert_in_lines = sum(l for l in vert_lengths if l >= 2)
    lam = vert_in_lines / recurrent_points if recurrent_points > 0 else 0

    # Trapping time
    vert_lines = [l for l in vert_lengths if l >= 2]
    trapping = np.mean(vert_lines) if vert_lines else 0

    return {
        "recurrence_rate": rr,
        "determinism": det,
        "mean_diagonal_length": mean_diag,
        "max_diagonal_length": max_diag,
        "entropy_diagonal": entropy_diag,
        "laminarity": lam,
        "trapping_time": trapping,
        "n_points": n,
    }


def _empty_rqa():
    """Return empty RQA metrics dict."""
    return {
        "recurrence_rate": np.nan,
        "determinism": np.nan,
        "mean_diagonal_length": np.nan,
        "max_diagonal_length": np.nan,
        "entropy_diagonal": np.nan,
        "laminarity": np.nan,
        "trapping_time": np.nan,
        "n_points": 0,
    }


def _extract_line_lengths(recurrence, direction="diagonal"):
    """Extract line lengths from recurrence matrix along diagonals or verticals."""
    n = recurrence.shape[0]
    lengths = []

    if direction == "diagonal":
        # Scan all diagonals (excluding main)
        for offset in range(1, n):
            diag = np.diag(recurrence, k=offset)
            lengths.extend(_runs_of_ones(diag))
            diag_neg = np.diag(recurrence, k=-offset)
            lengths.extend(_runs_of_ones(diag_neg))
    elif direction == "vertical":
        for col in range(n):
            lengths.extend(_runs_of_ones(recurrence[:, col]))

    return lengths


def _runs_of_ones(arr):
    """Find lengths of consecutive runs of 1s in a binary array."""
    lengths = []
    count = 0
    for val in arr:
        if val == 1:
            count += 1
        else:
            if count > 0:
                lengths.append(count)
            count = 0
    if count > 0:
        lengths.append(count)
    return lengths


def _plot_rqa_summary(rqa_df: pd.DataFrame, config: dict,
                      fig_dir: str, fmt: str):
    """Plot RQA metric distributions and phase comparisons."""
    metrics = ["recurrence_rate", "determinism", "laminarity",
               "trapping_time", "entropy_diagonal"]
    labels = ["Recurrence Rate", "Determinism", "Laminarity",
              "Trapping Time", "Diagonal Entropy"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    full = rqa_df[rqa_df["scope"] == "full_session"]

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        if i >= len(axes):
            break
        ax = axes[i]
        vals = full[metric].dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=15, color="steelblue", edgecolor="white", alpha=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Full-Session {label}")

    # Phase comparison for determinism
    if len(axes) > 5:
        ax = axes[5]
        phase_data = rqa_df[rqa_df["phase_id"] > 0].copy()
        if len(phase_data) > 0:
            phase_data["phase_label"] = "Phase " + phase_data["phase_id"].astype(str)
            sns.boxplot(data=phase_data, x="phase_label", y="determinism",
                        ax=ax, color="steelblue", width=0.5)
            ax.set_xlabel("")
            ax.set_ylabel("Determinism")
            ax.set_title("Determinism by Phase")

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"rqa_summary.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)


# ============================================================
# Convergent Cross Mapping (CCM)
# ============================================================

def _run_ccm(events_df: pd.DataFrame, config: dict,
             fig_dir: str, fmt: str) -> pd.DataFrame:
    """
    Run Convergent Cross Mapping to test causal coupling between
    reward history and choice behavior.

    CCM tests: does the choice attractor contain information about
    reward dynamics (reward -> choice causation), and vice versa?
    """
    if "rolling_choice_prop_a_clicks" not in events_df.columns:
        return pd.DataFrame()
    if "rolling_reward_rate_clicks" not in events_df.columns:
        return pd.DataFrame()

    session_ids = events_df["session_id"].unique()
    ccm_rows = []

    for sid in session_ids:
        sdf = events_df[events_df["session_id"] == sid].sort_values("timestamp_ms")
        choice = sdf["rolling_choice_prop_a_clicks"].dropna().values
        reward = sdf["rolling_reward_rate_clicks"].dropna().values

        n = min(len(choice), len(reward))
        if n < 50:
            continue

        choice = choice[:n]
        reward = reward[:n]

        # Build dataframe for pyEDM
        edm_df = pd.DataFrame({
            "time": np.arange(n),
            "choice": choice,
            "reward": reward,
        })

        try:
            # First find optimal E for choice
            simplex_out = pyEDM.EmbedDimension(
                dataFrame=edm_df, columns="choice", target="choice",
                lib=f"1 {n // 2}", pred=f"{n // 2 + 1} {n}",
                maxE=10, showPlot=False
            )
            best_E = int(simplex_out.loc[simplex_out["rho"].idxmax(), "E"])
            best_E = max(2, best_E)

            # CCM: choice cross-maps reward (tests reward -> choice causation)
            # Use multiple library sizes to check convergence
            lib_sizes = np.linspace(max(best_E + 2, 20), n - 10,
                                    min(10, (n - 10 - 20) // 5 + 1)).astype(int)
            lib_sizes = np.unique(lib_sizes)

            for lib_size in lib_sizes:
                try:
                    ccm_out = pyEDM.CCM(
                        dataFrame=edm_df,
                        columns="choice", target="reward",
                        E=best_E,
                        libSizes=f"{lib_size}",
                        sample=50,
                        showPlot=False
                    )
                    rho_cr = ccm_out["choice:reward"].mean()

                    ccm_out2 = pyEDM.CCM(
                        dataFrame=edm_df,
                        columns="reward", target="choice",
                        E=best_E,
                        libSizes=f"{lib_size}",
                        sample=50,
                        showPlot=False
                    )
                    rho_rc = ccm_out2["reward:choice"].mean()

                    ccm_rows.append({
                        "session_id": sid,
                        "E": best_E,
                        "lib_size": lib_size,
                        "rho_reward_causes_choice": rho_cr,
                        "rho_choice_causes_reward": rho_rc,
                    })
                except Exception:
                    continue

        except Exception as e:
            print(f"  CCM failed for session {sid}: {e}")
            continue

    ccm_df = pd.DataFrame(ccm_rows)

    if len(ccm_df) > 0:
        _plot_ccm(ccm_df, config, fig_dir, fmt)

    return ccm_df


def _plot_ccm(ccm_df: pd.DataFrame, config: dict, fig_dir: str, fmt: str):
    """Plot CCM convergence curves."""
    session_ids = ccm_df["session_id"].unique()

    n_cols = min(4, len(session_ids))
    n_rows = int(np.ceil(len(session_ids) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).reshape(-1)

    for i, sid in enumerate(session_ids):
        if i >= len(axes):
            break
        ax = axes[i]
        sdf = ccm_df[ccm_df["session_id"] == sid].sort_values("lib_size")

        ax.plot(sdf["lib_size"], sdf["rho_reward_causes_choice"],
                "o-", color="dodgerblue", markersize=3, linewidth=1,
                label="Reward → Choice")
        ax.plot(sdf["lib_size"], sdf["rho_choice_causes_reward"],
                "s-", color="orangered", markersize=3, linewidth=1,
                label="Choice → Reward")

        ax.set_xlabel("Library size")
        ax.set_ylabel("CCM ρ")
        ax.set_ylim(-0.1, 1.0)
        label = f"{sid[:8]}..." if len(str(sid)) > 8 else sid
        ax.set_title(label, fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(session_ids), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Convergent Cross Mapping: Causal Coupling", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"ccm_convergence.{fmt}"),
                dpi=config.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)


# ============================================================
# S-Map: State-Dependent Nonlinearity
# ============================================================

def _run_smap(events_df: pd.DataFrame, config: dict,
              fig_dir: str, fmt: str) -> pd.DataFrame:
    """
    Run S-Map to quantify state-dependent nonlinearity.

    Compare Simplex (linear, theta=0) vs S-Map (nonlinear, theta>0).
    If S-Map outperforms Simplex, the dynamics are state-dependent / nonlinear.
    """
    if "rolling_choice_prop_a_clicks" not in events_df.columns:
        return pd.DataFrame()

    session_ids = events_df["session_id"].unique()
    smap_rows = []

    for sid in session_ids:
        sdf = events_df[events_df["session_id"] == sid].sort_values("timestamp_ms")
        signal = sdf["rolling_choice_prop_a_clicks"].dropna().values

        if len(signal) < 50:
            continue

        edm_df = pd.DataFrame({"time": np.arange(len(signal)), "x": signal})

        try:
            # Find best E first
            simplex_out = pyEDM.EmbedDimension(
                dataFrame=edm_df, columns="x", target="x",
                lib=f"1 {len(signal) // 2}",
                pred=f"{len(signal) // 2 + 1} {len(signal)}",
                maxE=10, showPlot=False
            )
            best_E = int(simplex_out.loc[simplex_out["rho"].idxmax(), "E"])
            best_E = max(1, best_E)
            simplex_rho = simplex_out["rho"].max()

            # S-Map with varying theta
            smap_out = pyEDM.PredictNonlinear(
                dataFrame=edm_df, columns="x", target="x",
                lib=f"1 {len(signal) // 2}",
                pred=f"{len(signal) // 2 + 1} {len(signal)}",
                E=best_E, showPlot=False
            )

            theta_col = "Theta" if "Theta" in smap_out.columns else "theta"
            best_theta = float(smap_out.loc[smap_out["rho"].idxmax(), theta_col])
            best_smap_rho = smap_out["rho"].max()

            # Nonlinearity: improvement of S-Map over Simplex
            nonlinearity = best_smap_rho - simplex_rho

            smap_rows.append({
                "session_id": sid,
                "best_E": best_E,
                "simplex_rho": simplex_rho,
                "best_theta": best_theta,
                "smap_rho": best_smap_rho,
                "nonlinearity": nonlinearity,
                "n_points": len(signal),
            })

        except Exception as e:
            print(f"  S-Map failed for session {sid}: {e}")
            continue

    smap_df = pd.DataFrame(smap_rows)

    if len(smap_df) > 0:
        _plot_smap(smap_df, config, fig_dir, fmt)

    return smap_df


def _plot_smap(smap_df: pd.DataFrame, config: dict,
               fig_dir: str, fmt: str):
    """Plot S-Map nonlinearity results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Simplex vs S-Map rho
    axes[0].scatter(smap_df["simplex_rho"], smap_df["smap_rho"],
                    color="steelblue", alpha=0.7, s=40)
    lims = [min(axes[0].get_xlim()[0], axes[0].get_ylim()[0]),
            max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])]
    axes[0].plot(lims, lims, "--", color="gray", alpha=0.5)
    axes[0].set_xlabel("Simplex ρ (linear)")
    axes[0].set_ylabel("S-Map ρ (nonlinear)")
    axes[0].set_title("Linear vs Nonlinear Prediction")

    # Nonlinearity distribution
    axes[1].hist(smap_df["nonlinearity"], bins=15, color="teal",
                 edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Nonlinearity (Δρ)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("S-Map - Simplex (>0 = nonlinear)")

    # Best theta distribution
    axes[2].hist(smap_df["best_theta"], bins=15, color="purple",
                 edgecolor="white", alpha=0.8)
    axes[2].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[2].set_xlabel("Best θ")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Optimal S-Map θ (0 = linear)")

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"smap_nonlinearity.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)
