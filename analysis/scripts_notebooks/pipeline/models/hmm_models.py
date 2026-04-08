"""Hidden Markov Models for identifying latent behavioral states."""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp


def fit_hmm_models(events_df: pd.DataFrame, config: dict,
                   output_dir: str) -> pd.DataFrame:
    """Fit HMMs with varying state counts and compare. Returns model comparison table."""
    fmt = config.get("plot_format", "png")
    fig_dir = os.path.join(output_dir, "figures", "hmm")
    os.makedirs(fig_dir, exist_ok=True)

    rows = []
    all_state_sequences = {}

    for sid, sdf in events_df.groupby("session_id"):
        sdf = sdf.sort_values("timestamp_ms").reset_index(drop=True)

        if len(sdf) < 30:
            continue

        # Build observation matrix: choice_a, reward, rolling choice prop
        obs_cols = ["choice_a", "reward_outcome"]
        if "rolling_choice_prop_a_clicks" in sdf.columns:
            obs_cols.append("rolling_choice_prop_a_clicks")
        if "ici_s" in sdf.columns:
            obs_cols.append("ici_s")

        obs = sdf[obs_cols].fillna(0).values

        # Standardize continuous features (keep binary as-is)
        obs_scaled = obs.copy()
        for col_idx in range(len(obs_cols)):
            if obs_cols[col_idx] not in ("choice_a", "reward_outcome"):
                col_data = obs_scaled[:, col_idx]
                std = col_data.std()
                if std > 0:
                    obs_scaled[:, col_idx] = (col_data - col_data.mean()) / std

        best_n_states = 2
        best_bic = np.inf
        best_model = None

        for n_states in range(2, 5):
            result = _fit_single_hmm(obs_scaled, n_states, n_restarts=5)
            if result is None:
                continue

            n_params = _hmm_n_params(n_states, obs_scaled.shape[1])
            nll = -result["log_likelihood"]
            aic = 2 * n_params + 2 * nll
            bic = n_params * np.log(len(obs)) + 2 * nll

            rows.append({
                "session_id": sid,
                "model": f"hmm_{n_states}state",
                "n_params": n_params,
                "n_states": n_states,
                "nll": nll,
                "aic": aic,
                "bic": bic,
                "n_obs": len(obs),
                "log_likelihood": result["log_likelihood"],
            })

            if bic < best_bic:
                best_bic = bic
                best_n_states = n_states
                best_model = result

        if best_model is not None:
            all_state_sequences[sid] = {
                "states": best_model["states"],
                "n_states": best_n_states,
                "transition_matrix": best_model["transition_matrix"],
                "means": best_model["means"],
            }

    results_df = pd.DataFrame(rows)

    # Plot state sequences and transition matrices
    if all_state_sequences:
        _plot_hmm_states(events_df, all_state_sequences, config, fig_dir, fmt)
        _plot_transition_matrices(all_state_sequences, fig_dir, fmt, config)
        _plot_state_phase_alignment(events_df, all_state_sequences,
                                    config, fig_dir, fmt)

    # Cache state sequences as JSON for figure regeneration
    cache_path = os.path.join(output_dir, "tables", "hmm_state_sequences.json")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    _save_state_sequences(all_state_sequences, cache_path)

    return results_df, all_state_sequences


def _save_state_sequences(state_sequences: dict, path: str):
    """Serialize HMM state sequences to JSON (numpy arrays -> lists)."""
    serializable = {}
    for sid, seq in state_sequences.items():
        serializable[str(sid)] = {
            "states": seq["states"].tolist(),
            "n_states": int(seq["n_states"]),
            "transition_matrix": seq["transition_matrix"].tolist(),
            "means": seq["means"].tolist(),
        }
    with open(path, "w") as f:
        json.dump(serializable, f)


def load_state_sequences(path: str) -> dict:
    """Deserialize HMM state sequences from cached JSON."""
    with open(path) as f:
        raw = json.load(f)
    sequences = {}
    for sid, seq in raw.items():
        sequences[sid] = {
            "states": np.array(seq["states"]),
            "n_states": seq["n_states"],
            "transition_matrix": np.array(seq["transition_matrix"]),
            "means": np.array(seq["means"]),
        }
    return sequences


def summarize_hmm_states(hmm_state_sequences: dict, output_dir: str) -> pd.DataFrame:
    """Create a summary table of HMM emission parameters with descriptive labels.

    Parameters
    ----------
    hmm_state_sequences : dict
        Mapping of session_id -> {"states", "n_states", "transition_matrix", "means"}.
        The ``means`` array has shape (n_states, n_features) with columns ordered as
        [choice_a, reward_outcome, rolling_choice_prop_a_clicks, ici_s] (features
        present may vary).
    output_dir : str
        Root output directory; the CSV is saved under ``output_dir/tables/``.

    Returns
    -------
    pd.DataFrame
        Summary table with per-session state parameters and descriptive labels.
    """
    if not hmm_state_sequences:
        return pd.DataFrame()

    rows = []
    for sid, seq in hmm_state_sequences.items():
        means = seq["means"]  # shape (n_states, n_features)
        n_states = seq["n_states"]
        for s in range(n_states):
            row = {
                "session_id": sid,
                "state": s,
                "n_states": n_states,
                "mean_choice_a": means[s, 0] if means.shape[1] > 0 else np.nan,
                "mean_reward": means[s, 1] if means.shape[1] > 1 else np.nan,
            }
            if means.shape[1] > 3:
                row["mean_ici"] = means[s, 3]
            rows.append(row)

    summary = pd.DataFrame(rows)

    # Assign descriptive labels based on mean_choice_a within each session
    labels = []
    for _, grp in summary.groupby("session_id"):
        choice_means = grp["mean_choice_a"].values
        n = len(choice_means)
        sorted_idx = np.argsort(choice_means)
        state_labels = [""] * n
        if n == 2:
            state_labels[sorted_idx[0]] = "Exploiting B"
            state_labels[sorted_idx[1]] = "Exploiting A"
        elif n >= 3:
            state_labels[sorted_idx[0]] = "Exploiting B"
            state_labels[sorted_idx[-1]] = "Exploiting A"
            for mid in sorted_idx[1:-1]:
                state_labels[mid] = "Exploring/Switching"
        labels.extend(state_labels)
    summary["label"] = labels

    # Group-level averages across sessions
    group_avg = (
        summary.groupby("label")[["mean_choice_a", "mean_reward"]]
        .agg(["mean", "std", "count"])
    )
    group_avg.columns = ["_".join(c) for c in group_avg.columns]
    group_avg = group_avg.reset_index()

    # Add group averages as extra rows
    for _, row in group_avg.iterrows():
        summary = pd.concat([summary, pd.DataFrame([{
            "session_id": "GROUP_MEAN",
            "state": np.nan,
            "n_states": np.nan,
            "mean_choice_a": row["mean_choice_a_mean"],
            "mean_reward": row["mean_reward_mean"],
            "mean_ici": np.nan,
            "label": row["label"],
        }])], ignore_index=True)

    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    summary.to_csv(os.path.join(tables_dir, "hmm_state_summary.csv"), index=False)

    return summary


def _fit_single_hmm(obs, n_states, n_restarts=5):
    """Fit a single Gaussian HMM with multiple restarts."""
    best_ll = -np.inf
    best_model = None

    for seed in range(n_restarts):
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=200,
                tol=1e-4,
                random_state=seed + 42,
            )
            model.fit(obs)
            ll = model.score(obs)

            if ll > best_ll:
                best_ll = ll
                best_model = model
        except Exception:
            continue

    if best_model is None:
        return None

    states = best_model.predict(obs)

    # Reorder states by mean choice proportion (first observation dimension)
    # so state 0 = "choose B", state N-1 = "choose A"
    state_means = np.array([obs[states == s, 0].mean()
                            for s in range(best_model.n_components)])
    order = np.argsort(state_means)
    reorder_map = {old: new for new, old in enumerate(order)}
    states = np.array([reorder_map[s] for s in states])

    # Reorder transition matrix
    trans = best_model.transmat_[np.ix_(order, order)]

    return {
        "log_likelihood": best_ll,
        "states": states,
        "transition_matrix": trans,
        "means": best_model.means_[order],
        "model": best_model,
    }


def _hmm_n_params(n_states, n_features):
    """Count free parameters in a Gaussian HMM with diagonal covariance."""
    # Initial state probs: n_states - 1
    # Transition matrix: n_states * (n_states - 1)
    # Means: n_states * n_features
    # Diagonal covariances: n_states * n_features
    return ((n_states - 1) + n_states * (n_states - 1) +
            n_states * n_features + n_states * n_features)


def _plot_hmm_states(events_df: pd.DataFrame, state_sequences: dict,
                     config: dict, fig_dir: str, fmt: str):
    """Plot choice trajectories with HMM state overlay."""
    session_ids = list(state_sequences.keys())
    phase_boundaries_s = [p["start_ms"] / 1000
                          for p in config.get("phase_boundaries", [])[1:]]

    n_cols = min(4, len(session_ids))
    n_rows = int(np.ceil(len(session_ids) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).reshape(-1)

    state_colors = {0: "dodgerblue", 1: "orangered", 2: "green", 3: "purple"}

    for i, sid in enumerate(session_ids):
        if i >= len(axes):
            break
        ax = axes[i]
        sdf = events_df[events_df["session_id"] == sid].sort_values("timestamp_ms")
        seq = state_sequences[sid]
        states = seq["states"]
        n_states = seq["n_states"]

        times = sdf["elapsed_time_s"].values[:len(states)]
        signal = sdf["rolling_choice_prop_a_clicks"].values[:len(states)] \
            if "rolling_choice_prop_a_clicks" in sdf.columns else \
            sdf["choice_a"].values[:len(states)].astype(float)

        # Color by state
        for s in range(n_states):
            mask = states == s
            if mask.any():
                ax.scatter(times[mask], signal[mask], s=3, alpha=0.5,
                           color=state_colors.get(s, "gray"),
                           label=f"State {s}")

        # Phase boundaries
        for pb in phase_boundaries_s:
            ax.axvline(pb, color="black", linestyle="--", linewidth=0.8, alpha=0.4)

        ax.set_ylim(0, 1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("P(A)")
        label = f"{sid[:8]}..." if len(str(sid)) > 8 else sid
        ax.set_title(f"{label} ({n_states} states)", fontsize=9)
        if i == 0:
            ax.legend(fontsize=7, markerscale=3)

    for j in range(len(session_ids), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("HMM State Assignments", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"hmm_states.{fmt}"),
                dpi=config.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)


def _plot_transition_matrices(state_sequences: dict, fig_dir: str,
                              fmt: str, config: dict):
    """Plot transition matrices for all sessions."""
    session_ids = list(state_sequences.keys())

    n_cols = min(4, len(session_ids))
    n_rows = int(np.ceil(len(session_ids) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).reshape(-1)

    for i, sid in enumerate(session_ids):
        if i >= len(axes):
            break
        ax = axes[i]
        seq = state_sequences[sid]
        trans = seq["transition_matrix"]
        n_states = seq["n_states"]

        im = ax.imshow(trans, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        for row in range(n_states):
            for col in range(n_states):
                ax.text(col, row, f"{trans[row, col]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if trans[row, col] > 0.5 else "black")
        ax.set_xticks(range(n_states))
        ax.set_yticks(range(n_states))
        ax.set_xlabel("To state")
        ax.set_ylabel("From state")
        label = f"{sid[:8]}..." if len(str(sid)) > 8 else sid
        ax.set_title(label, fontsize=9)

    for j in range(len(session_ids), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("HMM Transition Matrices", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"hmm_transitions.{fmt}"),
                dpi=config.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)


def _plot_state_phase_alignment(events_df: pd.DataFrame,
                                state_sequences: dict,
                                config: dict, fig_dir: str, fmt: str):
    """Analyze how HMM states align with experimental phases."""
    phase_boundaries = config.get("phase_boundaries", [])
    rows = []

    for sid, seq in state_sequences.items():
        sdf = events_df[events_df["session_id"] == sid].sort_values("timestamp_ms")
        states = seq["states"]
        n_states = seq["n_states"]
        phases = sdf["phase_id"].values[:len(states)]

        for pb in phase_boundaries:
            phase_mask = phases == pb["id"]
            if phase_mask.sum() == 0:
                continue
            phase_states = states[phase_mask]
            for s in range(n_states):
                rows.append({
                    "session_id": sid,
                    "phase_id": pb["id"],
                    "state": s,
                    "proportion": (phase_states == s).mean(),
                })

    if not rows:
        return

    align_df = pd.DataFrame(rows)

    # Aggregate across sessions
    agg = align_df.groupby(["phase_id", "state"])["proportion"].agg(
        ["mean", "sem"]
    ).reset_index()

    n_phases = agg["phase_id"].nunique()
    fig, axes = plt.subplots(1, n_phases, figsize=(4 * n_phases, 4), sharey=True)
    if n_phases == 1:
        axes = [axes]

    state_colors = {0: "dodgerblue", 1: "orangered", 2: "green", 3: "purple"}

    for ax, pid in zip(axes, sorted(agg["phase_id"].unique())):
        phase_agg = agg[agg["phase_id"] == pid]
        bars = ax.bar(phase_agg["state"], phase_agg["mean"],
                      yerr=phase_agg["sem"], capsize=3,
                      color=[state_colors.get(s, "gray")
                             for s in phase_agg["state"]],
                      edgecolor="white", alpha=0.8)
        ax.set_xlabel("HMM State")
        ax.set_ylabel("Proportion of time")
        ax.set_title(f"Phase {pid}")
        ax.set_ylim(0, 1)

    fig.suptitle("HMM State Occupancy by Phase", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"hmm_phase_alignment.{fmt}"),
                dpi=config.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
