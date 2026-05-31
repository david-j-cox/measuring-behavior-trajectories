"""Change-point detection: Bayesian Online Change Point Detection (BOCPD)."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaln
from functools import lru_cache


def run_changepoint_analysis(events_df: pd.DataFrame, config: dict,
                             output_dir: str) -> dict:
    """Detect behavioral change points and compare to known phase boundaries."""
    results = {}
    fmt = config.get("plot_format", "png")
    fig_dir = os.path.join(output_dir, "figures", "changepoint")
    os.makedirs(fig_dir, exist_ok=True)

    # Run BOCPD per session
    cp_results = _detect_changepoints_all(events_df, config)
    results["changepoints"] = cp_results

    if len(cp_results) > 0:
        _plot_changepoints(events_df, cp_results, config, fig_dir, fmt)
        _plot_changepoint_alignment(cp_results, config, fig_dir, fmt)

        # Compute alignment with known phase boundaries
        alignment = _compute_boundary_alignment(cp_results, config)
        results["boundary_alignment"] = alignment

    return results


def bocpd_gaussian(data, hazard_rate=1/100, **kwargs):
    """Public entry point for BOCPD. See _bocpd_gaussian."""
    return _bocpd_gaussian(data, hazard_rate=hazard_rate, **kwargs)


def _bocpd_gaussian(data, hazard_rate=1/100, mu0=0.5, kappa0=1.0,
                    alpha0=1.0, beta0=0.05):
    """
    Bayesian Online Change Point Detection with Gaussian likelihood.

    Uses the conjugate Normal-Inverse-Gamma model.
    Returns run-length posterior and most probable change point locations.

    Adams & MacKay (2007).
    """
    T = len(data)
    # Run-length posterior: R[t, r] = P(run_length=r at time t)
    # For memory, keep only current and previous
    max_run = T + 1

    # Sufficient statistics for each run length
    # Using online updates of mean/precision
    mu = np.full(max_run, mu0)
    kappa = np.full(max_run, kappa0)
    alpha = np.full(max_run, alpha0)
    beta = np.full(max_run, beta0)

    # Run-length probabilities
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0

    # Most probable run length at each time
    max_run_length = np.zeros(T)
    changepoint_prob = np.zeros(T)

    for t in range(T):
        x = data[t]

        # Predictive probability for each run length
        pred_probs = _student_t_pdf(x, mu[:t+1], beta[:t+1] * (kappa[:t+1] + 1) /
                                    (alpha[:t+1] * kappa[:t+1]),
                                    2 * alpha[:t+1])

        # Growth probabilities
        H = hazard_rate
        growth_probs = R[t, :t+1] * pred_probs * (1 - H)

        # Change point probability
        cp_prob = np.sum(R[t, :t+1] * pred_probs * H)

        # Update run-length distribution
        R[t+1, 1:t+2] = growth_probs
        R[t+1, 0] = cp_prob

        # Normalize
        evidence = R[t+1, :t+2].sum()
        if evidence > 0:
            R[t+1, :t+2] /= evidence

        # Record
        max_run_length[t] = np.argmax(R[t+1, :t+2])
        changepoint_prob[t] = R[t+1, 0]

        # Update sufficient statistics
        mu_new = np.empty(t + 2)
        kappa_new = np.empty(t + 2)
        alpha_new = np.empty(t + 2)
        beta_new = np.empty(t + 2)

        # New run (changepoint)
        mu_new[0] = mu0
        kappa_new[0] = kappa0
        alpha_new[0] = alpha0
        beta_new[0] = beta0

        # Extended runs
        mu_new[1:t+2] = (kappa[:t+1] * mu[:t+1] + x) / (kappa[:t+1] + 1)
        kappa_new[1:t+2] = kappa[:t+1] + 1
        alpha_new[1:t+2] = alpha[:t+1] + 0.5
        beta_new[1:t+2] = (beta[:t+1] +
                           kappa[:t+1] * (x - mu[:t+1])**2 / (2 * (kappa[:t+1] + 1)))

        mu[:t+2] = mu_new
        kappa[:t+2] = kappa_new
        alpha[:t+2] = alpha_new
        beta[:t+2] = beta_new

    # Derive changepoint signal from run-length drops
    # A changepoint occurs when the MAP run length drops sharply
    cp_signal = np.zeros(T)
    for t in range(1, T):
        if max_run_length[t] < max_run_length[t-1] * 0.5 and max_run_length[t-1] > 5:
            cp_signal[t] = 1.0 - max_run_length[t] / max(max_run_length[t-1], 1)
        # Also flag when short run lengths dominate
        short_prob = R[t+1, :min(5, t+2)].sum()
        if short_prob > 0.5:
            cp_signal[t] = max(cp_signal[t], short_prob)

    return cp_signal, max_run_length


def _student_t_pdf(x, mu, var, nu):
    """Vectorized Student-t PDF (log-space for stability)."""
    nu = np.maximum(nu, 1e-8)
    var = np.maximum(var, 1e-8)
    log_p = (gammaln((nu + 1) / 2) - gammaln(nu / 2)
             - 0.5 * np.log(nu * np.pi * var)
             - (nu + 1) / 2 * np.log(1 + (x - mu)**2 / (nu * var)))
    return np.exp(log_p)


def _detect_changepoints_all(events_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run BOCPD on each session's rolling choice proportion."""
    rows = []

    for sid, sdf in events_df.groupby("session_id"):
        sdf = sdf.sort_values("timestamp_ms")

        if "rolling_choice_prop_a_clicks" not in sdf.columns:
            continue

        signal = sdf["rolling_choice_prop_a_clicks"].dropna().values
        times_ms = sdf.loc[sdf["rolling_choice_prop_a_clicks"].notna(),
                           "timestamp_ms"].values

        if len(signal) < 30:
            continue

        # Run BOCPD
        cp_prob, run_lengths = _bocpd_gaussian(signal)

        # Find peaks in changepoint signal (skip warmup period)
        warmup = min(20, len(signal) // 10)
        threshold = 0.3
        cp_indices = np.where((cp_prob > threshold) & (np.arange(len(cp_prob)) >= warmup))[0]

        # Merge nearby changepoints (within 20 clicks)
        if len(cp_indices) > 0:
            merged = [cp_indices[0]]
            for idx in cp_indices[1:]:
                if idx - merged[-1] > 20:
                    merged.append(idx)
                elif cp_prob[idx] > cp_prob[merged[-1]]:
                    merged[-1] = idx
            cp_indices = np.array(merged)

        for idx in cp_indices:
            if idx < len(times_ms):
                rows.append({
                    "session_id": sid,
                    "click_index": idx,
                    "timestamp_ms": times_ms[idx],
                    "elapsed_time_s": times_ms[idx] / 1000,
                    "cp_probability": cp_prob[idx],
                    "signal_value": signal[idx],
                })

        # Store full probability trace for plotting
        rows.append({
            "session_id": sid,
            "click_index": -1,  # sentinel for full trace
            "timestamp_ms": 0,
            "elapsed_time_s": 0,
            "cp_probability": 0,
            "signal_value": 0,
            "_cp_prob_trace": cp_prob.tolist(),
            "_signal_trace": signal.tolist(),
            "_times_trace": (times_ms / 1000).tolist(),
        })

    return pd.DataFrame(rows)


def _plot_changepoints(events_df: pd.DataFrame, cp_df: pd.DataFrame,
                       config: dict, fig_dir: str, fmt: str):
    """Plot choice trajectories with detected changepoints overlaid."""
    phase_boundaries_s = [p["start_ms"] / 1000
                          for p in config.get("phase_boundaries", [])[1:]]

    session_ids = cp_df[cp_df["click_index"] >= 0]["session_id"].unique() if len(cp_df) > 0 else []
    # Include sessions that have traces even if no CPs detected
    trace_sessions = cp_df[cp_df["click_index"] == -1]["session_id"].unique()
    session_ids = list(set(list(session_ids)) | set(trace_sessions))

    if len(session_ids) == 0:
        return

    n_cols = min(4, len(session_ids))
    n_rows = int(np.ceil(len(session_ids) / n_cols))
    fig, axes = plt.subplots(n_rows * 2, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows * 2),
                             gridspec_kw={"height_ratios": [3, 1] * n_rows})
    if n_rows * n_cols == 1:
        axes = axes.reshape(-1, 1)
    elif n_rows == 1:
        axes = axes.reshape(2, -1)
    else:
        # Reshape to (n_rows*2, n_cols)
        axes = axes.reshape(n_rows * 2, n_cols)

    for i, sid in enumerate(session_ids):
        col = i % n_cols
        row_block = i // n_cols
        ax_signal = axes[row_block * 2, col]
        ax_cp = axes[row_block * 2 + 1, col]

        # Get trace data
        trace_row = cp_df[(cp_df["session_id"] == sid) & (cp_df["click_index"] == -1)]
        if len(trace_row) == 0:
            ax_signal.set_visible(False)
            ax_cp.set_visible(False)
            continue

        trace = trace_row.iloc[0]
        if "_signal_trace" not in trace.index or trace["_signal_trace"] is None:
            ax_signal.set_visible(False)
            ax_cp.set_visible(False)
            continue

        signal = trace["_signal_trace"]
        times = trace["_times_trace"]
        cp_prob = trace["_cp_prob_trace"]

        # Plot signal
        ax_signal.plot(times, signal, color="steelblue", linewidth=0.8)
        ax_signal.set_ylabel("P(A)")
        ax_signal.set_ylim(0, 1)
        label = f"{sid[:8]}..." if len(str(sid)) > 8 else sid
        ax_signal.set_title(label, fontsize=9)

        # Mark detected changepoints
        sid_cps = cp_df[(cp_df["session_id"] == sid) & (cp_df["click_index"] >= 0)]
        for _, cp_row in sid_cps.iterrows():
            ax_signal.axvline(cp_row["elapsed_time_s"], color="red",
                              linewidth=1.5, alpha=0.7)
            ax_cp.axvline(cp_row["elapsed_time_s"], color="red",
                          linewidth=1.5, alpha=0.7)

        # Mark true phase boundaries
        for pb in phase_boundaries_s:
            ax_signal.axvline(pb, color="green", linestyle="--",
                              linewidth=1, alpha=0.5)
            ax_cp.axvline(pb, color="green", linestyle="--",
                          linewidth=1, alpha=0.5)

        # Plot CP probability
        ax_cp.plot(times, cp_prob, color="darkred", linewidth=0.8)
        ax_cp.axhline(0.3, color="gray", linestyle=":", alpha=0.5)
        ax_cp.set_ylabel("CP prob")
        ax_cp.set_xlabel("Time (s)")
        ax_cp.set_ylim(0, 1)

    # Hide unused axes
    total_plots = n_rows * n_cols
    for i in range(len(session_ids), total_plots):
        col = i % n_cols
        row_block = i // n_cols
        axes[row_block * 2, col].set_visible(False)
        axes[row_block * 2 + 1, col].set_visible(False)

    fig.suptitle("BOCPD: Detected Change Points (red) vs Phase Boundaries (green)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"changepoints_all.{fmt}"),
                dpi=config.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)


def _plot_changepoint_alignment(cp_df: pd.DataFrame, config: dict,
                                fig_dir: str, fmt: str):
    """Plot distribution of detected changepoints relative to phase boundaries."""
    phase_boundaries_s = [p["start_ms"] / 1000
                          for p in config.get("phase_boundaries", [])[1:]]
    if not phase_boundaries_s:
        return

    cps = cp_df[cp_df["click_index"] >= 0].copy()
    if len(cps) == 0:
        return

    # Compute distance to nearest phase boundary
    cps["nearest_boundary_s"] = cps["elapsed_time_s"].apply(
        lambda t: min(abs(t - b) for b in phase_boundaries_s)
    )
    cps["relative_to_nearest"] = cps["elapsed_time_s"].apply(
        lambda t: t - min(phase_boundaries_s, key=lambda b: abs(t - b))
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of CP times
    axes[0].hist(cps["elapsed_time_s"], bins=30, color="steelblue",
                 edgecolor="white", alpha=0.8)
    for pb in phase_boundaries_s:
        axes[0].axvline(pb, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Detected Changepoint Times")

    # Distance to nearest boundary
    axes[1].hist(cps["relative_to_nearest"], bins=30, color="teal",
                 edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Time relative to nearest boundary (s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Changepoint Lag from Phase Boundaries")

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"changepoint_alignment.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)


def _compute_boundary_alignment(cp_df: pd.DataFrame, config: dict) -> dict:
    """Quantify how well detected changepoints align with known boundaries."""
    phase_boundaries_s = [p["start_ms"] / 1000
                          for p in config.get("phase_boundaries", [])[1:]]
    if not phase_boundaries_s:
        return {}

    cps = cp_df[cp_df["click_index"] >= 0].copy()
    if len(cps) == 0:
        return {}

    # Per boundary: find closest detected CP for each session
    alignment = {}
    for boundary_s in phase_boundaries_s:
        boundary_label = f"{boundary_s:.0f}s"
        lags = []
        for sid, sid_cps in cps.groupby("session_id"):
            dists = (sid_cps["elapsed_time_s"] - boundary_s).values
            # Find closest CP within +/- 30s
            nearby = dists[(dists > -30) & (dists < 30)]
            if len(nearby) > 0:
                closest = nearby[np.argmin(np.abs(nearby))]
                lags.append(closest)

        alignment[boundary_label] = {
            "n_sessions_with_nearby_cp": len(lags),
            "mean_lag_s": np.mean(lags) if lags else np.nan,
            "median_lag_s": np.median(lags) if lags else np.nan,
            "sd_lag_s": np.std(lags) if lags else np.nan,
        }

    return alignment
