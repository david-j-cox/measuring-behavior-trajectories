"""Fractal analysis: DFA, Hurst exponent, and sample entropy."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_fractal_analysis(events_df: pd.DataFrame, config: dict,
                         output_dir: str) -> dict:
    """Run fractal/scaling analyses on choice time series."""
    results = {}
    fmt = config.get("plot_format", "png")
    fig_dir = os.path.join(output_dir, "figures", "fractal")
    os.makedirs(fig_dir, exist_ok=True)

    # DFA and Hurst exponent per session (full session and per-phase)
    dfa_results = _compute_dfa_all(events_df, config)
    results["dfa"] = dfa_results

    if len(dfa_results) > 0:
        _plot_dfa_summary(dfa_results, config, fig_dir, fmt)
        _plot_dfa_scaling(events_df, config, fig_dir, fmt)

    # Sample entropy
    sampen_results = _compute_sample_entropy_all(events_df, config)
    results["sample_entropy"] = sampen_results

    if len(sampen_results) > 0:
        _plot_entropy_summary(sampen_results, dfa_results, config, fig_dir, fmt)

    return results


def _compute_dfa_all(events_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute DFA exponent (Hurst-like) per session, overall and per phase."""
    rows = []
    phase_boundaries = config.get("phase_boundaries", [])

    for sid, sdf in events_df.groupby("session_id"):
        sdf = sdf.sort_values("timestamp_ms")
        choices = sdf["choice_a"].values

        if len(choices) < 50:
            continue

        # Full session DFA
        hurst, alpha_dfa = _compute_dfa(choices)
        rows.append({
            "session_id": sid,
            "scope": "full_session",
            "phase_id": 0,
            "n_clicks": len(choices),
            "hurst_exponent": hurst,
            "dfa_alpha": alpha_dfa,
        })

        # Per-phase DFA
        for pb in phase_boundaries:
            phase_df = sdf[sdf["phase_id"] == pb["id"]]
            if len(phase_df) < 30:
                rows.append({
                    "session_id": sid,
                    "scope": f"phase_{pb['id']}",
                    "phase_id": pb["id"],
                    "n_clicks": len(phase_df),
                    "hurst_exponent": np.nan,
                    "dfa_alpha": np.nan,
                })
                continue

            phase_choices = phase_df["choice_a"].values
            h, a = _compute_dfa(phase_choices)
            rows.append({
                "session_id": sid,
                "scope": f"phase_{pb['id']}",
                "phase_id": pb["id"],
                "n_clicks": len(phase_choices),
                "hurst_exponent": h,
                "dfa_alpha": a,
            })

    return pd.DataFrame(rows)


def _compute_dfa(signal, min_window=4, max_window=None):
    """
    Compute DFA exponent (alpha) for a binary choice signal.

    Implementation of Detrended Fluctuation Analysis (Peng et al., 1994).

    Returns (hurst_exponent, dfa_alpha).
    DFA alpha interpretation:
      < 0.5: anti-persistent (alternation tendency)
      = 0.5: uncorrelated (random)
      > 0.5: persistent (momentum / runs)
      = 1.0: 1/f noise
      > 1.0: non-stationary, unbounded correlations
    """
    if len(signal) < 20:
        return np.nan, np.nan

    try:
        signal = signal.astype(float)
        N = len(signal)

        # Step 1: Integrate (cumulative sum of deviations from mean)
        profile = np.cumsum(signal - signal.mean())

        # Step 2: Choose window sizes
        if max_window is None:
            max_window = N // 4
        n_sizes = min(20, max_window - min_window)
        if n_sizes < 4:
            return np.nan, np.nan

        window_sizes = np.unique(
            np.logspace(np.log10(min_window), np.log10(max_window), n_sizes
                        ).astype(int))
        window_sizes = window_sizes[window_sizes >= min_window]

        # Step 3: Compute fluctuation for each window size
        fluctuations = []
        valid_sizes = []
        for w in window_sizes:
            n_segments = N // w
            if n_segments < 1:
                continue
            F2 = []
            for seg in range(n_segments):
                segment = profile[seg * w:(seg + 1) * w]
                t = np.arange(w)
                # Linear detrend (order 1)
                coeffs = np.polyfit(t, segment, 1)
                trend = np.polyval(coeffs, t)
                F2.append(np.mean((segment - trend) ** 2))
            if F2:
                fluctuations.append(np.sqrt(np.mean(F2)))
                valid_sizes.append(w)

        if len(fluctuations) < 4:
            return np.nan, np.nan

        # Step 4: Log-log linear fit
        log_n = np.log(np.array(valid_sizes))
        log_F = np.log(np.array(fluctuations))
        alpha, _ = np.polyfit(log_n, log_F, 1)

        return alpha, alpha
    except Exception:
        return np.nan, np.nan


def _compute_sample_entropy_all(events_df: pd.DataFrame,
                                config: dict) -> pd.DataFrame:
    """Compute sample entropy of choice sequences."""
    rows = []
    phase_boundaries = config.get("phase_boundaries", [])

    for sid, sdf in events_df.groupby("session_id"):
        sdf = sdf.sort_values("timestamp_ms")

        # Use rolling choice proportion for entropy (binary is less informative)
        if "rolling_choice_prop_a_clicks" in sdf.columns:
            signal = sdf["rolling_choice_prop_a_clicks"].dropna().values
        else:
            signal = sdf["choice_a"].values.astype(float)

        if len(signal) < 30:
            continue

        # Full session
        se = _sample_entropy(signal)
        rows.append({
            "session_id": sid,
            "scope": "full_session",
            "phase_id": 0,
            "sample_entropy": se,
            "n_points": len(signal),
        })

        # Per phase
        for pb in phase_boundaries:
            phase_df = sdf[sdf["phase_id"] == pb["id"]]
            if "rolling_choice_prop_a_clicks" in phase_df.columns:
                phase_signal = phase_df["rolling_choice_prop_a_clicks"].dropna().values
            else:
                phase_signal = phase_df["choice_a"].values.astype(float)

            if len(phase_signal) < 20:
                se_phase = np.nan
            else:
                se_phase = _sample_entropy(phase_signal)

            rows.append({
                "session_id": sid,
                "scope": f"phase_{pb['id']}",
                "phase_id": pb["id"],
                "sample_entropy": se_phase,
                "n_points": len(phase_signal),
            })

    return pd.DataFrame(rows)


def _sample_entropy(signal, m=2, r_factor=0.2):
    """
    Compute sample entropy with embedding dimension m and tolerance r.

    Vectorized implementation of Richman & Moorman (2000).
    Subsamples long signals to keep runtime manageable.
    """
    N = len(signal)
    if N < m + 2:
        return np.nan

    # Subsample long signals to cap at ~300 points
    if N > 300:
        step = N // 300
        signal = signal[::step]
        N = len(signal)

    r = r_factor * np.std(signal)
    if r == 0:
        return np.nan

    try:
        def _count_matches_vec(template_length):
            templates = np.array([signal[i:i + template_length]
                                  for i in range(N - template_length)])
            n_templates = len(templates)
            count = 0
            for i in range(n_templates):
                # Vectorized comparison of template i against all j > i
                diffs = np.abs(templates[i + 1:] - templates[i])
                if diffs.ndim == 1:
                    matches = diffs < r
                else:
                    matches = np.all(diffs < r, axis=1) if diffs.ndim > 1 else diffs < r
                count += np.sum(matches)
            return count

        A = _count_matches_vec(m + 1)
        B = _count_matches_vec(m)

        if B == 0:
            return np.nan
        return -np.log(A / B) if A > 0 else np.nan
    except Exception:
        return np.nan


def _plot_dfa_summary(dfa_df: pd.DataFrame, config: dict,
                      fig_dir: str, fmt: str):
    """Plot DFA results: full-session distribution and phase comparison."""
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Full-session DFA alpha distribution
    full = dfa_df[dfa_df["scope"] == "full_session"]
    if len(full) > 0:
        axes[0].hist(full["dfa_alpha"].dropna(), bins=15, color="steelblue",
                     edgecolor="white", alpha=0.8)
        axes[0].axvline(0.5, color="red", linestyle="--", linewidth=1.5,
                        label="Random (0.5)")
        axes[0].axvline(1.0, color="orange", linestyle="--", linewidth=1.5,
                        label="1/f noise (1.0)")
        median_val = full["dfa_alpha"].median()
        axes[0].axvline(median_val, color="darkblue", linestyle="-",
                        linewidth=2, label=f"Median ({median_val:.2f})")
        axes[0].set_xlabel("DFA Alpha")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Full-Session DFA Exponent")
        axes[0].legend(fontsize=8)

    # Phase comparison
    phase_data = dfa_df[dfa_df["phase_id"] > 0].copy()
    if len(phase_data) > 0:
        phase_data["phase_label"] = "Phase " + phase_data["phase_id"].astype(str)
        sns.boxplot(data=phase_data, x="phase_label", y="dfa_alpha",
                    ax=axes[1], color="steelblue", width=0.5)
        axes[1].axhline(0.5, color="red", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("")
        axes[1].set_ylabel("DFA Alpha")
        axes[1].set_title("DFA by Phase")

    # Scatter: DFA alpha vs total clicks (check for length effects)
    if len(full) > 0:
        axes[2].scatter(full["n_clicks"], full["dfa_alpha"], color="steelblue",
                        alpha=0.6, s=30)
        axes[2].axhline(0.5, color="red", linestyle="--", alpha=0.5)
        axes[2].set_xlabel("Total Clicks")
        axes[2].set_ylabel("DFA Alpha")
        axes[2].set_title("DFA vs Session Length")

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"dfa_summary.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)


def _plot_dfa_scaling(events_df: pd.DataFrame, config: dict,
                      fig_dir: str, fmt: str):
    """Plot the DFA log-log scaling relation for example sessions."""
    session_ids = events_df["session_id"].unique()
    n_examples = min(6, len(session_ids))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i, sid in enumerate(session_ids[:n_examples]):
        ax = axes[i]
        sdf = events_df[events_df["session_id"] == sid].sort_values("timestamp_ms")
        choices = sdf["choice_a"].values.astype(float)

        if len(choices) < 50:
            ax.set_visible(False)
            continue

        # Compute DFA fluctuation function manually for plotting
        try:
            profile = np.cumsum(choices - choices.mean())
            # Window sizes
            min_win = 4
            max_win = len(profile) // 4
            n_sizes = min(20, max_win - min_win)
            if n_sizes < 4:
                ax.set_visible(False)
                continue
            window_sizes = np.unique(np.logspace(
                np.log10(min_win), np.log10(max_win), n_sizes
            ).astype(int))

            fluctuations = []
            for w in window_sizes:
                n_segments = len(profile) // w
                if n_segments < 1:
                    continue
                F2 = []
                for seg in range(n_segments):
                    segment = profile[seg * w:(seg + 1) * w]
                    t = np.arange(w)
                    # Linear detrend
                    coeffs = np.polyfit(t, segment, 1)
                    trend = np.polyval(coeffs, t)
                    F2.append(np.mean((segment - trend) ** 2))
                if F2:
                    fluctuations.append((w, np.sqrt(np.mean(F2))))

            if len(fluctuations) >= 4:
                ws, fs = zip(*fluctuations)
                ws, fs = np.array(ws), np.array(fs)
                # Log-log fit
                log_w = np.log10(ws)
                log_f = np.log10(fs)
                slope, intercept = np.polyfit(log_w, log_f, 1)

                ax.loglog(ws, fs, "o", color="steelblue", markersize=4)
                fit_line = 10**(slope * log_w + intercept)
                ax.loglog(ws, fit_line, "-", color="red", linewidth=1.5)

                label = f"{sid[:8]}..." if len(str(sid)) > 8 else sid
                ax.set_title(f"{label} (α={slope:.2f})", fontsize=9)
                ax.set_xlabel("Window size")
                ax.set_ylabel("F(n)")
        except Exception:
            ax.set_visible(False)

    for j in range(n_examples, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("DFA Log-Log Scaling (slope = α)", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"dfa_scaling.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)


def _plot_entropy_summary(sampen_df: pd.DataFrame, dfa_df: pd.DataFrame,
                          config: dict, fig_dir: str, fmt: str):
    """Plot sample entropy results and relationship with DFA."""
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    full_se = sampen_df[sampen_df["scope"] == "full_session"]
    full_dfa = dfa_df[dfa_df["scope"] == "full_session"]

    # Distribution
    if len(full_se) > 0:
        axes[0].hist(full_se["sample_entropy"].dropna(), bins=15,
                     color="teal", edgecolor="white", alpha=0.8)
        axes[0].set_xlabel("Sample Entropy")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Full-Session Sample Entropy")

    # Phase comparison
    phase_se = sampen_df[sampen_df["phase_id"] > 0].copy()
    if len(phase_se) > 0:
        phase_se["phase_label"] = "Phase " + phase_se["phase_id"].astype(str)
        sns.boxplot(data=phase_se, x="phase_label", y="sample_entropy",
                    ax=axes[1], color="teal", width=0.5)
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Sample Entropy")
        axes[1].set_title("Sample Entropy by Phase")

    # DFA vs Sample Entropy
    if len(full_se) > 0 and len(full_dfa) > 0:
        merged = full_se.merge(full_dfa[["session_id", "dfa_alpha"]],
                               on="session_id")
        if len(merged) > 0:
            axes[2].scatter(merged["dfa_alpha"], merged["sample_entropy"],
                            color="purple", alpha=0.6, s=30)
            axes[2].set_xlabel("DFA Alpha")
            axes[2].set_ylabel("Sample Entropy")
            axes[2].set_title("DFA vs Sample Entropy")

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"entropy_summary.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)
