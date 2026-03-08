"""Sensitivity and robustness checks for key analysis parameters."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

from analysis_pipeline.fractal_analysis import compute_dfa
from analysis_pipeline.changepoint_analysis import bocpd_gaussian


def run_robustness_checks(events_df: pd.DataFrame, config: dict,
                          output_dir: str) -> dict:
    """
    Test whether key results are stable across parameter variations.

    Checks:
      1. Rolling window sensitivity (DFA alpha, mean autocorrelation)
      2. DFA min-window range sensitivity
      3. BOCPD hazard rate sensitivity (# changepoints, boundary alignment)

    Returns a dict with the full results and saves summary CSV + figure.
    """
    results = {}
    fmt = config.get("plot_format", "png")
    fig_dir = os.path.join(output_dir, "figures", "robustness")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    summary_rows = []

    # ---- 1. Rolling window sensitivity ----
    print("  [1/3] Rolling window sensitivity ...")
    rw_results, rw_rows = _rolling_window_sensitivity(events_df, config)
    results["rolling_window"] = rw_results
    summary_rows.extend(rw_rows)

    # ---- 2. DFA min-window range sensitivity ----
    print("  [2/3] DFA window range sensitivity ...")
    dfa_results, dfa_rows = _dfa_window_sensitivity(events_df, config)
    results["dfa_window"] = dfa_results
    summary_rows.extend(dfa_rows)

    # ---- 3. BOCPD hazard rate sensitivity ----
    print("  [3/3] BOCPD hazard rate sensitivity ...")
    bocpd_results, bocpd_rows = _bocpd_hazard_sensitivity(events_df, config)
    results["bocpd_hazard"] = bocpd_results
    summary_rows.extend(bocpd_rows)

    # ---- Save summary CSV ----
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(tables_dir, "robustness_summary.csv"), index=False
    )
    results["summary"] = summary_df

    # ---- Generate multi-panel figure ----
    _plot_robustness(rw_results, dfa_results, bocpd_results, config,
                     fig_dir, fmt)

    return results


# ====================================================================
# 1. Rolling window sensitivity
# ====================================================================

def _rolling_window_sensitivity(events_df, config):
    """Re-compute DFA alpha and mean autocorrelation under different
    rolling-window sizes for the choice proportion signal."""
    window_sizes = [10, 15, 20, 30]
    default_window = config.get("rolling_window_clicks", 20)

    records = {}  # window -> {session_id -> {dfa_alpha, mean_acf}}

    for win in window_sizes:
        session_metrics = []
        for sid, sdf in events_df.groupby("session_id"):
            sdf = sdf.sort_values("timestamp_ms")
            choices = sdf["choice_a"].values.astype(float)
            if len(choices) < max(win, 50):
                continue

            # Recompute rolling choice proportion with this window
            rolling_prop = pd.Series(choices).rolling(win, min_periods=1).mean().values

            # DFA on the rolling signal
            _, alpha = compute_dfa(rolling_prop)

            # Mean autocorrelation (lags 1-10)
            x = rolling_prop - rolling_prop.mean()
            var = np.var(x)
            if var > 0 and len(x) > 20:
                acfs = []
                for lag in range(1, 11):
                    acf = np.mean(x[lag:] * x[:-lag]) / var
                    acfs.append(acf)
                mean_acf = np.nanmean(acfs)
            else:
                mean_acf = np.nan

            session_metrics.append({
                "session_id": sid,
                "dfa_alpha": alpha,
                "mean_acf": mean_acf,
            })

        records[win] = pd.DataFrame(session_metrics)

    # Build summary rows
    summary_rows = []
    default_df = records.get(default_window, pd.DataFrame())

    for win, df in records.items():
        if len(df) == 0:
            continue

        # Correlation with default
        corr_alpha = np.nan
        corr_acf = np.nan
        if len(default_df) > 0 and win != default_window:
            merged = df.merge(default_df, on="session_id",
                              suffixes=("", "_default"))
            valid = merged.dropna(subset=["dfa_alpha", "dfa_alpha_default"])
            if len(valid) >= 3:
                corr_alpha = np.corrcoef(valid["dfa_alpha"],
                                         valid["dfa_alpha_default"])[0, 1]
            valid_acf = merged.dropna(subset=["mean_acf", "mean_acf_default"])
            if len(valid_acf) >= 3:
                corr_acf = np.corrcoef(valid_acf["mean_acf"],
                                       valid_acf["mean_acf_default"])[0, 1]
        elif win == default_window:
            corr_alpha = 1.0
            corr_acf = 1.0

        summary_rows.append({
            "analysis": "rolling_window",
            "parameter": "window_clicks",
            "value": win,
            "metric": "dfa_alpha",
            "mean": df["dfa_alpha"].mean(),
            "sd": df["dfa_alpha"].std(),
            "correlation_with_default": corr_alpha,
        })
        summary_rows.append({
            "analysis": "rolling_window",
            "parameter": "window_clicks",
            "value": win,
            "metric": "mean_acf",
            "mean": df["mean_acf"].mean(),
            "sd": df["mean_acf"].std(),
            "correlation_with_default": corr_acf,
        })

    return records, summary_rows


# ====================================================================
# 2. DFA min-window range sensitivity
# ====================================================================

def _dfa_window_sensitivity(events_df, config):
    """Run DFA with different minimum window sizes and compare results."""
    min_windows = [4, 8, 12, 16]
    default_min = 4  # the default in _compute_dfa

    records = {}  # min_win -> {session_id -> dfa_alpha}

    for min_win in min_windows:
        session_metrics = []
        for sid, sdf in events_df.groupby("session_id"):
            sdf = sdf.sort_values("timestamp_ms")
            choices = sdf["choice_a"].values.astype(float)
            if len(choices) < 50:
                continue

            _, alpha = compute_dfa(choices, min_window=min_win)
            session_metrics.append({
                "session_id": sid,
                "dfa_alpha": alpha,
            })

        records[min_win] = pd.DataFrame(session_metrics)

    # Pairwise correlations
    pair_corrs = {}
    for (w1, df1), (w2, df2) in combinations(records.items(), 2):
        merged = df1.merge(df2, on="session_id", suffixes=(f"_{w1}", f"_{w2}"))
        col1 = f"dfa_alpha_{w1}"
        col2 = f"dfa_alpha_{w2}"
        valid = merged.dropna(subset=[col1, col2])
        if len(valid) >= 3:
            pair_corrs[(w1, w2)] = np.corrcoef(valid[col1], valid[col2])[0, 1]
        else:
            pair_corrs[(w1, w2)] = np.nan

    # Summary rows
    summary_rows = []
    default_df = records.get(default_min, pd.DataFrame())

    for min_win, df in records.items():
        if len(df) == 0:
            continue
        corr = np.nan
        if len(default_df) > 0 and min_win != default_min:
            merged = df.merge(default_df, on="session_id",
                              suffixes=("", "_default"))
            valid = merged.dropna(subset=["dfa_alpha", "dfa_alpha_default"])
            if len(valid) >= 3:
                corr = np.corrcoef(valid["dfa_alpha"],
                                   valid["dfa_alpha_default"])[0, 1]
        elif min_win == default_min:
            corr = 1.0

        summary_rows.append({
            "analysis": "dfa_min_window",
            "parameter": "min_window",
            "value": min_win,
            "metric": "dfa_alpha",
            "mean": df["dfa_alpha"].mean(),
            "sd": df["dfa_alpha"].std(),
            "correlation_with_default": corr,
        })

    return {"records": records, "pair_correlations": pair_corrs}, summary_rows


# ====================================================================
# 3. BOCPD hazard rate sensitivity
# ====================================================================

def _bocpd_hazard_sensitivity(events_df, config):
    """Run BOCPD with different hazard rates, count changepoints,
    and measure alignment with true phase boundaries."""
    hazard_rates = [1/50, 1/100, 1/200, 1/300]
    default_hazard = 1/100
    phase_boundaries_s = [p["start_ms"] / 1000
                          for p in config.get("phase_boundaries", [])[1:]]

    records = {}  # hazard -> DataFrame of per-session results

    for hz in hazard_rates:
        session_metrics = []
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
            cp_prob, run_lengths = bocpd_gaussian(signal, hazard_rate=hz)

            # Find changepoints (same logic as _detect_changepoints_all)
            warmup = min(20, len(signal) // 10)
            threshold = 0.3
            cp_indices = np.where(
                (cp_prob > threshold) & (np.arange(len(cp_prob)) >= warmup)
            )[0]

            # Merge nearby changepoints (within 20 clicks)
            if len(cp_indices) > 0:
                merged = [cp_indices[0]]
                for idx in cp_indices[1:]:
                    if idx - merged[-1] > 20:
                        merged.append(idx)
                    elif cp_prob[idx] > cp_prob[merged[-1]]:
                        merged[-1] = idx
                cp_indices = np.array(merged)

            n_cps = len(cp_indices)

            # Compute alignment with boundaries
            mean_alignment = np.nan
            if n_cps > 0 and phase_boundaries_s:
                cp_times_s = times_ms[cp_indices] / 1000
                min_dists = []
                for boundary_s in phase_boundaries_s:
                    dists = np.abs(cp_times_s - boundary_s)
                    if len(dists) > 0:
                        min_dists.append(np.min(dists))
                mean_alignment = np.mean(min_dists) if min_dists else np.nan

            session_metrics.append({
                "session_id": sid,
                "n_changepoints": n_cps,
                "mean_boundary_distance_s": mean_alignment,
            })

        records[hz] = pd.DataFrame(session_metrics)

    # Summary rows
    summary_rows = []
    default_df = records.get(default_hazard, pd.DataFrame())

    for hz, df in records.items():
        if len(df) == 0:
            continue

        corr_n = np.nan
        if len(default_df) > 0 and hz != default_hazard:
            merged = df.merge(default_df, on="session_id",
                              suffixes=("", "_default"))
            valid = merged.dropna(
                subset=["n_changepoints", "n_changepoints_default"])
            if len(valid) >= 3:
                corr_n = np.corrcoef(valid["n_changepoints"],
                                     valid["n_changepoints_default"])[0, 1]
        elif hz == default_hazard:
            corr_n = 1.0

        summary_rows.append({
            "analysis": "bocpd_hazard",
            "parameter": "hazard_rate",
            "value": hz,
            "metric": "n_changepoints",
            "mean": df["n_changepoints"].mean(),
            "sd": df["n_changepoints"].std(),
            "correlation_with_default": corr_n,
        })
        summary_rows.append({
            "analysis": "bocpd_hazard",
            "parameter": "hazard_rate",
            "value": hz,
            "metric": "mean_boundary_distance_s",
            "mean": df["mean_boundary_distance_s"].mean(),
            "sd": df["mean_boundary_distance_s"].std(),
            "correlation_with_default": np.nan,
        })

    return records, summary_rows


# ====================================================================
# Plot
# ====================================================================

def _plot_robustness(rw_results, dfa_results, bocpd_results, config,
                     fig_dir, fmt):
    """Generate a 2x2 multi-panel robustness figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ---- Panel 1: DFA alpha vs rolling window size ----
    ax = axes[0, 0]
    windows = sorted(rw_results.keys())
    means, sds = [], []
    for w in windows:
        df = rw_results[w]
        vals = df["dfa_alpha"].dropna()
        means.append(vals.mean() if len(vals) > 0 else np.nan)
        sds.append(vals.std() if len(vals) > 0 else np.nan)
    means, sds = np.array(means), np.array(sds)

    ax.errorbar(windows, means, yerr=sds, fmt="o-", color="steelblue",
                capsize=4, linewidth=1.5, markersize=6)
    default_win = config.get("rolling_window_clicks", 20)
    ax.axvline(default_win, color="red", linestyle="--", alpha=0.5,
               label=f"Default ({default_win})")
    ax.set_xlabel("Rolling Window Size (clicks)")
    ax.set_ylabel("DFA Alpha")
    ax.set_title("DFA Alpha vs Rolling Window")
    ax.legend(fontsize=8)

    # ---- Panel 2: DFA alpha correlation across min-window settings ----
    ax = axes[0, 1]
    dfa_records = dfa_results["records"]
    pair_corrs = dfa_results["pair_correlations"]
    min_windows = sorted(dfa_records.keys())

    # Build correlation matrix
    n_mw = len(min_windows)
    corr_matrix = np.ones((n_mw, n_mw))
    for (w1, w2), r in pair_corrs.items():
        i1 = min_windows.index(w1)
        i2 = min_windows.index(w2)
        corr_matrix[i1, i2] = r
        corr_matrix[i2, i1] = r

    im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_mw))
    ax.set_xticklabels(min_windows)
    ax.set_yticks(range(n_mw))
    ax.set_yticklabels(min_windows)
    ax.set_xlabel("DFA Min Window")
    ax.set_ylabel("DFA Min Window")
    ax.set_title("DFA Alpha Cross-Correlation")

    # Annotate cells
    for i in range(n_mw):
        for j in range(n_mw):
            val = corr_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if val > 0.4 else "white")

    fig.colorbar(im, ax=ax, shrink=0.8)

    # ---- Panel 3: Number of changepoints vs hazard rate ----
    ax = axes[1, 0]
    hazard_rates = sorted(bocpd_results.keys())
    cp_means, cp_sds = [], []
    for hz in hazard_rates:
        df = bocpd_results[hz]
        vals = df["n_changepoints"].dropna()
        cp_means.append(vals.mean() if len(vals) > 0 else np.nan)
        cp_sds.append(vals.std() if len(vals) > 0 else np.nan)
    cp_means, cp_sds = np.array(cp_means), np.array(cp_sds)

    # Use 1/hazard_rate for x-axis (expected run length)
    expected_run = [1/h for h in hazard_rates]
    ax.errorbar(expected_run, cp_means, yerr=cp_sds, fmt="o-",
                color="darkred", capsize=4, linewidth=1.5, markersize=6)
    ax.axvline(100, color="red", linestyle="--", alpha=0.5,
               label="Default (1/100)")
    ax.set_xlabel("Expected Run Length (1/hazard)")
    ax.set_ylabel("Detected Changepoints")
    ax.set_title("Changepoints vs Hazard Rate")
    ax.legend(fontsize=8)

    # ---- Panel 4: Boundary alignment vs hazard rate ----
    ax = axes[1, 1]
    align_means, align_sds = [], []
    for hz in hazard_rates:
        df = bocpd_results[hz]
        vals = df["mean_boundary_distance_s"].dropna()
        align_means.append(vals.mean() if len(vals) > 0 else np.nan)
        align_sds.append(vals.std() if len(vals) > 0 else np.nan)
    align_means, align_sds = np.array(align_means), np.array(align_sds)

    ax.errorbar(expected_run, align_means, yerr=align_sds, fmt="s-",
                color="teal", capsize=4, linewidth=1.5, markersize=6)
    ax.axvline(100, color="red", linestyle="--", alpha=0.5,
               label="Default (1/100)")
    ax.set_xlabel("Expected Run Length (1/hazard)")
    ax.set_ylabel("Mean Distance to Nearest Boundary (s)")
    ax.set_title("Changepoint-Boundary Alignment vs Hazard Rate")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"robustness_checks.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)
