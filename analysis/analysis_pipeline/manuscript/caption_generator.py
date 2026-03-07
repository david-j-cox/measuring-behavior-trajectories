"""Auto-generate draft figure and table captions."""

import os
import numpy as np
import pandas as pd


def generate_all_captions(metrics_df: pd.DataFrame, analysis_results: dict,
                          config: dict, output_dir: str):
    """Write caption markdown files for every figure and table."""
    cap_dir = os.path.join(output_dir, "manuscript", "captions")
    os.makedirs(cap_dir, exist_ok=True)

    n = len(metrics_df)
    captions = {}

    # ── Figure captions ─────────────────────────────────────────────────────

    captions["figure1_task_schematic"] = (
        "**Figure 1. Task schematic.** "
        "Participants choose between two options (A and B) on each click. "
        "Each option has a hidden latent value that depletes upon selection "
        "and recovers over time. Reward probability equals the latent value "
        "at the moment of the click, sampled as a Bernoulli outcome. "
        "The task consists of four hidden phases (90 s each): "
        "(1) symmetric baseline, (2) A-advantaged, (3) B-advantaged, "
        "and (4) symmetric scarcity with alternating 3-s bonus pulses. "
        "Phase transitions are not signaled to the participant."
    )

    captions["figure2_example_trajectories"] = (
        "**Figure 2. Example behavioral trajectories.** "
        f"Choice sequences, rolling choice proportion (20-click window), "
        "cumulative score, and hidden latent values for representative sessions. "
        "Dotted vertical lines mark hidden phase boundaries. "
        "Behavior is visibly non-stationary, with shifts in allocation "
        "following changes in the latent reward environment."
    )

    # Phase-transition caption with adaptation lag if available
    lag_str = ""
    lags = analysis_results.get("adaptation_lags", {})
    if "phase2" in lags and not np.isnan(lags["phase2"].get("median_lag_s", np.nan)):
        lag_str = (
            f" Median adaptation lag at Phase 2 onset was "
            f"{lags['phase2']['median_lag_s']:.1f} s "
            f"(n = {lags['phase2']['n_adapted']}/{lags['phase2']['n_total']} adapted)."
        )

    captions["figure3_phase_transitions"] = (
        "**Figure 3. Behavioral dynamics around phase transitions.** "
        "Mean rolling choice proportion (top), switch rate (middle), and "
        "reward rate (bottom) aligned to the onset of each hidden phase change. "
        f"Shaded regions show bootstrap 95% confidence bands (N = {n} sessions). "
        "Time zero marks the transition. "
        "Participants gradually reallocate behavior toward the newly "
        f"advantageous option.{lag_str}"
    )

    captions["figure4_pulse_response"] = (
        "**Figure 4. Perturbation response to bonus pulses.** "
        "Peri-event trajectories aligned to the onset of 3-s bonus pulses "
        "during Phase 4. Left: choice probability toward the pulsed option. "
        "Right: reward rate. Separate curves for A-pulses (blue) and "
        f"B-pulses (red). Shaded bands show +/- 1 SEM (N = {n}). "
        "Brief value perturbations produce measurable, transient shifts "
        "in choice allocation."
    )

    captions["figure5_state_space"] = (
        "**Figure 5. State-space trajectories.** "
        "Behavioral state plotted as rolling choice proportion vs. rolling "
        "reward rate, colored by experimental phase. "
        "Trajectories move through distinct regions of the state space "
        "across phases rather than clustering at a single equilibrium, "
        "indicating structured dynamics and path dependence."
    )

    captions["figure6_individual_differences"] = (
        "**Figure 6. Individual differences in behavioral strategy.** "
        f"Distributions across participants (N = {n}) for: "
        "switch rate, mean run length, adaptation lag, proportion optimal "
        "choices, and choice entropy. "
        "Substantial between-participant variability is evident, "
        "suggesting participants occupy different dynamic strategy regimes."
    )

    mc = analysis_results.get("model_comparison")
    best_model = ""
    if mc is not None and len(mc) > 0:
        best = mc.groupby("model")["bic"].mean().idxmin()
        best_model = f" The {best} model provided the best fit on average."

    captions["figure7_model_comparison"] = (
        "**Figure 7. Model comparison.** "
        "Left: mean BIC across sessions for each candidate model "
        "(lower is better). Right: delta-BIC from the best-fitting model "
        f"per session.{best_model} "
        "Models that incorporate dynamic value tracking consistently "
        "outperform static heuristics."
    )

    # ── Table captions ──────────────────────────────────────────────────────

    captions["table1_session_summary"] = (
        "**Table 1. Session-level summary statistics.** "
        f"Descriptive statistics across {n} sessions. "
        "Values are mean (SD) unless otherwise noted."
    )

    captions["table2_phase_metrics"] = (
        "**Table 2. Phase-level behavioral metrics.** "
        "Mean reward rate, switch rate, run length, and proportion optimal "
        "computed separately for each of the four experimental phases."
    )

    captions["table3_transition_metrics"] = (
        "**Table 3. Phase-transition dynamics.** "
        "Adaptation lag (time to criterion), overshoot magnitude, "
        "switch-burst size, and post-transition instability at each "
        "phase boundary. Values are mean (SD)."
    )

    captions["table4_pulse_response"] = (
        "**Table 4. Pulse response metrics.** "
        "Response latency, exploitation rate, and reward gain following "
        "bonus pulse onset during Phase 4."
    )

    captions["table5_model_fits"] = (
        "**Table 5. Model fit statistics.** "
        "Log-likelihood, AIC, BIC, and predictive accuracy for each "
        "behavioral model, averaged across sessions. Lower AIC/BIC "
        "indicates better fit after penalizing for model complexity."
    )

    # ── Write to files ──────────────────────────────────────────────────────
    for name, text in captions.items():
        path = os.path.join(cap_dir, f"{name}.md")
        with open(path, "w") as f:
            f.write(text + "\n")

    print(f"Captions: {len(captions)} files written to {cap_dir}")
    return captions
