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

    # ── Dynamical analysis figure captions ──────────────────────────────────

    captions["figure8_rqa"] = (
        "**Figure 8. Recurrence quantification analysis.** "
        "(A) Example recurrence plot for a representative session, showing "
        "the pattern of state revisitations across the choice sequence. "
        "(B) Distributions of RQA metrics across sessions: determinism, "
        "recurrence rate, and laminarity. High determinism indicates that "
        "revisitations occur in predictable sequences rather than at random. "
        f"(C) Determinism by experimental phase (N = {n})."
    )

    captions["figure9_dfa"] = (
        "**Figure 9. Detrended fluctuation analysis.** "
        "(A) Distribution of DFA exponents (alpha) across sessions. "
        "Dashed red line marks alpha = 0.5 (uncorrelated random process); "
        "dashed orange line marks alpha = 1.0 (1/f noise). "
        "The median exponent falls between these benchmarks, indicating "
        "persistent long-range temporal correlations in the choice sequence. "
        f"(B) DFA exponents by experimental phase (N = {n}). "
        "Values above 2.0 are clipped as unreliable estimates from short series."
    )

    captions["figure10_entropy"] = (
        "**Figure 10. Sample entropy.** "
        "(A) Distribution of sample entropy values across sessions. "
        "Higher values indicate less predictable choice sequences. "
        "(B) Sample entropy by experimental phase. Phase 4 (scarcity + pulses) "
        f"shows elevated entropy relative to the asymmetric phases (N = {n})."
    )

    captions["figure11_ccm"] = (
        "**Figure 11. Convergent cross-mapping.** "
        "(A) Group-average cross-map skill (rho) as a function of library size "
        "for the reward-to-choice and choice-to-reward directions. "
        "Increasing rho with library size indicates causal coupling. "
        "(B) Per-session asymmetry: each point is one session, plotted by "
        "cross-map skill in each direction. Points above the diagonal indicate "
        f"stronger reward-to-choice than choice-to-reward causation (N = {n})."
    )

    captions["figure12_smap"] = (
        "**Figure 12. S-Map nonlinearity analysis.** "
        "(A) Linear (simplex) vs. nonlinear (S-Map) prediction skill for each "
        "session. Points above the diagonal indicate state-dependent dynamics. "
        "(B) Distribution of nonlinearity scores (delta-rho = S-Map rho minus "
        f"simplex rho). Positive values indicate nonlinear dynamics (N = {n})."
    )

    captions["figure13_hmm"] = (
        "**Figure 13. Hidden Markov model analysis.** "
        "(A) Example session with choice trajectory colored by HMM-assigned "
        "latent state. Vertical dashed lines mark hidden phase boundaries. "
        "(B) Mean state occupancy by experimental phase, aggregated across "
        f"sessions (N = {n}). State transitions cluster near phase boundaries, "
        "suggesting that HMM states capture environmentally driven behavioral modes."
    )

    captions["figure14_phenotypes"] = (
        "**Figure 14. Behavioral phenotypes.** "
        "(A) PCA biplot of session-level behavioral and dynamical features, "
        "colored by GMM cluster assignment. Feature loading vectors (red arrows) "
        "show how each metric contributes to the principal components. "
        "(B) Cluster profile heatmap showing mean z-scored feature values for "
        f"each cluster (N = {n})."
    )

    captions["figure15_null_comparison"] = (
        "**Figure 15. Null model comparison.** "
        "Dynamical metrics (DFA alpha, sample entropy, recurrence rate, "
        "determinism, laminarity, trapping time) computed on real behavioral data "
        "vs. simulated data from three null models: random choice, matching-law "
        f"allocation, and win-stay/lose-shift. Error bars show +/- 1 SEM (N = {n}). "
        "Differences between real and null distributions indicate dynamical "
        "structure in behavior that simple models cannot reproduce."
    )

    captions["figure16_robustness"] = (
        "**Figure 16. Sensitivity and robustness checks.** "
        "(A) Mean DFA exponent as a function of rolling window size, showing "
        "that the qualitative conclusion (alpha > 0.5) is robust across window "
        "sizes. (B) Cross-correlation matrix of DFA exponents computed at "
        "different minimum window sizes, confirming high rank-order stability."
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

    captions["table5_matching_law"] = (
        "**Table 5. Generalized matching law (Baum, 1974) parameter estimates.** "
        "Sensitivity (*s*), bias (*b*), and variance explained (R\u00b2) per "
        "session, computed via log-ratio regression on non-overlapping bins "
        "of local reinforcement and response ratios. "
        "Classification: undermatching (*s* < 0.9), strict matching "
        "(0.9 \u2264 *s* \u2264 1.1), overmatching (*s* > 1.1). Summary row "
        "reports M (SD) across sessions."
    )

    captions["table6_model_fits"] = (
        "**Table 6. Trial-level model fit statistics.** "
        "Log-likelihood, AIC, BIC, and predictive accuracy for each "
        "trial-level behavioral model, averaged across sessions. Lower AIC/BIC "
        "indicates better fit after penalizing for model complexity. "
        "Matching law models are reported separately in Table 5 because "
        "they use session-level log-ratio regression rather than "
        "trial-level likelihoods."
    )

    captions["table7_dynamical_summary"] = (
        "**Table 7. Dynamical analysis summary.** "
        f"Mean, standard deviation, median, and range for each dynamical metric "
        f"across sessions (N = {n}). RQA = recurrence quantification analysis; "
        "DFA = detrended fluctuation analysis; EDM = empirical dynamic modeling; "
        "S-Map = state-dependent mapping."
    )

    # ── Write to files ──────────────────────────────────────────────────────
    for name, text in captions.items():
        path = os.path.join(cap_dir, f"{name}.md")
        with open(path, "w") as f:
            f.write(text + "\n")

    print(f"Captions: {len(captions)} files written to {cap_dir}")
    return captions
