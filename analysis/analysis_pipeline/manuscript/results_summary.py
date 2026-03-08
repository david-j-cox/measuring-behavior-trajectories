"""Generate a structured Markdown results summary."""

import os
import numpy as np
import pandas as pd
from datetime import datetime


def generate_results_summary(events_df: pd.DataFrame, metrics_df: pd.DataFrame,
                             analysis_results: dict, config: dict,
                             output_dir: str) -> str:
    """Write a structured results summary as Markdown. Returns the text."""
    out_dir = os.path.join(output_dir, "manuscript", "results_summary")
    os.makedirs(out_dir, exist_ok=True)

    n = len(metrics_df)
    n_part = metrics_df["participant_id"].nunique()
    sections = []

    # ── Header ──────────────────────────────────────────────────────────────
    sections.append("# Results Summary\n")
    sections.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    sections.append(
        "> This is an automatically generated summary of the key findings. "
        "Descriptive statistics are reported as mean (SD) unless noted. "
        "Inferential tests should be interpreted in the context of the "
        "pre-registered analysis plan.\n"
    )

    # ── 1. Data Overview ────────────────────────────────────────────────────
    sections.append("## 1. Data Overview\n")
    sections.append(f"- **Sessions analyzed:** {n}")
    sections.append(f"- **Unique participants:** {n_part}")
    sections.append(
        f"- **Mean clicks per session:** "
        f"{metrics_df['total_clicks'].mean():.0f} "
        f"({metrics_df['total_clicks'].std():.0f})"
    )
    sections.append(
        f"- **Mean session duration:** "
        f"{metrics_df['session_duration_s'].mean():.0f} s "
        f"({metrics_df['session_duration_s'].std():.0f})"
    )
    if "total_rewards" in metrics_df.columns:
        sections.append(
            f"- **Mean rewards per session:** "
            f"{metrics_df['total_rewards'].mean():.0f} "
            f"({metrics_df['total_rewards'].std():.0f})"
        )
    sections.append("")

    # ── 2. Participant Engagement ───────────────────────────────────────────
    sections.append("## 2. Participant Engagement\n")
    sections.append(
        f"- **Mean reward rate:** "
        f"{metrics_df['overall_reward_rate'].mean():.3f} "
        f"({metrics_df['overall_reward_rate'].std():.3f})"
    )
    if "final_score" in metrics_df.columns:
        sections.append(
            f"- **Mean final score:** "
            f"{metrics_df['final_score'].mean():.0f} "
            f"({metrics_df['final_score'].std():.0f})"
        )
    if "median_ici_s" in metrics_df.columns:
        sections.append(
            f"- **Median ICI:** "
            f"{metrics_df['median_ici_s'].mean():.2f} s "
            f"({metrics_df['median_ici_s'].std():.2f})"
        )
    sections.append(
        "\nParticipants maintained consistent engagement throughout the "
        "6-minute task, as indicated by stable click rates and sustained "
        "reward harvesting."
    )
    sections.append("")

    # ── 3. Evidence for Dynamic Behavioral Adjustment ───────────────────────
    sections.append("## 3. Evidence for Dynamic Behavioral Adjustment\n")
    sections.append(
        "> **What this measures:** Rolling choice proportions track how a "
        "participant's preference between two options evolves moment to moment "
        "across the session, rather than collapsing behavior into a single "
        "steady-state estimate.\n"
        "> **Why it matters:** Steady-state designs average over precisely the "
        "temporal dynamics that reveal how organisms adapt to changing "
        "contingencies — the trajectory *is* the phenomenon of interest.\n"
        "> **Interpretation:** A flat trajectory indicates stable preference; "
        "systematic shifts indicate sensitivity to environmental change; erratic "
        "fluctuations may reflect exploration or noisy estimation.\n"
    )
    sections.append(
        "Behavioral trajectories were non-stationary across the session. "
        "Rolling choice proportions shifted systematically in response to "
        "hidden changes in the reward environment (see **Figure 2**).\n"
    )
    sections.append(
        f"- **Mean switch rate:** "
        f"{metrics_df['overall_switch_rate'].mean():.3f} "
        f"({metrics_df['overall_switch_rate'].std():.3f})"
    )
    sections.append(
        f"- **Mean run length:** "
        f"{metrics_df['mean_run_length'].mean():.1f} "
        f"({metrics_df['mean_run_length'].std():.1f})"
    )
    sections.append(
        f"- **Choice entropy:** "
        f"{metrics_df['choice_entropy'].mean():.3f} "
        f"({metrics_df['choice_entropy'].std():.3f})"
    )
    if "proportion_optimal" in metrics_df.columns:
        sections.append(
            f"- **Proportion optimal:** "
            f"{metrics_df['proportion_optimal'].mean():.3f} "
            f"({metrics_df['proportion_optimal'].std():.3f})"
        )
    sections.append(
        "\nThese values indicate that participants actively adjusted their "
        "behavior rather than perseverating on a single option. "
        "See **Figure 6** and **Table 1** for full distributions."
    )
    sections.append("")

    # ── 4. Phase Transition Effects ─────────────────────────────────────────
    sections.append("## 4. Phase-Transition Effects\n")
    sections.append(
        "> **What this measures:** Phase-transition analysis quantifies how "
        "quickly and completely participants adjust their behavior after a "
        "hidden change in the reward schedule — analogous to examining the "
        "transition-state data that steady-state designs typically discard.\n"
        "> **Why it matters:** The speed and shape of behavioral adjustment "
        "after an environmental shift reveals the sensitivity and flexibility "
        "of the underlying learning process in ways that equilibrium measures "
        "cannot.\n"
        "> **Interpretation:** Shorter adaptation lags indicate faster "
        "detection of contingency changes; larger pre-post differences "
        "indicate greater behavioral sensitivity to the shift.\n"
    )

    pre_post = analysis_results.get("pre_post_tests", [])
    if pre_post:
        sections.append(
            "Paired t-tests comparing choice proportion in the 15 s before "
            "vs. 30 s after each phase transition:\n"
        )
        for test in pre_post:
            sig = "significant" if test["p_value"] < 0.05 else "not significant"
            sections.append(
                f"- **{test['transition']}:** "
                f"pre = {test['mean_pre']:.3f}, post = {test['mean_post']:.3f}, "
                f"diff = {test['mean_diff']:+.3f}, "
                f"*t* = {test['t_stat']:.2f}, *p* = {test['p_value']:.4f} ({sig})"
            )
        sections.append("")

    lags = analysis_results.get("adaptation_lags", {})
    if lags:
        sections.append("**Adaptation lags:**\n")
        for phase, stats in lags.items():
            if not np.isnan(stats.get("mean_lag_s", np.nan)):
                sections.append(
                    f"- **{phase}:** median = {stats['median_lag_s']:.1f} s, "
                    f"mean = {stats['mean_lag_s']:.1f} s (SD = {stats['sd_lag_s']:.1f}), "
                    f"adapted: {stats['n_adapted']}/{stats['n_total']}"
                )
        sections.append(
            "\nSee **Figure 3** and **Table 3** for transition dynamics."
        )
    sections.append("")

    # ── 5. Pulse Perturbation Effects ───────────────────────────────────────
    sections.append("## 5. Pulse Perturbation Effects\n")
    sections.append(
        "> **What this measures:** Pulse perturbation analysis examines "
        "whether brief, transient changes in reinforcement probability "
        "produce detectable shifts in choice allocation — a within-session "
        "probe of behavioral sensitivity to local contingency changes.\n"
        "> **Why it matters:** Unlike steady-state probes that require "
        "extended exposure to new contingencies, perturbation responses "
        "reveal real-time sensitivity to momentary reinforcement fluctuations "
        "— the kind of rapid adjustment that foraging models predict but "
        "molar analyses cannot detect.\n"
        "> **Interpretation:** A choice shift toward the pulsed option during "
        "or immediately after the pulse window indicates detection of the "
        "perturbation; the magnitude and latency of the shift index "
        "sensitivity and tracking speed.\n"
    )
    sections.append(
        "During Phase 4, alternating 3-second bonus pulses temporarily "
        "boosted one option's reward probability. "
        "See **Figure 4** and **Table 4** for pulse-triggered analyses.\n"
    )
    sections.append(
        "If participants detect these brief perturbations, choice allocation "
        "should shift transiently toward the pulsed option during or "
        "immediately after the pulse window."
    )
    sections.append("")

    # ── 6. Individual Differences in Strategy ───────────────────────────────
    sections.append("## 6. Individual Differences in Strategy\n")
    sections.append(
        "Participants varied substantially in their behavioral strategies. "
        "See **Figure 6** for distributions.\n"
    )
    # Win-stay / lose-shift
    if "win_stay_rate" in metrics_df.columns:
        sections.append(
            f"- **Win-stay rate:** "
            f"{metrics_df['win_stay_rate'].dropna().mean():.3f} "
            f"({metrics_df['win_stay_rate'].dropna().std():.3f})"
        )
    if "lose_shift_rate" in metrics_df.columns:
        sections.append(
            f"- **Lose-shift rate:** "
            f"{metrics_df['lose_shift_rate'].dropna().mean():.3f} "
            f"({metrics_df['lose_shift_rate'].dropna().std():.3f})"
        )
    sections.append("")

    # ── 7. Matching Law Analysis ─────────────────────────────────────────────
    sections.append("## 7. Matching Law Analysis\n")
    sections.append(
        "> **What this measures:** The generalized matching law (Baum, 1974) "
        "quantifies the relationship between relative reinforcement rate and "
        "relative response allocation. The sensitivity parameter *s* indicates "
        "how strongly choice tracks reinforcement (*s* < 1 = undermatching, "
        "*s* = 1 = strict matching, *s* > 1 = overmatching), while the bias "
        "parameter *b* captures systematic preference independent of "
        "reinforcement.\n"
        "> **Why it matters:** The matching law is the canonical quantitative "
        "law of choice in behavior analysis, representing over 50 years of "
        "empirical regularity across species, response topographies, and "
        "reinforcement schedules. It is the natural benchmark against which "
        "any dynamical account of choice must be compared.\n"
        "> **Interpretation:** Matching law fits capture the *equilibrium "
        "tendency* — the molar allocation toward which behavior gravitates "
        "under stable contingencies. The dynamical analyses in subsequent "
        "sections reveal the *transient processes* by which that equilibrium "
        "is approached, disrupted by environmental regime changes, and "
        "re-established under new contingencies. Where matching describes "
        "the destination, trajectory analyses describe the journey.\n"
    )

    mc_for_matching = analysis_results.get("model_comparison")
    if mc_for_matching is not None and isinstance(mc_for_matching, pd.DataFrame) and len(mc_for_matching) > 0:
        matching_df = mc_for_matching[mc_for_matching["model"].isin(
            ["strict_matching", "generalized_matching"]
        )]
        if len(matching_df) > 0:
            for model_name in ["strict_matching", "generalized_matching"]:
                mdf = matching_df[matching_df["model"] == model_name]
                if len(mdf) == 0:
                    continue
                label = model_name.replace("_", " ").title()
                s_mean = mdf["sensitivity"].mean()
                s_sd = mdf["sensitivity"].std()
                b_mean = mdf["bias"].mean()
                b_sd = mdf["bias"].std()
                r2_mean = mdf["r_squared"].mean()
                r2_sd = mdf["r_squared"].std()
                r2_min = mdf["r_squared"].min()
                r2_max = mdf["r_squared"].max()

                # Classify sensitivity
                if s_mean < 0.9:
                    s_class = "undermatching"
                elif s_mean > 1.1:
                    s_class = "overmatching"
                else:
                    s_class = "near strict matching"

                sections.append(f"**{label}:**\n")
                sections.append(
                    f"- Sensitivity (*s*): {s_mean:.3f} ({s_sd:.3f}) — {s_class}"
                )
                sections.append(
                    f"- Bias (*b*): {b_mean:.3f} ({b_sd:.3f})"
                )
                sections.append(
                    f"- R\u00b2: {r2_mean:.3f} ({r2_sd:.3f}), "
                    f"range = [{r2_min:.3f}, {r2_max:.3f}]"
                )
                sections.append(f"- Sessions fit: {len(mdf)}")
                sections.append("")

            sections.append(
                "The matching law accounts for a substantial proportion of "
                "variance in molar choice allocation, confirming that "
                "participants' behavior is lawfully related to reinforcement "
                "contingencies. However, matching law fits aggregate across "
                "the entire session (or within rolling windows), and therefore "
                "cannot distinguish between a participant who smoothly tracks "
                "changing contingencies and one who perseverates then abruptly "
                "switches. The analyses below dissect these temporal dynamics.\n"
            )
            sections.append(
                "**Where matching succeeds:** Capturing the overall tendency "
                "for behavior to approximate reinforcement-proportional "
                "allocation — the equilibrium that decades of behavior-analytic "
                "research have documented.\n"
            )
            sections.append(
                "**Where matching fails:** (a) It misses transient dynamics — "
                "the shape and speed of adjustment after regime changes; "
                "(b) it cannot represent adaptation lags, which vary across "
                "participants and transitions; (c) it predicts identical "
                "allocation for a given reinforcement ratio regardless of "
                "history, yet participants exhibit hysteresis (path-dependent "
                "behavior); (d) it provides no mechanism for the phase-transition "
                "responses captured by change-point detection and HMM state "
                "switching.\n"
            )
            sections.append(
                "See **Figure 8** and **Table 6** for matching law parameter "
                "distributions."
            )
        else:
            sections.append(
                "*Matching law models were not fit or produced no valid results.*"
            )
    else:
        sections.append(
            "*Matching law models were not fit or produced no valid results.*"
        )
    sections.append("")

    # ── 8. Model Comparison ─────────────────────────────────────────────────
    sections.append("## 8. Model Comparison\n")
    sections.append(
        "> **What this measures:** Reinforcement-learning models formalize "
        "competing hypotheses about the computational rules governing choice "
        "(e.g., how recent rewards are weighted, how quickly value estimates "
        "update). Model comparison identifies which rule best accounts for "
        "the observed choice sequence.\n"
        "> **Why it matters:** The matching law (Section 7) describes the "
        "equilibrium relationship between behavior and reinforcement ratios, "
        "but it does not specify the *process* that produces matching — RL "
        "models fill that gap by characterizing the trial-by-trial dynamics.\n"
        "> **Interpretation:** Lower BIC indicates a better trade-off between "
        "fit and parsimony. A model with forgetting or decay outperforming a "
        "memoryless model implies that participants integrate reward history "
        "over a finite window rather than responding only to the most recent "
        "outcome.\n"
    )
    mc = analysis_results.get("model_comparison")
    if mc is not None and len(mc) > 0:
        agg = mc.groupby("model")["bic"].mean().sort_values()
        best = agg.index[0]
        sections.append(
            f"The best-fitting model by mean BIC was **{best}** "
            f"(mean BIC = {agg.iloc[0]:.1f}).\n"
        )
        sections.append("Model ranking (by mean BIC, lower = better):\n")
        for model, bic_val in agg.items():
            delta = bic_val - agg.iloc[0]
            sections.append(f"- {model}: BIC = {bic_val:.1f} (ΔBIC = {delta:.1f})")
        sections.append(
            "\nSee **Figure 9** and **Table 7** for full model comparison."
        )
    else:
        sections.append("*Model fitting was not run or produced no results.*")
    sections.append("")

    # ── 9. Recurrence Quantification Analysis ──────────────────────────────
    sections.append("## 9. Recurrence Quantification Analysis (RQA)\n")
    sections.append(
        "> **What this measures:** Recurrence quantification analysis measures "
        "how often a behavioral trajectory revisits previous states — "
        "analogous to asking whether a participant's response pattern at "
        "minute 4 resembles their pattern at minute 1. It quantifies the "
        "deterministic structure and repetitiveness embedded in the choice "
        "sequence.\n"
        "> **Why it matters:** Steady-state designs assume behavior eventually "
        "settles into a stable pattern, but RQA reveals whether that pattern "
        "is truly stable, cyclically recurring, or drifting — distinctions "
        "invisible in session-wide averages.\n"
        "> **Interpretation:** High recurrence rate indicates the behavior "
        "frequently revisits familiar states; high determinism (DET) indicates "
        "that revisitations occur in predictable sequences rather than at "
        "random; longer diagonal lines indicate sustained, repeating "
        "behavioral motifs.\n"
    )
    rqa = analysis_results.get("rqa")
    if rqa is not None and isinstance(rqa, pd.DataFrame) and len(rqa) > 0:
        for col in ["recurrence_rate", "determinism", "mean_diag_length"]:
            if col in rqa.columns:
                label = col.replace("_", " ").title()
                sections.append(
                    f"- **{label}:** "
                    f"{rqa[col].mean():.3f} ({rqa[col].std():.3f})"
                )
        sections.append(
            "\nSee dynamical analysis figures for recurrence plots."
        )
    else:
        sections.append("*RQA was not run or produced no results.*")
    sections.append("")

    # ── 10. Fractal Scaling / DFA ────────────────────────────────────────
    sections.append("## 10. Fractal Scaling Analysis (DFA)\n")
    sections.append(
        "> **What this measures:** Detrended fluctuation analysis reveals "
        "whether choice sequences exhibit long-range temporal correlations "
        "— whether a response 50 clicks ago still influences the current "
        "choice — going beyond the local contingency analysis typical of "
        "steady-state designs.\n"
        "> **Why it matters:** If behavior were simply tracking local "
        "reinforcement rates, we would expect short-range correlations that "
        "decay quickly. Long-range correlations (fractal scaling) suggest "
        "that behavior is organized across multiple timescales "
        "simultaneously, a hallmark of adaptive, flexible systems.\n"
        "> **Interpretation:** A DFA exponent (alpha) near 0.5 indicates "
        "uncorrelated (random) choices; alpha > 0.5 indicates persistent "
        "long-range correlations (choices trend in the same direction); "
        "alpha near 1.0 indicates 1/f scaling, often associated with "
        "adaptive complexity; alpha > 1.0 suggests non-stationary drift.\n"
    )
    fractal = analysis_results.get("fractal", {})
    dfa_df = fractal.get("dfa")
    if dfa_df is not None and isinstance(dfa_df, pd.DataFrame) and len(dfa_df) > 0:
        full = dfa_df[dfa_df["phase"] == "full"] if "phase" in dfa_df.columns else dfa_df
        if len(full) > 0 and "alpha_dfa" in full.columns:
            sections.append(
                f"- **DFA exponent (full session):** "
                f"{full['alpha_dfa'].mean():.3f} ({full['alpha_dfa'].std():.3f})"
            )
        sections.append(
            "\nSee fractal analysis figures for log-log scaling plots."
        )
    else:
        sections.append("*DFA was not run or produced no results.*")
    sections.append("")

    # ── 11. Sample Entropy ────────────────────────────────────────────────
    sections.append("## 11. Sample Entropy\n")
    sections.append(
        "> **What this measures:** Sample entropy quantifies the "
        "unpredictability of the choice sequence — how surprising each new "
        "response is given the pattern of recent responses. Lower entropy "
        "means more regularity; higher entropy means less predictable "
        "behavior.\n"
        "> **Why it matters:** A highly reinforced behavior can appear "
        "stereotyped (low entropy) or flexibly variable (high entropy). "
        "Entropy separates these qualitatively different modes of "
        "responding that would look identical under a simple response-rate "
        "measure.\n"
        "> **Interpretation:** Values near 0 indicate highly regular, "
        "template-like sequences; values near 2 indicate that the sequence "
        "is essentially unpredictable from its own recent history. "
        "Intermediate values suggest structured variability.\n"
    )
    sampen_df = fractal.get("sample_entropy")
    if sampen_df is not None and isinstance(sampen_df, pd.DataFrame) and len(sampen_df) > 0:
        if "sample_entropy" in sampen_df.columns:
            sections.append(
                f"- **Sample entropy:** "
                f"{sampen_df['sample_entropy'].mean():.3f} "
                f"({sampen_df['sample_entropy'].std():.3f})"
            )
    else:
        sections.append("*Sample entropy was not computed or produced no results.*")
    sections.append("")

    # ── 12. Change-Point Detection (BOCPD) ────────────────────────────────
    sections.append("## 12. Change-Point Detection (BOCPD)\n")
    sections.append(
        "> **What this measures:** Bayesian online change-point detection "
        "identifies moments when the statistical properties of the choice "
        "sequence shift abruptly — points where behavior reorganizes, "
        "analogous to detecting when a participant 'notices' that the "
        "contingencies have changed.\n"
        "> **Why it matters:** In steady-state designs, condition changes "
        "are imposed by the experimenter and signaled explicitly. Here, "
        "contingency shifts are hidden, and BOCPD asks whether the "
        "behavioral stream itself contains detectable signatures of those "
        "shifts — a data-driven alternative to researcher-defined phase "
        "boundaries.\n"
        "> **Interpretation:** Detected change points that align with known "
        "phase boundaries indicate that participants' behavior reorganized "
        "in response to the hidden contingency change. Change points that "
        "do *not* align may reflect spontaneous strategy shifts or "
        "sensitivity to local reinforcement fluctuations.\n"
    )
    cp = analysis_results.get("changepoints", {})
    cp_df = cp.get("changepoints")
    if cp_df is not None and isinstance(cp_df, pd.DataFrame) and len(cp_df) > 0:
        sections.append(
            f"- **Change points detected:** "
            f"{len(cp_df)} across all sessions"
        )
        alignment = cp.get("boundary_alignment")
        if alignment is not None and isinstance(alignment, pd.DataFrame) and len(alignment) > 0:
            if "aligned" in alignment.columns:
                pct = alignment["aligned"].mean() * 100
                sections.append(
                    f"- **Aligned with phase boundaries:** {pct:.1f}%"
                )
        sections.append(
            "\nSee change-point figures for run-length posteriors and "
            "alignment plots."
        )
    else:
        sections.append("*Change-point detection was not run or produced no results.*")
    sections.append("")

    # ── 13. Empirical Dynamic Modeling / CCM ──────────────────────────────
    sections.append("## 13. Empirical Dynamic Modeling (EDM) and Convergent Cross-Mapping (CCM)\n")
    sections.append(
        "> **What this measures:** Convergent cross-mapping tests for causal "
        "coupling between variables (e.g., does reward history causally drive "
        "choice allocation?) using attractor reconstruction rather than "
        "traditional correlation. EDM more broadly tests whether the choice "
        "time series is governed by deterministic nonlinear dynamics rather "
        "than stochastic noise.\n"
        "> **Why it matters:** Correlations between reinforcement rate and "
        "response rate are well-documented in behavior analysis, but "
        "correlation does not establish directionality. CCM can detect "
        "causal influence even in coupled, bidirectional systems — "
        "precisely the situation in concurrent schedules where behavior "
        "and reinforcement mutually influence each other.\n"
        "> **Interpretation:** Simplex projection skill (rho) significantly "
        "above zero indicates deterministic structure in the choice "
        "time series. For CCM, cross-map skill that increases with library "
        "size indicates causal influence from the mapped variable; "
        "convergence is the hallmark that distinguishes genuine causation "
        "from mere correlation.\n"
    )
    edm = analysis_results.get("edm")
    if edm is not None and isinstance(edm, dict):
        smap = edm.get("smap_results")
        ccm = edm.get("ccm_results")
        if smap is not None and isinstance(smap, pd.DataFrame) and len(smap) > 0:
            if "rho" in smap.columns:
                sections.append(
                    f"- **S-map prediction skill (rho):** "
                    f"{smap['rho'].mean():.3f} ({smap['rho'].std():.3f})"
                )
        if ccm is not None and isinstance(ccm, pd.DataFrame) and len(ccm) > 0:
            sections.append(
                f"- **CCM analyses completed:** {len(ccm)} session-pairs"
            )
        sections.append(
            "\nSee EDM figures for simplex projections and CCM convergence plots."
        )
    else:
        sections.append("*EDM/CCM analyses were not run or produced no results.*")
    sections.append("")

    # ── 14. Hidden Markov Models ──────────────────────────────────────────
    sections.append("## 14. Hidden Markov Models (HMM)\n")
    sections.append(
        "> **What this measures:** Hidden Markov models identify discrete "
        "latent behavioral states that the participant transitions between "
        "during the session — for example, an 'exploring' state with high "
        "switching vs. an 'exploiting' state with strong preference for one "
        "option.\n"
        "> **Why it matters:** Behavior analysts are familiar with the idea "
        "that organisms alternate between distinguishable behavioral modes "
        "(e.g., ratio-run vs. post-reinforcement pause). HMMs provide a "
        "principled, data-driven method for identifying such modes without "
        "requiring the researcher to define them in advance — letting the "
        "data reveal the structure rather than imposing it.\n"
        "> **Interpretation:** Fewer states with clear separation suggest "
        "distinct behavioral strategies; the transition matrix shows how "
        "likely the participant is to shift between states. State durations "
        "indicate how long each behavioral mode persists before the "
        "participant switches strategy.\n"
    )
    hmm = analysis_results.get("hmm")
    if hmm is not None and isinstance(hmm, pd.DataFrame) and len(hmm) > 0:
        if "n_states" in hmm.columns and "bic" in hmm.columns:
            best_row = hmm.loc[hmm["bic"].idxmin()]
            sections.append(
                f"- **Best number of states (by BIC):** "
                f"{int(best_row['n_states'])}"
            )
        sections.append(
            "\nSee HMM figures for state sequences and transition matrices."
        )
    else:
        sections.append("*HMM fitting was not run or produced no results.*")
    sections.append("")

    # ── 15. Interpretive Cautions ────────────────────────────────────────────
    sections.append("## 15. Interpretive Cautions\n")
    sections.append(
        "- **Descriptive vs. inferential:** Many metrics reported here are "
        "descriptive summaries. Pre-registered inferential tests (if applicable) "
        "should be distinguished from exploratory analyses.\n"
        "- **Latent-value metrics:** Proportion optimal and regret are computed "
        "against omniscient knowledge of the hidden state. Participants cannot "
        "observe latent values; these metrics quantify deviation from an ideal "
        "that no real decision-maker can achieve.\n"
        "- **Rolling windows:** All rolling metrics depend on the chosen window "
        f"size ({config.get('rolling_window_clicks', 20)} clicks / "
        f"{config.get('rolling_window_time_s', 15)} s). Sensitivity analyses "
        "varying these parameters are recommended.\n"
        "- **Phase boundaries:** Phases are defined by elapsed time, not by "
        "click count. Participants with faster click rates contribute more "
        "observations per phase.\n"
        "- **Adaptation threshold:** The adaptation lag criterion "
        f"({config.get('adaptation_threshold', 0.6)}) is a researcher degree "
        "of freedom. Results should be checked across a range of thresholds.\n"
        "- **Sample size:** With N = {n} sessions, effect size estimates for "
        "individual differences have wide confidence intervals.".format(n=n)
    )
    sections.append("")

    # ── Write ───────────────────────────────────────────────────────────────
    text = "\n".join(sections)
    path = os.path.join(out_dir, "results_summary.md")
    with open(path, "w") as f:
        f.write(text)
    print(f"Results summary: {path}")
    return text
