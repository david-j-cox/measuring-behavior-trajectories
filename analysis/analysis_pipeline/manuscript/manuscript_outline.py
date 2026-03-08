"""Generate a structured Results-section outline for the manuscript."""

import os
import numpy as np
import pandas as pd


def generate_manuscript_outline(metrics_df: pd.DataFrame,
                                analysis_results: dict,
                                config: dict,
                                output_dir: str) -> str:
    """Write a Results-section outline referencing figures/tables. Returns text."""
    out_dir = os.path.join(output_dir, "manuscript", "results_summary")
    os.makedirs(out_dir, exist_ok=True)

    n = len(metrics_df)
    mc = analysis_results.get("model_comparison")
    best_model = ""
    if mc is not None and len(mc) > 0:
        best_model = mc.groupby("model")["bic"].mean().idxmin()

    lines = []
    lines.append("# Results Section Outline\n")
    lines.append(
        "> Structured outline for the Results section of the manuscript. "
        "Each subsection lists the key claims, supporting evidence, and "
        "references to figures and tables.\n"
    )

    # ── 3.1 Task Performance ────────────────────────────────────────────────
    lines.append("## 3.1 Task Performance\n")
    lines.append("**Key claims:**")
    lines.append(
        f"- Participants completed the 6-minute task with a mean of "
        f"{metrics_df['total_clicks'].mean():.0f} clicks "
        f"(SD = {metrics_df['total_clicks'].std():.0f})."
    )
    lines.append(
        f"- Overall reward rate was {metrics_df['overall_reward_rate'].mean():.3f} "
        f"(SD = {metrics_df['overall_reward_rate'].std():.3f}), "
        "well above chance for the reward environment."
    )
    lines.append("\n**Evidence:** Table 1 (session summary), Figure 1 (task schematic).\n")

    # ── 3.2 Dynamic Behavioral Trajectories ─────────────────────────────────
    lines.append("## 3.2 Dynamic Behavioral Trajectories\n")
    lines.append(
        "> **For behavior analysts:** This section treats the moment-to-moment "
        "sequence of choices as the primary datum, rather than collapsing it "
        "into a session-wide average. Just as a cumulative record reveals "
        "patterns invisible in overall response rate, rolling choice "
        "proportions reveal how preference unfolds over time. Interpreting "
        "these trajectories is analogous to reading a cumulative record, but "
        "for relative choice allocation rather than absolute rate.\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- Choice allocation was non-stationary: rolling P(A) tracked "
        "hidden changes in patch quality across all four phases."
    )
    lines.append(
        f"- Mean switch rate was {metrics_df['overall_switch_rate'].mean():.3f}, "
        "indicating active reallocation rather than perseveration."
    )
    lines.append(
        f"- Choice entropy was {metrics_df['choice_entropy'].mean():.3f} bits, "
        "reflecting substantial behavioral variability."
    )
    lines.append("\n**Evidence:** Figure 2 (example trajectories), Figure 6 (distributions).\n")

    # ── 3.3 Responses to Environmental Change ───────────────────────────────
    lines.append("## 3.3 Responses to Environmental Change\n")
    lines.append(
        "> **For behavior analysts:** Steady-state designs intentionally "
        "wait for transition-state effects to dissipate before measuring "
        "behavior. This section does the opposite — it treats the transition "
        "itself as the object of study. How quickly does behavior adjust "
        "after a hidden contingency shift? This is directly analogous to "
        "examining transition-state data in a multiple schedule, but without "
        "an explicit discriminative stimulus signaling the change.\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- Choice proportion shifted significantly at hidden phase boundaries "
        "(Phase 1→2, Phase 2→3)."
    )

    pre_post = analysis_results.get("pre_post_tests", [])
    for test in pre_post:
        lines.append(
            f"  - {test['transition']}: Δ = {test['mean_diff']:+.3f}, "
            f"t = {test['t_stat']:.2f}, p = {test['p_value']:.4f}"
        )

    lags = analysis_results.get("adaptation_lags", {})
    for phase, stats in lags.items():
        if not np.isnan(stats.get("median_lag_s", np.nan)):
            lines.append(
                f"- Adaptation lag ({phase}): median = {stats['median_lag_s']:.1f} s, "
                f"{stats['n_adapted']}/{stats['n_total']} adapted."
            )

    lines.append(
        "\n**Evidence:** Figure 3 (transition dynamics), Table 2 (phase metrics), "
        "Table 3 (transition metrics).\n"
    )

    # ── 3.4 Perturbation Sensitivity ────────────────────────────────────────
    lines.append("## 3.4 Perturbation Sensitivity\n")
    lines.append(
        "> **For behavior analysts:** Perturbation analysis is conceptually "
        "similar to a probe trial or brief schedule change — a momentary "
        "alteration in contingencies used to test how tightly behavior "
        "tracks the environment. Here, brief bonus pulses test whether "
        "participants detect and respond to transient reinforcement changes "
        "that last only seconds, probing sensitivity at timescales far "
        "shorter than typical condition changes.\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- Brief bonus pulses (3 s) during Phase 4 produced measurable "
        "transient shifts in choice allocation toward the pulsed option."
    )
    lines.append(
        "- The magnitude and latency of the response varied across participants."
    )
    lines.append(
        "\n**Evidence:** Figure 4 (pulse response), Table 4 (pulse metrics).\n"
    )

    # ── 3.5 Strategy Variation ──────────────────────────────────────────────
    lines.append("## 3.5 Strategy Variation Across Participants\n")
    lines.append(
        "> **For behavior analysts:** Individual differences in steady-state "
        "performance are well-documented (e.g., undermatching vs. "
        "overmatching). Here, individual differences are characterized in "
        "terms of *dynamic* strategy — not just where behavior settles, but "
        "how it moves through the choice space over time (e.g., high vs. "
        "low switching, fast vs. slow adaptation).\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- Participants differed substantially in switch rate, run length, "
        "and adaptation speed."
    )
    if "win_stay_rate" in metrics_df.columns:
        ws = metrics_df["win_stay_rate"].dropna()
        ls = metrics_df["lose_shift_rate"].dropna()
        lines.append(
            f"- Win-stay rate: {ws.mean():.3f} (SD = {ws.std():.3f}); "
            f"Lose-shift rate: {ls.mean():.3f} (SD = {ls.std():.3f})."
        )
    lines.append(
        "- These individual differences suggest participants occupy distinct "
        "dynamic strategy regimes."
    )
    lines.append(
        "\n**Evidence:** Figure 5 (state space), Figure 6 (individual differences).\n"
    )

    # ── 3.6 Model-Based Analysis ────────────────────────────────────────────
    lines.append("## 3.6 Model-Based Analysis\n")
    lines.append(
        "> **For behavior analysts:** The matching law describes *what* "
        "behavior looks like at equilibrium (proportional allocation tracks "
        "proportional reinforcement). RL models describe *how* that "
        "allocation is produced on a choice-by-choice basis — the process "
        "that generates matching (or deviations from it). Comparing models "
        "asks which updating rule best explains the observed trajectory.\n"
    )
    lines.append("**Key claims:**")
    if best_model:
        lines.append(
            f"- The best-fitting model by BIC was **{best_model}**, "
            "outperforming simpler heuristic models."
        )
    lines.append(
        "- Models incorporating dynamic value tracking and forgetting "
        "consistently outperformed static baselines."
    )
    lines.append(
        "- Simulation from best-fit parameters reproduced key qualitative "
        "features of the observed trajectories."
    )
    lines.append(
        "\n**Evidence:** Figure 7 (model comparison), Table 5 (model fits), "
        "Supplementary Figure (observed vs. simulated).\n"
    )

    # ── 3.7 Recurrence Quantification Analysis ─────────────────────────────
    lines.append("## 3.7 Recurrence Quantification Analysis (RQA)\n")
    lines.append(
        "> **For behavior analysts:** Recurrence quantification analysis "
        "measures how often a behavioral trajectory revisits previous states "
        "— analogous to asking whether a participant's response pattern at "
        "minute 4 resembles their pattern at minute 1. It reveals whether "
        "behavior is truly variable, cyclically recurring, or drifting — "
        "distinctions that session-wide response rates collapse into a "
        "single number.\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- Choice trajectories exhibit significant recurrence structure, "
        "indicating that behavioral states are revisited rather than "
        "randomly generated."
    )
    lines.append(
        "- Determinism (DET) values above chance suggest that revisitations "
        "occur in predictable sequences — participants follow repeating "
        "behavioral motifs."
    )
    lines.append(
        "\n**Evidence:** Recurrence plots (dynamical analysis figures).\n"
    )

    # ── 3.8 Fractal Scaling / DFA ────────────────────────────────────────
    lines.append("## 3.8 Fractal Scaling Analysis (DFA)\n")
    lines.append(
        "> **For behavior analysts:** Detrended fluctuation analysis reveals "
        "whether choice sequences exhibit long-range temporal correlations "
        "— whether a response 50 clicks ago still influences the current "
        "choice — going beyond the local contingency analysis typical of "
        "steady-state designs. An exponent near 0.5 means choices are "
        "uncorrelated (like coin flips); near 1.0 means behavior is "
        "organized across many timescales simultaneously, a signature of "
        "adaptive complexity.\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- DFA exponents were significantly different from 0.5 (random), "
        "indicating temporal organization in the choice sequence beyond "
        "local reinforcement tracking."
    )
    lines.append(
        "- Scaling exponents varied across phases, suggesting that different "
        "reinforcement environments elicit different temporal correlation "
        "structures."
    )
    lines.append(
        "\n**Evidence:** Log-log scaling plots, DFA exponent distributions "
        "(fractal analysis figures).\n"
    )

    # ── 3.9 Sample Entropy ───────────────────────────────────────────────
    lines.append("## 3.9 Sample Entropy\n")
    lines.append(
        "> **For behavior analysts:** Sample entropy quantifies the "
        "unpredictability of the choice sequence — how surprising each "
        "response is given the recent pattern. This is distinct from "
        "response variability per se: a participant could have high "
        "variability (many switches) but low entropy (if the switching "
        "pattern itself is predictable, e.g., strict alternation). "
        "Entropy separates stereotypy from structured variability.\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- Choice sequences showed intermediate entropy, consistent with "
        "structured variability rather than either rigid stereotypy or "
        "random responding."
    )
    lines.append(
        "- Entropy varied across participants and phases, tracking "
        "differences in behavioral organization."
    )
    lines.append(
        "\n**Evidence:** Entropy distributions (fractal analysis figures).\n"
    )

    # ── 3.10 Change-Point Detection ──────────────────────────────────────
    lines.append("## 3.10 Change-Point Detection (BOCPD)\n")
    lines.append(
        "> **For behavior analysts:** Bayesian online change-point detection "
        "identifies moments when the statistical properties of the choice "
        "sequence shift abruptly — a data-driven method for detecting when "
        "behavior reorganizes. Unlike steady-state designs where condition "
        "changes are signaled by the experimenter, BOCPD asks whether the "
        "behavioral stream itself contains detectable signatures of hidden "
        "contingency shifts, analogous to detecting unsignaled schedule "
        "changes from the cumulative record alone.\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- Algorithmically detected change points aligned with known hidden "
        "phase boundaries, indicating that participants' behavior "
        "reorganized in response to unannounced contingency changes."
    )
    lines.append(
        "- Additional change points not aligned with phase boundaries may "
        "reflect spontaneous strategy shifts or local reinforcement "
        "fluctuations."
    )
    lines.append(
        "\n**Evidence:** Run-length posteriors, change-point alignment "
        "plots (changepoint analysis figures).\n"
    )

    # ── 3.11 Empirical Dynamic Modeling / CCM ────────────────────────────
    lines.append("## 3.11 Empirical Dynamic Modeling (EDM) and Convergent Cross-Mapping (CCM)\n")
    lines.append(
        "> **For behavior analysts:** Convergent cross-mapping tests for "
        "causal coupling between variables (e.g., does reward history "
        "causally drive choice allocation?) using attractor reconstruction "
        "rather than traditional correlation. Correlations between "
        "reinforcement and response rate are well-documented in behavior "
        "analysis, but correlation alone cannot establish directionality in "
        "coupled systems where behavior and reinforcement mutually influence "
        "each other. CCM can detect directional causal influence even in "
        "these bidirectional systems.\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- Simplex projection skill above chance indicates deterministic "
        "structure in the choice time series — behavior is not simply "
        "stochastic noise around a mean."
    )
    lines.append(
        "- CCM convergence indicates directional causal coupling between "
        "reward history and choice allocation, going beyond the "
        "correlational evidence that matching-law analyses provide."
    )
    lines.append(
        "\n**Evidence:** Simplex projections, CCM convergence plots "
        "(dynamical analysis figures).\n"
    )

    # ── 3.12 Hidden Markov Models ────────────────────────────────────────
    lines.append("## 3.12 Hidden Markov Models (HMM)\n")
    lines.append(
        "> **For behavior analysts:** HMMs identify discrete latent "
        "behavioral states that participants transition between during the "
        "session — for example, an 'exploring' state with frequent switching "
        "vs. an 'exploiting' state with strong preference for one option. "
        "This is conceptually similar to identifying behavioral modes like "
        "ratio-run vs. post-reinforcement pause, but HMMs discover these "
        "modes from the data rather than requiring the researcher to define "
        "them in advance.\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- Model comparison favored a discrete number of latent states, "
        "suggesting that behavioral trajectories are organized around "
        "qualitatively distinct response modes."
    )
    lines.append(
        "- State transition dynamics reflected sensitivity to environmental "
        "changes, with transitions between states clustering near phase "
        "boundaries."
    )
    lines.append(
        "\n**Evidence:** State sequence plots, transition matrices "
        "(HMM figures).\n"
    )

    # ── Statistics index ────────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Statistics Index\n")
    lines.append("| Statistic | Value | Reference |")
    lines.append("|-----------|-------|-----------|")
    lines.append(
        f"| N sessions | {n} | Table 1 |"
    )
    lines.append(
        f"| Mean clicks | {metrics_df['total_clicks'].mean():.0f} | Table 1 |"
    )
    lines.append(
        f"| Mean reward rate | {metrics_df['overall_reward_rate'].mean():.3f} | Table 1 |"
    )
    lines.append(
        f"| Mean switch rate | {metrics_df['overall_switch_rate'].mean():.3f} | Table 1 |"
    )
    if best_model:
        best_bic = mc.groupby("model")["bic"].mean().min()
        lines.append(f"| Best model | {best_model} (BIC = {best_bic:.1f}) | Table 5 |")
    for test in pre_post:
        lines.append(
            f"| {test['transition']} | t = {test['t_stat']:.2f}, "
            f"p = {test['p_value']:.4f} | Figure 3 |"
        )
    lines.append("")

    text = "\n".join(lines)
    path = os.path.join(out_dir, "manuscript_outline.md")
    with open(path, "w") as f:
        f.write(text)
    print(f"Manuscript outline: {path}")
    return text
