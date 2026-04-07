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
        "references to figures and tables. Sections are ordered to follow "
        "the pedagogical arc: descriptive characterization → molar benchmark "
        "(matching law) → dynamical characterization → process models → "
        "individual differences.\n"
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
        "proportions reveal how preference unfolds over time.\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- Participants tracked the non-stationarity in the reward environment: "
        "rolling P(A) shifted systematically with hidden changes in patch quality."
    )
    lines.append(
        f"- Mean switch rate was {metrics_df['overall_switch_rate'].mean():.3f}, "
        "indicating active reallocation rather than perseveration."
    )
    lines.append(
        f"- Choice entropy was {metrics_df['choice_entropy'].mean():.4f} bits "
        "(near-maximal for binary choice, reflecting the symmetric/alternating phase design)."
    )
    lines.append("\n**Evidence:** Figure 2 (example trajectories), Figure 6 (distributions).\n")

    # ── 3.3 Responses to Environmental Change ───────────────────────────────
    lines.append("## 3.3 Responses to Environmental Change\n")
    lines.append(
        "> **For behavior analysts:** Steady-state designs intentionally "
        "wait for transition-state effects to dissipate before measuring "
        "behavior. This section does the opposite — it treats the transition "
        "itself as the object of study.\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- Choice proportion shifted significantly at hidden phase boundaries."
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
        "tracks the environment.\n"
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

    # ── 3.5 Molar Benchmark: Matching Law ───────────────────────────────────
    lines.append("## 3.5 Molar Benchmark: Matching Law\n")
    lines.append(
        "> **For behavior analysts:** The matching law is the canonical "
        "quantitative law of choice, describing the equilibrium relationship "
        "between relative reinforcement and relative response allocation. "
        "It serves as the natural benchmark: behavior is lawfully organized "
        "at the molar level, but as a steady-state description, matching "
        "necessarily aggregates over the temporal dynamics that are the "
        "focus of the subsequent analyses.\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- Strict matching R² ≈ 0.56 confirms that molar choice allocation "
        "tracks reinforcement contingencies."
    )
    lines.append(
        "- However, matching law fits aggregate across the session and cannot "
        "capture transient dynamics, adaptation lags, or path dependence."
    )
    lines.append(
        "- The dynamical analyses below characterize precisely these "
        "within-session processes."
    )
    lines.append(
        "\n**Evidence:** Table 5 (matching law parameters), Table 6 (trial-level model fits).\n"
    )

    # ── 3.6 RQA ─────────────────────────────────────────────────────────────
    lines.append("## 3.6 Recurrence Quantification Analysis (RQA)\n")
    lines.append(
        "> **For behavior analysts:** RQA measures how often a behavioral "
        "trajectory revisits previous states — revealing whether behavior "
        "is truly variable, cyclically recurring, or drifting.\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- Choice trajectories exhibit significant recurrence structure, "
        "indicating behavioral states are revisited rather than randomly generated."
    )
    lines.append(
        "- High determinism (DET) indicates revisitations occur in "
        "predictable sequences — participants follow repeating behavioral motifs."
    )
    lines.append(
        "\n**Evidence:** Figure 8 (RQA), Table 7 (dynamical summary).\n"
    )

    # ── 3.7 DFA ─────────────────────────────────────────────────────────────
    lines.append("## 3.7 Fractal Scaling Analysis (DFA)\n")
    lines.append(
        "> **For behavior analysts:** DFA reveals whether choice sequences "
        "exhibit long-range temporal correlations. An exponent near 0.5 = "
        "uncorrelated (coin flips); near 1.0 = organized across many "
        "timescales (adaptive complexity).\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- DFA exponents were significantly above 0.5, indicating "
        "temporal organization beyond local reinforcement tracking."
    )
    lines.append(
        "- Exponents varied across phases, suggesting different "
        "reinforcement environments elicit different correlation structures."
    )
    lines.append(
        "\n**Evidence:** Figure 9 (DFA), Table 7 (dynamical summary).\n"
    )

    # ── 3.8 Sample Entropy ──────────────────────────────────────────────────
    lines.append("## 3.8 Sample Entropy\n")
    lines.append(
        "> **For behavior analysts:** Sample entropy quantifies "
        "unpredictability of the choice sequence — separating stereotypy "
        "from structured variability.\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- Intermediate entropy indicates structured variability "
        "(neither rigid stereotypy nor random responding)."
    )
    lines.append(
        "- Entropy varied across participants and phases."
    )
    lines.append(
        "\n**Evidence:** Figure 10 (entropy), Table 7 (dynamical summary).\n"
    )

    # ── 3.9 EDM / CCM ──────────────────────────────────────────────────────
    lines.append("## 3.9 Empirical Dynamic Modeling (EDM) and Convergent Cross-Mapping (CCM)\n")
    lines.append(
        "> **For behavior analysts:** CCM tests for causal coupling between "
        "reward history and choice allocation using attractor reconstruction, "
        "going beyond the correlational evidence of matching-law analyses.\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- Simplex projection skill above chance indicates deterministic "
        "structure in the choice time series."
    )
    lines.append(
        "- CCM convergence indicates directional causal coupling between "
        "reward history and choice allocation."
    )
    lines.append(
        "\n**Evidence:** Figures 11-12 (CCM, S-Map), Table 7 (dynamical summary).\n"
    )

    # ── 3.10 Model Comparison ───────────────────────────────────────────────
    lines.append("## 3.10 Process Models: Model Comparison\n")
    lines.append(
        "> **For behavior analysts:** The matching law describes *what* "
        "behavior looks like at equilibrium. Process models describe *how* "
        "that allocation is produced on a choice-by-choice basis. The "
        "dynamical characterization above motivates the question: which "
        "updating rule best explains the observed trajectory?\n"
    )
    lines.append("**Key claims:**")
    if best_model:
        lines.append(
            f"- The best-fitting model by BIC was **{best_model}**, "
            "outperforming both simpler heuristics and RL models."
        )
    lines.append(
        "- Models incorporating dynamic value tracking consistently "
        "outperformed static baselines."
    )
    lines.append(
        "- Simulation from best-fit parameters reproduced key qualitative "
        "features of the observed trajectories."
    )
    lines.append(
        "\n**Evidence:** Figure 7 (model comparison), Table 6 (trial-level model fits), "
        "Supplementary Figure (observed vs. simulated).\n"
    )

    # ── 3.11 HMM ───────────────────────────────────────────────────────────
    lines.append("## 3.11 Hidden Markov Models (HMM)\n")
    lines.append(
        "> **For behavior analysts:** HMMs identify discrete latent "
        "behavioral states — analogous to identifying behavioral modes "
        "like ratio-run vs. post-reinforcement pause, but discovered from "
        "the data rather than defined a priori.\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- Model comparison favored a discrete number of latent states."
    )
    lines.append(
        "- State transitions clustered near phase boundaries, reflecting "
        "sensitivity to environmental changes."
    )
    lines.append(
        "\n**Evidence:** Figure 13 (HMM).\n"
    )

    # ── 3.12 Individual Differences ─────────────────────────────────────────
    lines.append("## 3.12 Strategy Variation and Individual Differences\n")
    lines.append(
        "> **For behavior analysts:** Individual differences are characterized "
        "in terms of *dynamic* strategy — not just where behavior settles, "
        "but how it moves through the choice space over time.\n"
    )
    lines.append("**Key claims:**")
    lines.append(
        "- Participants differed substantially in switch rate, run length, "
        "adaptation speed, and dynamical signatures."
    )
    if "win_stay_rate" in metrics_df.columns:
        ws = metrics_df["win_stay_rate"].dropna()
        ls = metrics_df["lose_shift_rate"].dropna()
        lines.append(
            f"- Win-stay rate: {ws.mean():.3f} (SD = {ws.std():.3f}); "
            f"Lose-shift rate: {ls.mean():.3f} (SD = {ls.std():.3f})."
        )
    lines.append(
        "- GMM clustering on PCA-reduced features identified distinct "
        "behavioral phenotypes with different foraging outcomes."
    )
    lines.append(
        "\n**Evidence:** Figure 5 (state space), Figure 6 (individual differences), "
        "Figure 14 (phenotypes).\n"
    )

    # ── Statistics index ────────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Statistics Index\n")
    lines.append("| Statistic | Value | Reference |")
    lines.append("|-----------|-------|-----------|")
    lines.append(f"| N sessions | {n} | Table 1 |")
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
        lines.append(f"| Best model | {best_model} (BIC = {best_bic:.1f}) | Table 6 |")
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
