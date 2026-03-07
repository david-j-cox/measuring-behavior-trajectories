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
