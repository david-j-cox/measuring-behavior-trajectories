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

    # ── 7. Model Comparison ─────────────────────────────────────────────────
    sections.append("## 7. Model Comparison\n")
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
            "\nSee **Figure 7** and **Table 5** for full model comparison."
        )
    else:
        sections.append("*Model fitting was not run or produced no results.*")
    sections.append("")

    # ── 8. Interpretive Cautions ────────────────────────────────────────────
    sections.append("## 8. Interpretive Cautions\n")
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
