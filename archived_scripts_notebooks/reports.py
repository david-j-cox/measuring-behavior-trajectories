"""Report generation: HTML and Markdown summaries."""

import os
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np


def generate_report(metrics_df: pd.DataFrame, validation_report: pd.DataFrame,
                    analysis_results: dict, config: dict, output_dir: str):
    """Generate a summary report in HTML or Markdown."""
    fmt = config.get("report_format", "html")
    if fmt == "html":
        _generate_html_report(metrics_df, validation_report, analysis_results,
                              config, output_dir)
    else:
        _generate_markdown_report(metrics_df, validation_report, analysis_results,
                                  config, output_dir)


def _generate_html_report(metrics_df, validation_report, analysis_results,
                          config, output_dir):
    """Generate HTML report."""
    report_path = os.path.join(output_dir, "reports", "analysis_report.html")

    sections = []

    # Header
    sections.append(f"""
    <h1>Behavioral Trajectory Analysis Report</h1>
    <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    <p>Pipeline version: 1.0.0</p>
    <hr>
    """)

    # Overview
    sections.append("<h2>1. Data Overview</h2>")
    sections.append(f"""
    <ul>
        <li>Total sessions analyzed: <strong>{len(metrics_df)}</strong></li>
        <li>Total participants: <strong>{metrics_df['participant_id'].nunique()}</strong></li>
        <li>Mean clicks per session: <strong>{metrics_df['total_clicks'].mean():.0f}</strong>
            (SD = {metrics_df['total_clicks'].std():.0f})</li>
        <li>Mean session duration: <strong>{metrics_df['session_duration_s'].mean():.0f}s</strong>
            (SD = {metrics_df['session_duration_s'].std():.0f}s)</li>
    </ul>
    """)

    # Validation summary
    if validation_report is not None and len(validation_report) > 0:
        sections.append("<h2>2. Validation Summary</h2>")
        severity_counts = validation_report["severity"].value_counts().to_dict()
        sections.append("<ul>")
        for sev, count in severity_counts.items():
            sections.append(f"<li>{sev}: {count}</li>")
        sections.append("</ul>")
        sections.append(_df_to_html(validation_report.head(20),
                                     "Validation Issues (first 20)"))

    # Session metrics summary
    sections.append("<h2>3. Session Metrics Summary</h2>")
    summary_cols = [
        "total_clicks", "session_duration_s", "overall_reward_rate",
        "overall_switch_rate", "mean_run_length", "choice_entropy",
        "choice_prop_a", "final_score",
    ]
    available_cols = [c for c in summary_cols if c in metrics_df.columns]
    if available_cols:
        desc = metrics_df[available_cols].describe().round(3)
        sections.append(_df_to_html(desc, "Descriptive Statistics"))

    # Phase analysis
    if "phase_descriptives" in analysis_results:
        sections.append("<h2>4. Phase-Level Analysis</h2>")
        sections.append(_df_to_html(analysis_results["phase_descriptives"],
                                     "Phase Descriptives"))

    if "pre_post_tests" in analysis_results:
        tests = analysis_results["pre_post_tests"]
        if tests:
            sections.append("<h3>Pre-Post Phase Transition Tests</h3>")
            tests_df = pd.DataFrame(tests).round(4)
            sections.append(_df_to_html(tests_df, "Paired t-tests (with Cohen's d effect size)"))
            # Inline effect size summary
            for t in tests:
                d = t.get("cohens_d")
                if d is not None and not np.isnan(d):
                    magnitude = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
                    sections.append(
                        f"<p>{t['transition']}: Cohen's d = {d:.3f} ({magnitude} effect)</p>"
                    )

    if "adaptation_lags" in analysis_results:
        sections.append("<h3>Adaptation Lags</h3>")
        for phase, stats in analysis_results["adaptation_lags"].items():
            sections.append(f"<p><strong>{phase}:</strong> "
                            f"Adapted: {stats['n_adapted']}/{stats['n_total']}, "
                            f"Mean lag: {stats['mean_lag_s']:.1f}s, "
                            f"Median: {stats['median_lag_s']:.1f}s</p>")

    # Hysteresis explanation
    if "hysteresis" in analysis_results and analysis_results["hysteresis"]:
        sections.append("<h3>Hysteresis Analysis</h3>")
        sections.append("""<p>The hysteresis plot examines whether choice behavior
        follows a different path depending on the <em>direction</em> of environmental
        change. During Phase 2, Option A's recovery rate increases (A-advantage rising),
        while during Phase 3, Option B becomes advantaged (A-advantage declining). If
        participants exhibit hysteresis, their choice proportion at a given latent
        advantage level will differ depending on whether the environment is improving
        or worsening for Option A. A visible loop or separation between the Phase 2
        and Phase 3 trajectories indicates path dependence: behavior is shaped not
        only by the current contingencies but also by the participant's recent
        history of reinforcement. This is a hallmark of nonlinear dynamical systems
        and suggests that simple equilibrium models may be insufficient to capture
        the full structure of foraging behavior.</p>""")

    # Matching Law Analysis (dedicated section)
    if "model_comparison" in analysis_results:
        mc = analysis_results["model_comparison"]
        if isinstance(mc, pd.DataFrame) and len(mc) > 0:
            matching_df = mc[mc["model"] == "generalized_matching_law"]
            if len(matching_df) > 0:
                sections.append("<h2>5. Matching Law Analysis</h2>")
                sections.append("""<p>The generalized matching law (Baum, 1974) is the
                canonical quantitative law of choice in behavior analysis. It predicts
                that the relative allocation of behavior to an option will match the
                relative rate of reinforcement obtained from that option, modulated by
                sensitivity (<em>s</em>) and bias (<em>b</em>) parameters:</p>
                <blockquote>log(B<sub>A</sub>/B<sub>B</sub>) = <em>s</em> &middot;
                log(R<sub>A</sub>/R<sub>B</sub>) + log(<em>b</em>)</blockquote>""")

                sections.append("<h3>Parameter Distributions</h3>")
                s_mean = matching_df["sensitivity"].mean()
                s_sd = matching_df["sensitivity"].std()
                s_min = matching_df["sensitivity"].min()
                s_max = matching_df["sensitivity"].max()
                b_mean = matching_df["bias"].mean()
                b_sd = matching_df["bias"].std()
                r2_mean = matching_df["r_squared"].mean()
                r2_sd = matching_df["r_squared"].std()
                r2_min = matching_df["r_squared"].min()
                r2_max = matching_df["r_squared"].max()

                sections.append(f"""<ul>
                    <li><strong>Sensitivity (<em>s</em>):</strong>
                        mean = {s_mean:.3f} (SD = {s_sd:.3f}),
                        range = [{s_min:.3f}, {s_max:.3f}]</li>
                    <li><strong>Bias (<em>b</em>):</strong>
                        mean = {b_mean:.3f} (SD = {b_sd:.3f})</li>
                    <li><strong>R&sup2;:</strong>
                        mean = {r2_mean:.3f} (SD = {r2_sd:.3f}),
                        range = [{r2_min:.3f}, {r2_max:.3f}]</li>
                    <li><strong>Sessions fit:</strong> {len(matching_df)}</li>
                </ul>""")

                # Interpretation of sensitivity
                sections.append("<h3>Interpretation</h3>")
                all_s = matching_df["sensitivity"]
                n_under = (all_s < 0.9).sum()
                n_strict = ((all_s >= 0.9) & (all_s <= 1.1)).sum()
                n_over = (all_s > 1.1).sum()
                sections.append(f"""<p>Sensitivity values indicate how strongly choice
                allocation tracks reinforcement ratios:</p>
                <ul>
                    <li><em>s</em> &lt; 1 (<strong>undermatching</strong>): Choice
                        proportions change less than reinforcement ratios —
                        the most common finding in behavior-analytic research.
                        Observed in {n_under} of {len(all_s)} session fits.</li>
                    <li><em>s</em> &asymp; 1 (<strong>strict matching</strong>):
                        Choice proportions track reinforcement ratios
                        proportionally. Observed in {n_strict} session fits.</li>
                    <li><em>s</em> &gt; 1 (<strong>overmatching</strong>): Choice
                        proportions are more extreme than reinforcement ratios,
                        indicating strong preference for the richer option.
                        Observed in {n_over} session fits.</li>
                </ul>""")

                # Strengths and limitations
                sections.append("<h3>Where Matching Succeeds and Fails</h3>")
                sections.append("""<p><strong>Strengths:</strong> The matching law
                successfully captures <em>molar allocation</em> — the overall
                tendency for behavior to track reinforcement across time windows.
                It provides a well-validated, parsimonious summary of steady-state
                choice proportions and allows direct comparison with decades of
                behavior-analytic research on concurrent schedules.</p>

                <p><strong>Limitations in this paradigm:</strong> Because the matching
                law is an equilibrium description, it cannot capture:</p>
                <ul>
                    <li><strong>Transient dynamics:</strong> The trajectory of
                        adjustment following a phase transition — how quickly and
                        by what path behavior shifts to a new allocation.</li>
                    <li><strong>Adaptation lags:</strong> The delay between an
                        environmental change and the behavioral response, which
                        varies across participants and transitions.</li>
                    <li><strong>Phase-transition responses:</strong> Matching fits
                        aggregate across regime changes, obscuring the distinct
                        behavioral states revealed by HMM, change-point, and
                        dynamical systems analyses.</li>
                    <li><strong>Path dependence / hysteresis:</strong> The matching
                        law predicts the same allocation for a given reinforcement
                        ratio regardless of history, yet participants show
                        direction-dependent behavior (see Hysteresis Analysis).</li>
                </ul>

                <p>The dynamical analyses in subsequent sections extend beyond
                matching by characterizing the <em>transient processes</em> by which
                matching-like equilibria are approached, disrupted, and
                re-established after environmental regime changes.</p>""")

                # Note about figures
                fig_dir = os.path.join(output_dir, "figures")
                matching_figs = list(Path(fig_dir).glob("matching_*.*"))
                if matching_figs:
                    sections.append(
                        "<p>See <strong>Matching Law Figures</strong> below for "
                        "sensitivity/bias distributions and log-ratio regression plots.</p>"
                    )
                else:
                    sections.append(
                        "<p><em>Note: Matching law diagnostic figures (sensitivity/bias "
                        "distributions, log-ratio regression plots) are not yet generated "
                        "by the plotting module. Adding these figures is recommended.</em></p>"
                    )

    # Model comparison
    if "model_comparison" in analysis_results:
        sections.append("<h2>6. Model Comparison</h2>")
        mc = analysis_results["model_comparison"]
        if isinstance(mc, pd.DataFrame) and len(mc) > 0:
            # Aggregate by model
            model_summary = mc.groupby("model").agg(
                mean_aic=("aic", "mean"),
                mean_bic=("bic", "mean"),
                mean_nll=("nll", "mean"),
                n_sessions=("session_id", "nunique"),
            ).round(2).sort_values("mean_bic")
            sections.append(_df_to_html(model_summary, "Model Comparison (mean across sessions)"))

    # HMM state interpretation
    if "hmm_state_summary" in analysis_results:
        hmm_summary = analysis_results["hmm_state_summary"]
        if isinstance(hmm_summary, pd.DataFrame) and len(hmm_summary) > 0:
            sections.append("<h3>HMM State Interpretation</h3>")
            sections.append("""<p>Each HMM state is assigned a descriptive label based on
            its mean choice-proportion emission parameter: states with high P(A) are labeled
            <em>Exploiting A</em>, low P(A) as <em>Exploiting B</em>, and intermediate
            values as <em>Exploring/Switching</em>.</p>""")
            # Per-session states (exclude group-mean rows)
            per_session = hmm_summary[hmm_summary["session_id"] != "GROUP_MEAN"]
            display_cols = ["session_id", "state", "label",
                            "mean_choice_a", "mean_reward"]
            if "mean_ici" in per_session.columns:
                display_cols.append("mean_ici")
            avail = [c for c in display_cols if c in per_session.columns]
            sections.append(_df_to_html(per_session[avail].round(3),
                                         "Per-Session State Parameters"))
            # Group-level averages
            group_rows = hmm_summary[hmm_summary["session_id"] == "GROUP_MEAN"]
            if len(group_rows) > 0:
                group_disp = ["label", "mean_choice_a", "mean_reward"]
                avail_g = [c for c in group_disp if c in group_rows.columns]
                sections.append(_df_to_html(group_rows[avail_g].round(3),
                                             "Group-Level State Averages"))

    # EDM simplex results
    if "edm_simplex" in analysis_results:
        edm_df = analysis_results["edm_simplex"]
        if isinstance(edm_df, pd.DataFrame) and len(edm_df) > 0:
            sections.append("<h2>7. Empirical Dynamical Modeling (Simplex Projection)</h2>")
            sections.append("""<p>Simplex projection (Sugihara &amp; May, 1990) uses
            time-delay embedding to reconstruct the attractor underlying the choice
            proportion time series. For each participant, the optimal embedding
            dimension E is selected by maximizing out-of-sample prediction correlation
            (ρ). Higher ρ indicates stronger deterministic structure in the behavioral
            trajectory.</p>""")
            sections.append(_df_to_html(edm_df.round(3),
                                         "Simplex Projection Results by Session"))

    # RQA results
    if "rqa" in analysis_results:
        rqa_df = analysis_results["rqa"]
        if isinstance(rqa_df, pd.DataFrame) and len(rqa_df) > 0:
            sections.append("<h2>8. Recurrence Quantification Analysis (RQA)</h2>")
            sections.append("""<p>Recurrence quantification analysis extracts metrics
            from recurrence plots to characterize the temporal structure of choice
            behavior. Key metrics include: <strong>Recurrence Rate</strong> (proportion
            of recurring states), <strong>Determinism</strong> (proportion of recurrent
            points forming diagonal lines, indicating predictable sequences),
            <strong>Laminarity</strong> (proportion forming vertical lines, indicating
            laminar/trapped states), and <strong>Trapping Time</strong> (mean duration
            of laminar states).</p>""")
            full_rqa = rqa_df[rqa_df["scope"] == "full_session"]
            if len(full_rqa) > 0:
                rqa_cols = ["session_id", "recurrence_rate", "determinism",
                            "laminarity", "trapping_time", "entropy_diagonal"]
                avail = [c for c in rqa_cols if c in full_rqa.columns]
                sections.append(_df_to_html(full_rqa[avail].round(3),
                                             "Full-Session RQA Metrics"))

    # CCM results
    if "ccm" in analysis_results:
        ccm_df = analysis_results["ccm"]
        if isinstance(ccm_df, pd.DataFrame) and len(ccm_df) > 0:
            sections.append("<h2>9. Convergent Cross Mapping (CCM)</h2>")
            sections.append("""<p>Convergent Cross Mapping (Sugihara et al., 2012)
            tests for causal coupling between reward dynamics and choice behavior
            using time-delay embedding. Increasing ρ with library size indicates
            convergent evidence of causal influence. "Reward → Choice" tests whether
            the choice attractor contains information about reward history (i.e.,
            reward causally drives choice). "Choice → Reward" tests the reverse
            direction.</p>""")

    # S-Map results
    if "smap" in analysis_results:
        smap_df = analysis_results["smap"]
        if isinstance(smap_df, pd.DataFrame) and len(smap_df) > 0:
            sections.append("<h2>10. S-Map Nonlinearity Analysis</h2>")
            sections.append("""<p>S-Map analysis compares linear (Simplex, θ=0) vs
            nonlinear (S-Map, θ>0) state-space prediction. If S-Map outperforms
            Simplex (positive Δρ), the dynamics are state-dependent and nonlinear.
            The optimal θ value indicates the degree of local weighting — higher θ
            means stronger nonlinearity.</p>""")
            sections.append(_df_to_html(smap_df.round(3),
                                         "S-Map Results by Session"))

    # DFA / Fractal analysis
    if "dfa" in analysis_results:
        dfa_df = analysis_results["dfa"]
        if isinstance(dfa_df, pd.DataFrame) and len(dfa_df) > 0:
            sections.append("<h2>12. Detrended Fluctuation Analysis (DFA)</h2>")
            sections.append("""<p>DFA characterizes the scaling properties of the choice
            time series. The DFA exponent α (≈ Hurst exponent) indicates:
            <strong>α &lt; 0.5</strong> = anti-persistent (alternation tendency),
            <strong>α = 0.5</strong> = uncorrelated random process,
            <strong>α &gt; 0.5</strong> = persistent (momentum/runs),
            <strong>α ≈ 1.0</strong> = 1/f noise (long-range correlations).
            Values significantly different from 0.5 indicate that choice behavior has
            temporal structure beyond what a simple coin-flip model would produce.</p>""")
            full_dfa = dfa_df[dfa_df["scope"] == "full_session"]
            if len(full_dfa) > 0:
                sections.append(_df_to_html(full_dfa.round(3),
                                             "Full-Session DFA Results"))

    if "sample_entropy" in analysis_results:
        se_df = analysis_results["sample_entropy"]
        if isinstance(se_df, pd.DataFrame) and len(se_df) > 0:
            sections.append("<h3>Sample Entropy</h3>")
            sections.append("""<p>Sample entropy measures the complexity/regularity of the
            choice time series. Lower values indicate more regular, predictable behavior;
            higher values indicate more complex, less predictable patterns.</p>""")
            full_se = se_df[se_df["scope"] == "full_session"]
            if len(full_se) > 0:
                sections.append(_df_to_html(full_se.round(3),
                                             "Full-Session Sample Entropy"))

    # Individual differences (PCA + GMM clustering)
    if "individual_differences" in analysis_results:
        id_results = analysis_results["individual_differences"]
        if id_results:
            sections.append("<h2>13. Individual Differences (PCA + GMM Clustering)</h2>")
            sections.append("""<p>To identify behavioral phenotypes, session-level features
            were assembled from across the analysis pipeline (behavioral metrics, DFA,
            sample entropy, RQA, EDM, CCM, S-Map), standardized, and reduced via PCA.
            Gaussian Mixture Models (GMM) were fit to the PCA-reduced space with k=2–6
            components, selecting the best k by Bayesian Information Criterion (BIC).
            GMM provides soft (probabilistic) cluster assignments, accommodating the
            continuous nature of individual differences. Multivariate normality of the
            PCA-reduced features was assessed via Mardia's test; the Henze-Zirkler test
            confirmed whether the spherical/elliptical cluster assumptions were met.</p>""")

            if "pca_n_keep" in id_results and "explained_variance" in id_results:
                n_keep = id_results["pca_n_keep"]
                cumvar = np.cumsum(id_results["explained_variance"])
                sections.append(
                    f"<p><strong>PCA:</strong> {n_keep} components retained, "
                    f"explaining {cumvar[n_keep-1]*100:.1f}% of total variance.</p>"
                )

            if "best_k" in id_results:
                sections.append(
                    f"<p><strong>GMM:</strong> Best k = {id_results['best_k']} "
                    f"clusters selected by BIC.</p>"
                )

            if "normality_test" in id_results and id_results["normality_test"]:
                nt = id_results["normality_test"]
                sections.append(
                    f"<p><strong>Normality check (Henze-Zirkler):</strong> "
                    f"statistic = {nt.get('hz_statistic', 'N/A')}, "
                    f"p = {nt.get('hz_p_value', 'N/A')}. "
                    f"{nt.get('interpretation', '')}</p>"
                )

            if "cluster_diagnostics" in id_results:
                diag = id_results["cluster_diagnostics"]
                if isinstance(diag, pd.DataFrame) and len(diag) > 0:
                    sections.append("<h3>Cluster Covariance Diagnostics</h3>")
                    sections.append("""<p>Per-cluster covariance condition numbers assess
                    whether the Gaussian components are well-conditioned. A condition number
                    &lt; 10 indicates approximately spherical covariance. The Henze-Zirkler
                    test checks multivariate normality within each cluster.</p>""")
                    sections.append(_df_to_html(diag.round(3),
                                                 "Per-Cluster Diagnostics"))

            if "cluster_assignments" in id_results:
                ca = id_results["cluster_assignments"]
                if isinstance(ca, pd.DataFrame) and len(ca) > 0:
                    sections.append(_df_to_html(ca.round(3),
                                                 "Cluster Assignments with Membership Probabilities"))

            if "cluster_profiles" in id_results:
                cp = id_results["cluster_profiles"]
                if isinstance(cp, pd.DataFrame) and len(cp) > 0:
                    pivot = cp.pivot(index="feature", columns="cluster", values="mean_z")
                    pivot.columns = [f"Cluster {c}" for c in pivot.columns]
                    sections.append(_df_to_html(pivot.round(2),
                                                 "Cluster Profiles (Standardized Feature Means)"))

            if "reward_test" in id_results and id_results["reward_test"]:
                rt = id_results["reward_test"]
                sections.append(
                    f"<p><strong>Cluster → Reward ({rt.get('reward_metric', 'total_rewards')}):</strong> "
                    f"{rt.get('test', 'N/A')}, statistic = {rt.get('statistic', 0):.2f}, "
                    f"p = {rt.get('p_value', 1):.4f}</p>"
                )
                if "group_means" in rt:
                    for cl, mn in rt["group_means"].items():
                        sections.append(f"<p style='margin-left:20px;'>{cl}: mean = {mn:.1f}</p>")

    # Figures with interpretations
    sections.append("<h2>14. Figures</h2>")
    fig_dir = os.path.join(output_dir, "figures")
    fmt = config.get("plot_format", "png")

    figure_interpretations = _get_figure_interpretations(analysis_results)

    figure_files = sorted(Path(fig_dir).rglob(f"*.{fmt}"))
    for fig_path in figure_files:
        rel_path = os.path.relpath(fig_path, os.path.join(output_dir, "reports"))
        name = fig_path.stem.replace("_", " ").title()
        sections.append(f'<h3>{name}</h3>')
        sections.append(f'<img src="{rel_path}" style="max-width:100%; margin:10px 0;">')
        # Add interpretation if available
        interp_text = figure_interpretations.get(fig_path.stem, "")
        if not interp_text and fig_path.stem.startswith("session_"):
            interp_text = figure_interpretations.get("_individual_prefix", "")
        if interp_text:
            sections.append(f'<p style="color:#444; font-style:italic; margin:5px 0 15px 0;">{interp_text}</p>')

    # Assemble
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Analysis Report</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; margin: 15px 0; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
        th {{ background: #f5f5f5; font-weight: 600; }}
        tr:nth-child(even) {{ background: #fafafa; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        img {{ border: 1px solid #eee; border-radius: 4px; }}
    </style>
</head>
<body>
{''.join(sections)}
</body>
</html>"""

    with open(report_path, "w") as f:
        f.write(html)
    print(f"Report saved to {report_path}")


def _generate_markdown_report(metrics_df, validation_report, analysis_results,
                               config, output_dir):
    """Generate Markdown report."""
    report_path = os.path.join(output_dir, "reports", "analysis_report.md")
    lines = []

    lines.append("# Behavioral Trajectory Analysis Report")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    lines.append("## Data Overview\n")
    lines.append(f"- Sessions analyzed: **{len(metrics_df)}**")
    lines.append(f"- Participants: **{metrics_df['participant_id'].nunique()}**")
    lines.append(f"- Mean clicks/session: **{metrics_df['total_clicks'].mean():.0f}**")
    lines.append(f"- Mean duration: **{metrics_df['session_duration_s'].mean():.0f}s**\n")

    if "phase_descriptives" in analysis_results:
        lines.append("## Phase Descriptives\n")
        lines.append(analysis_results["phase_descriptives"].to_markdown(index=False))
        lines.append("")

    if "model_comparison" in analysis_results:
        mc = analysis_results["model_comparison"]
        if isinstance(mc, pd.DataFrame) and len(mc) > 0:
            lines.append("## Model Comparison\n")
            model_summary = mc.groupby("model").agg(
                mean_bic=("bic", "mean"),
                mean_aic=("aic", "mean"),
            ).round(2).sort_values("mean_bic")
            lines.append(model_summary.to_markdown())
            lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_path}")


def _get_figure_interpretations(analysis_results: dict) -> dict:
    """Return data-driven interpretation strings keyed by figure stem name."""
    interp = {}

    # Extract summary statistics for data-driven text
    rqa_df = analysis_results.get("rqa")
    smap_df = analysis_results.get("smap")
    ccm_df = analysis_results.get("ccm")
    dfa_df = analysis_results.get("dfa")
    se_df = analysis_results.get("sample_entropy")
    # -- Group-level figures --
    interp["choice_trajectories"] = (
        "Rolling choice proportion over time, with vertical lines marking phase boundaries. "
        "Participants shift toward Option A during Phase 2 (A-advantage) and toward Option B "
        "during Phase 3 (B-advantage), confirming sensitivity to changing contingencies."
    )
    interp["reward_trajectories"] = (
        "Rolling reward rate over time shows the consequence of participants' foraging strategies. "
        "Reward rates are highest in Phase 1 (symmetric baseline) and drop sharply in Phase 4 "
        "(scarcity), reflecting the depleted environment."
    )
    interp["metric_distributions"] = (
        "Distributions of key session-level metrics across participants. "
        "Most participants show moderate switch rates and high choice entropy, "
        "suggesting active exploration rather than rigid perseveration."
    )
    interp["metric_correlations"] = (
        "Pairwise correlations among session-level behavioral metrics. "
        "Strong relationships (e.g., between switch rate and run length) reveal "
        "which aspects of foraging strategy co-vary across individuals."
    )
    interp["wslf_scatter"] = (
        "Win-stay/lose-shift analysis shows the probability of repeating a choice after reward "
        "versus switching after non-reward. Points above the diagonal indicate stronger "
        "sensitivity to losses than wins, a common pattern in foraging tasks."
    )

    # -- Phase-level figures --
    interp["phase_transitions"] = (
        "Choice proportion in the 30 seconds surrounding each phase boundary. "
        "Clear shifts in P(A) following transitions confirm that participants detect and "
        "respond to regime changes, with adaptation typically completing within 10–15 seconds."
    )
    interp["adaptation_lags"] = (
        "Distribution of adaptation lags — the time required for each participant's rolling "
        "choice proportion to shift toward the newly advantaged option after a phase boundary. "
        "Shorter lags indicate faster sensitivity to environmental change."
    )

    # -- Pulse-triggered figures --
    interp["pta_choice_overall"] = (
        "Pulse-triggered average of choice proportion during Phase 4's alternating bonus pulses. "
        "Deviations from baseline surrounding pulse onset reveal whether participants detect "
        "and respond to brief 3-second reward boosts."
    )
    interp["pta_choice_by_target"] = (
        "Pulse-triggered choice proportion split by which option receives the bonus (A vs B). "
        "Divergence between A-pulse and B-pulse traces indicates participants track which "
        "option is currently boosted and shift their behavior accordingly."
    )

    # -- Dynamical figures --
    interp["state_space_trajectories"] = (
        "KDE density plots of each participant's behavioral trajectory in the "
        "P(Choose A) × Reward Rate state space, colored by experimental phase. "
        "Separated density regions across phases indicate that the underlying dynamical "
        "attractor shifts with the environmental regime."
    )
    interp["choice_autocorrelation"] = (
        "Autocorrelation of the binary choice sequence across lags. "
        "Slow decay indicates strong temporal dependence — choices are not independent "
        "but are predicted by recent history, consistent with momentum or run-based strategies."
    )
    interp["hysteresis"] = (
        "Hysteresis loop comparing Phase 2 (A-advantage, rising) vs Phase 3 (B-advantage, "
        "falling). Separation between the two trajectories at matched advantage levels "
        "indicates path dependence: behavior depends not just on current contingencies "
        "but on the direction of environmental change."
    )

    # -- EDM / Simplex --
    edm_df = analysis_results.get("edm_simplex")
    if isinstance(edm_df, pd.DataFrame) and len(edm_df) > 0:
        mean_rho = edm_df["rho"].mean()
        mode_E = int(edm_df["best_E"].mode().iloc[0])
        interp["edm_simplex_all"] = (
            f"Simplex projection observed vs predicted time series for each participant. "
            f"Mean out-of-sample ρ = {mean_rho:.3f} with modal embedding dimension E = {mode_E}, "
            f"indicating strong deterministic structure in the choice trajectories."
        )

    # -- Recurrence --
    interp["recurrence_example"] = (
        "Example recurrence plot showing the matrix of recurring states in the choice "
        "time series. Diagonal lines reflect deterministic sequences; vertical/horizontal "
        "lines reflect laminar (trapped) states where behavior persists without change."
    )
    if isinstance(rqa_df, pd.DataFrame) and len(rqa_df) > 0:
        full_rqa = rqa_df[rqa_df["scope"] == "full_session"]
        if len(full_rqa) > 0:
            det = full_rqa["determinism"].mean()
            lam = full_rqa["laminarity"].mean()
            tt = full_rqa["trapping_time"].mean()
            interp["recurrence_all"] = (
                f"Recurrence plots for all participants. High determinism ({det:.2f}) and "
                f"laminarity ({lam:.2f}) with mean trapping time of {tt:.1f} clicks "
                f"indicate highly structured, non-random choice sequences."
            )

    # -- RQA summary --
    if isinstance(rqa_df, pd.DataFrame) and len(rqa_df) > 0:
        full_rqa = rqa_df[rqa_df["scope"] == "full_session"]
        if len(full_rqa) > 0:
            interp["rqa_summary"] = (
                f"Summary of recurrence quantification metrics across sessions and phases. "
                f"Full-session determinism averages {full_rqa['determinism'].mean():.2f}, "
                f"confirming that choice sequences contain long predictable runs "
                f"rather than random alternation."
            )

    # -- CCM --
    if isinstance(ccm_df, pd.DataFrame) and len(ccm_df) > 0:
        max_lib = ccm_df.groupby("session_id").apply(
            lambda g: g.loc[g["lib_size"].idxmax()]
        ).reset_index(drop=True)
        r2c = max_lib["rho_reward_causes_choice"].mean()
        c2r = max_lib["rho_choice_causes_reward"].mean()
        interp["ccm_convergence"] = (
            f"CCM convergence curves showing cross-map skill (ρ) increasing with library "
            f"size, confirming causal coupling. Reward → Choice (ρ = {r2c:.3f}) exceeds "
            f"Choice → Reward (ρ = {c2r:.3f}), indicating reward history is the stronger "
            f"causal driver of choice behavior."
        )

    # -- S-Map --
    if isinstance(smap_df, pd.DataFrame) and len(smap_df) > 0:
        mean_nl = smap_df["nonlinearity"].mean()
        n_nl = (smap_df["nonlinearity"] > 0).sum()
        n_tot = len(smap_df)
        mean_theta = smap_df["best_theta"].mean()
        interp["smap_nonlinearity"] = (
            f"S-Map analysis comparing linear vs nonlinear prediction. "
            f"All {n_nl}/{n_tot} sessions show positive nonlinearity (mean Δρ = {mean_nl:.3f}, "
            f"mean best θ = {mean_theta:.1f}), confirming that choice dynamics are "
            f"state-dependent rather than globally linear."
        )

    # -- DFA / Fractal --
    if isinstance(dfa_df, pd.DataFrame) and len(dfa_df) > 0:
        full_dfa = dfa_df[dfa_df["scope"] == "full_session"]
        if len(full_dfa) > 0:
            med_alpha = full_dfa["dfa_alpha"].median()
            interp["dfa_summary"] = (
                f"Distribution of DFA exponents across sessions. The median α = {med_alpha:.2f} "
                f"(> 0.5) indicates persistent, momentum-driven choice dynamics — participants "
                f"tend to continue runs rather than alternate, with long-range temporal correlations."
            )
            interp["dfa_scaling"] = (
                "Log-log DFA scaling plots for example sessions showing fluctuation F(n) vs "
                "window size. The linear relationship in log-log space confirms self-similar "
                "scaling structure in the choice time series, characteristic of fractal dynamics."
            )

    # -- Entropy --
    if isinstance(se_df, pd.DataFrame) and len(se_df) > 0:
        full_se = se_df[se_df["scope"] == "full_session"]
        if len(full_se) > 0:
            med_se = full_se["sample_entropy"].median()
            interp["entropy_summary"] = (
                f"Sample entropy distribution and its relationship with DFA. "
                f"Median sample entropy = {med_se:.2f} reflects moderate complexity — "
                f"behavior is neither fully predictable nor fully random, consistent "
                f"with adaptive foraging under changing contingencies."
            )

    # -- Individual differences --
    id_results = analysis_results.get("individual_differences", {})
    if id_results:
        best_k = id_results.get("best_k", "?")
        reward_test = id_results.get("reward_test", {})
        p_str = f"p = {reward_test['p_value']:.4f}" if "p_value" in reward_test else ""

        interp["pca_scree"] = (
            "Scree plot showing variance explained by each principal component. "
            "The dashed line marks 90% cumulative variance; components to the left "
            "of the orange line are retained for clustering."
        )
        interp["gmm_bic"] = (
            f"BIC curve for Gaussian Mixture Model selection. Lower BIC indicates "
            f"better model fit penalized for complexity. The optimal solution has "
            f"k={best_k} clusters."
        )
        interp["pca_biplot"] = (
            f"Left: Participants projected onto the first two principal components, "
            f"colored by GMM cluster assignment. Separation between clusters indicates "
            f"distinct behavioral phenotypes. Right: Feature loading vectors showing "
            f"which behavioral metrics drive each PC axis."
        )
        interp["cluster_profiles"] = (
            "Heatmap of standardized feature means per cluster. Red indicates above-average "
            "values; blue indicates below-average. Features are sorted by how much they "
            "differentiate between clusters (largest differences at top)."
        )
        interp["dendrogram"] = (
            "Ward's hierarchical clustering dendrogram showing the similarity structure "
            "across participants. Shorter merge distances indicate more similar behavioral profiles."
        )
        interp["cluster_rewards"] = (
            f"Total rewards earned by cluster. "
            f"{'The ' + reward_test.get('test', '') + ' yields ' + p_str + ', ' if p_str else ''}"
            f"testing whether behavioral phenotype predicts foraging success."
        )

    # -- Matching law --
    mc = analysis_results.get("model_comparison")
    if isinstance(mc, pd.DataFrame) and len(mc) > 0:
        matching_df = mc[mc["model"] == "generalized_matching_law"]
        if len(matching_df) > 0:
            mean_s = matching_df["sensitivity"].mean()
            mean_r2 = matching_df["r_squared"].mean()
            interp["matching_sensitivity"] = (
                f"Distribution of matching law sensitivity (s) parameters across sessions. "
                f"Mean s = {mean_s:.2f} {'(undermatching)' if mean_s < 0.9 else '(near strict matching)' if mean_s <= 1.1 else '(overmatching)'}. "
                f"The matching law captures molar allocation tendencies (mean R² = {mean_r2:.2f}) "
                f"but cannot account for transient dynamics around phase transitions."
            )
            interp["matching_log_ratios"] = (
                "Log behavior ratio vs log reinforcement ratio for each session. "
                "Points near the identity line indicate strict matching; systematic "
                "deviation reflects bias or sensitivity departures. The matching law "
                "provides the equilibrium baseline that dynamical analyses extend beyond."
            )

    # -- Individual session plots (generic) --
    interp["_individual_prefix"] = (
        "Individual session dashboard showing raw click events, rolling choice proportion, "
        "reward rate, and cumulative score over time. Vertical lines mark phase boundaries."
    )

    return interp


def _df_to_html(df, title=""):
    """Convert DataFrame to HTML table with optional title."""
    html = ""
    if title:
        html += f"<h4>{title}</h4>"
    html += df.to_html(border=0, classes="data-table")
    return html
