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
            sections.append(_df_to_html(pd.DataFrame(tests).round(4),
                                         "Paired t-tests"))

    if "adaptation_lags" in analysis_results:
        sections.append("<h3>Adaptation Lags</h3>")
        for phase, stats in analysis_results["adaptation_lags"].items():
            sections.append(f"<p><strong>{phase}:</strong> "
                            f"Adapted: {stats['n_adapted']}/{stats['n_total']}, "
                            f"Mean lag: {stats['mean_lag_s']:.1f}s, "
                            f"Median: {stats['median_lag_s']:.1f}s</p>")

    # Model comparison
    if "model_comparison" in analysis_results:
        sections.append("<h2>5. Model Comparison</h2>")
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

    # Figures
    sections.append("<h2>6. Figures</h2>")
    fig_dir = os.path.join(output_dir, "figures")
    fmt = config.get("plot_format", "png")
    figure_files = sorted(Path(fig_dir).rglob(f"*.{fmt}"))
    for fig_path in figure_files:
        rel_path = os.path.relpath(fig_path, os.path.join(output_dir, "reports"))
        name = fig_path.stem.replace("_", " ").title()
        sections.append(f'<h3>{name}</h3>')
        sections.append(f'<img src="{rel_path}" style="max-width:100%; margin:10px 0;">')

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


def _df_to_html(df, title=""):
    """Convert DataFrame to HTML table with optional title."""
    html = ""
    if title:
        html += f"<h4>{title}</h4>"
    html += df.to_html(border=0, classes="data-table")
    return html
