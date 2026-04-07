"""Assemble a complete manuscript output bundle."""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from analysis_pipeline.manuscript import __version__ as manuscript_version

# Columns that contain personally-identifiable information
_ID_COLUMNS = ["participant_id", "prolific_id", "session_id"]


def _deidentify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with identifying columns replaced by anonymous
    sequential IDs (P001, S001, etc.).

    The mapping is deterministic within a single call so that the same
    participant/session always gets the same anonymous label.
    """
    df = df.copy()

    for col, prefix in [
        ("participant_id", "P"),
        ("prolific_id", "P"),  # map to same participant namespace
        ("session_id", "S"),
    ]:
        if col not in df.columns:
            continue
        unique_vals = df[col].dropna().unique()
        id_map = {
            val: f"{prefix}{str(i + 1).zfill(3)}"
            for i, val in enumerate(sorted(unique_vals, key=str))
        }
        df[col] = df[col].map(id_map)

    return df


def export_manuscript_package(events_df: pd.DataFrame, metrics_df: pd.DataFrame,
                              analysis_results: dict, config: dict,
                              output_dir: str):
    """
    Copy all manuscript outputs into a single self-contained directory:
      outputs/manuscript/manuscript_package/
    """
    pkg_dir = os.path.join(output_dir, "manuscript", "manuscript_package")
    os.makedirs(pkg_dir, exist_ok=True)

    src_root = os.path.join(output_dir, "manuscript")

    # ── 1. Copy figures ─────────────────────────────────────────────────────
    for subdir in ["figures", "supplementary"]:
        src = os.path.join(src_root, subdir)
        dst = os.path.join(pkg_dir, subdir)
        if os.path.exists(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    # ── 2. Copy tables ──────────────────────────────────────────────────────
    src = os.path.join(src_root, "tables")
    dst = os.path.join(pkg_dir, "tables")
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # ── 3. Copy captions ────────────────────────────────────────────────────
    src = os.path.join(src_root, "captions")
    dst = os.path.join(pkg_dir, "captions")
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # ── 4. Copy results summary and outline ─────────────────────────────────
    src = os.path.join(src_root, "results_summary")
    dst = os.path.join(pkg_dir, "results_summary")
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # ── 5. Write analysis config used ───────────────────────────────────────
    config_path = os.path.join(pkg_dir, "analysis_config.json")
    # Convert any non-serializable values
    config_clean = _make_serializable(config)
    with open(config_path, "w") as f:
        json.dump(config_clean, f, indent=2)

    # ── 5b. Copy config.yaml verbatim ───────────────────────────────────────
    # Look for config.yaml next to run_analysis.py (the analysis root)
    analysis_root = os.path.dirname(output_dir)  # outputs/ sits inside analysis/
    config_yaml_src = os.path.join(analysis_root, "config.yaml")
    if os.path.isfile(config_yaml_src):
        shutil.copy2(config_yaml_src, os.path.join(pkg_dir, "config.yaml"))
    else:
        # Fall back: write the in-memory config as YAML
        with open(os.path.join(pkg_dir, "config.yaml"), "w") as f:
            yaml.dump(config_clean, f, default_flow_style=False, sort_keys=False)

    # ── 5c. De-identified data export ───────────────────────────────────────
    data_dir = os.path.join(pkg_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    events_deident = _deidentify_dataframe(events_df)
    events_deident.to_csv(
        os.path.join(data_dir, "events_deidentified.csv"), index=False
    )

    metrics_deident = _deidentify_dataframe(metrics_df)
    metrics_deident.to_csv(
        os.path.join(data_dir, "session_metrics.csv"), index=False
    )

    # ── 5d. Data README ─────────────────────────────────────────────────────
    data_readme = _build_data_readme()
    with open(os.path.join(data_dir, "README.md"), "w") as f:
        f.write(data_readme)

    # ── 5e. One-command reproduction script ─────────────────────────────────
    _write_reproduce_script(pkg_dir)

    # Copy requirements.txt into the package
    req_src = os.path.join(analysis_root, "requirements.txt")
    if os.path.isfile(req_src):
        shutil.copy2(req_src, os.path.join(pkg_dir, "requirements.txt"))

    # ── 6. Write manifest / metadata ────────────────────────────────────────
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "manuscript_module_version": manuscript_version,
        "n_sessions": len(metrics_df),
        "n_participants": int(metrics_df["participant_id"].nunique()),
        "n_events": len(events_df),
        "contents": {},
    }

    for subdir in ["figures", "supplementary", "tables", "captions",
                    "results_summary", "data"]:
        d = os.path.join(pkg_dir, subdir)
        if os.path.exists(d):
            files = sorted(os.listdir(d))
            manifest["contents"][subdir] = files

    manifest_path = os.path.join(pkg_dir, "MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── 7. Write a README ──────────────────────────────────────────────────
    readme = _build_readme(manifest)
    with open(os.path.join(pkg_dir, "README.md"), "w") as f:
        f.write(readme)

    # Count files
    total_files = sum(
        len(files) for _, _, files in os.walk(pkg_dir)
    )
    print(f"Manuscript package: {total_files} files in {pkg_dir}")


def _make_serializable(obj):
    """Recursively convert config to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def _write_reproduce_script(pkg_dir: str):
    """Write a one-command reproduction bash script into the package."""
    script = """\
#!/bin/bash
# Reproduce all analyses from the manuscript
# Usage: ./reproduce.sh
set -e

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Running full analysis pipeline ==="
python run_analysis.py \\
    --config config.yaml \\
    --steps validate transform metrics plots phase pulse dynamical \\
           fractal models hmm individual_differences null_comparison \\
           robustness report manuscript

echo "=== Done ==="
"""
    script_path = os.path.join(pkg_dir, "reproduce.sh")
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)


def _build_data_readme():
    """Build a README describing the de-identified data files."""
    return """\
# De-identified Data

## De-identification Procedure

All personally-identifiable information has been removed from these data files
before inclusion in this manuscript package. The following columns were replaced
with anonymous sequential identifiers:

| Original Column   | Replacement Format | Description                          |
|--------------------|--------------------|--------------------------------------|
| `participant_id`   | P001, P002, ...    | Anonymous participant label           |
| `prolific_id`      | P001, P002, ...    | Mapped to same namespace as above     |
| `session_id`        | S001, S002, ...    | Anonymous session label               |

The mapping between original identifiers and anonymous labels is **not**
included in this package and is stored separately under restricted access.

## Files

- **events_deidentified.csv** -- Trial-level event log with all behavioral
  events (clicks, rewards, timestamps, derived variables). See
  `outcome_definitions.md` in the project repository for complete column
  definitions and derivation details.

- **session_metrics.csv** -- Session-level summary metrics (one row per
  session). Includes aggregate measures such as total clicks, switch rates,
  reward rates, and phase-level statistics.

## Column Reference

For full column definitions, derivation formulae, and outcome variable
descriptions, refer to `outcome_definitions.md` in the project repository.

## Ethics Statement

[PLACEHOLDER] This study was approved by [Institution] Institutional Review
Board (Protocol #XXXX-XXXX). All participants provided informed consent prior
to participation. Data were collected via the Prolific platform. No
personally-identifiable information is included in this dataset.
"""


def _build_readme(manifest):
    """Build a brief README for the manuscript package."""
    lines = [
        "# Manuscript Output Package",
        "",
        f"Generated: {manifest['generated_at']}",
        f"Pipeline version: {manifest['manuscript_module_version']}",
        "",
        f"- Sessions: {manifest['n_sessions']}",
        f"- Participants: {manifest['n_participants']}",
        f"- Total events: {manifest['n_events']}",
        "",
        "## Contents",
        "",
    ]
    for subdir, files in manifest.get("contents", {}).items():
        lines.append(f"### {subdir}/")
        for f in files:
            lines.append(f"- {f}")
        lines.append("")

    lines.append("## Reproducing")
    lines.append("")
    lines.append("Run the included reproduction script:")
    lines.append("")
    lines.append("```bash")
    lines.append("./reproduce.sh")
    lines.append("```")
    lines.append("")
    lines.append("Or manually:")
    lines.append("")
    lines.append("```bash")
    lines.append("pip install -r requirements.txt")
    lines.append("python run_analysis.py --config config.yaml \\")
    lines.append("    --steps validate transform metrics plots phase pulse dynamical \\")
    lines.append("           fractal models hmm individual_differences null_comparison \\")
    lines.append("           robustness report manuscript")
    lines.append("```")
    lines.append("")
    lines.append("The `analysis_config.json` file records the exact configuration used.")
    lines.append("The `config.yaml` file is the YAML configuration consumed by the pipeline.")
    lines.append("")
    lines.append("## De-identified Data")
    lines.append("")
    lines.append("The `data/` directory contains de-identified versions of the behavioral")
    lines.append("data. See `data/README.md` for details on the de-identification procedure.")
    lines.append("")
    return "\n".join(lines)
