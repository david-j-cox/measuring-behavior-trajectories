"""Assemble a complete manuscript output bundle."""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from analysis_pipeline.manuscript import __version__ as manuscript_version


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

    # ── 6. Write manifest / metadata ────────────────────────────────────────
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "manuscript_module_version": manuscript_version,
        "n_sessions": len(metrics_df),
        "n_participants": int(metrics_df["participant_id"].nunique()),
        "n_events": len(events_df),
        "contents": {},
    }

    for subdir in ["figures", "supplementary", "tables", "captions", "results_summary"]:
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
    lines.append("```bash")
    lines.append("cd analysis")
    lines.append("source venv/bin/activate")
    lines.append("python run_manuscript_analysis.py --config config.yaml")
    lines.append("```")
    lines.append("")
    lines.append("The `analysis_config.json` file records the exact configuration used.")
    lines.append("")
    return "\n".join(lines)
