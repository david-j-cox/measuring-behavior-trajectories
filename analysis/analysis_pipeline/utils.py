"""Shared utilities."""

import yaml
import os
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def ensure_dirs(output_dir: str):
    """Create output directory structure."""
    subdirs = [
        "figures/individual", "figures/group", "figures/phase",
        "figures/pulse", "figures/dynamical", "figures/models",
        "tables", "reports", "diagnostics"
    ]
    for sub in subdirs:
        Path(os.path.join(output_dir, sub)).mkdir(parents=True, exist_ok=True)
