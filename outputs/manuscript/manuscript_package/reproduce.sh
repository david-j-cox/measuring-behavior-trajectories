#!/bin/bash
# Reproduce all analyses from the manuscript
# Usage: ./reproduce.sh
set -e

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Running full analysis pipeline ==="
python run_analysis.py \
    --config config.yaml \
    --steps validate transform metrics plots phase pulse dynamical \
           fractal models individual_differences report

echo "=== Done ==="
