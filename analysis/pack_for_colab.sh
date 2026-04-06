#!/bin/bash
# Pack analysis code for Colab upload.
# Run from the analysis/ directory:
#   bash pack_for_colab.sh
#
# This creates analysis_code.zip containing:
#   - analysis_pipeline/ (all Python modules)
#   - run_empirical_paper.py (entry point)
#   - config.yaml
#
# Then upload analysis_code.zip + data/events.csv to Colab.

set -e
cd "$(dirname "$0")"

ZIP_NAME="analysis_code.zip"
rm -f "$ZIP_NAME"

zip -r "$ZIP_NAME" \
    analysis_pipeline/*.py \
    analysis_pipeline/models/*.py \
    analysis_pipeline/empirical_paper/*.py \
    run_empirical_paper.py \
    config.yaml \
    -x "analysis_pipeline/manuscript_tutorial_archive/*" \
    -x "analysis_pipeline/__pycache__/*" \
    -x "analysis_pipeline/models/__pycache__/*"

echo ""
echo "Created: $ZIP_NAME ($(du -h "$ZIP_NAME" | cut -f1))"
echo ""
echo "Upload these two files to Colab:"
echo "  1. $ZIP_NAME"
echo "  2. data/events.csv"
