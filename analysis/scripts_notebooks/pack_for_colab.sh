#!/bin/bash
# Pack analysis code for Colab upload.
# Run from the analysis/ directory:
#   bash pack_for_colab.sh
#
# This creates analysis_code.zip containing:
#   - pipeline/ (all Python modules)
#   - run.py (entry point)
#   - config.yaml
#
# Then upload analysis_code.zip + data/events.csv to Colab.

set -e
cd "$(dirname "$0")"

ZIP_NAME="analysis_code.zip"
rm -f "$ZIP_NAME"

zip -r "$ZIP_NAME" \
    pipeline/*.py \
    pipeline/models/*.py \
    pipeline/figures/*.py \
    run.py \
    config.yaml \
    -x "pipeline/manuscript_tutorial_archive/*" \
    -x "pipeline/__pycache__/*" \
    -x "pipeline/models/__pycache__/*"

echo ""
echo "Created: $ZIP_NAME ($(du -h "$ZIP_NAME" | cut -f1))"
echo ""
echo "Upload these two files to Colab:"
echo "  1. $ZIP_NAME"
echo "  2. data/events.csv"
