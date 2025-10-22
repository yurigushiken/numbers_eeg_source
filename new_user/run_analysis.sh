#!/bin/bash
# ============================================================================
# Quick-run script for new users (macOS/Linux)
# Run this file from Terminal to execute the example analysis
# ============================================================================

echo ""
echo "========================================"
echo " Running Example Analysis (13_31)"
echo "========================================"
echo ""
echo "This will run the full pipeline:"
echo "  1. Sensor-space analysis"
echo "  2. Source-space analyses (if significant)"
echo ""

# Activate the conda environment
# Note: You may need to run 'conda init bash' first if conda commands don't work
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate numbers_eeg_source

# Run the full analysis pipeline with the example sensor config
python -m code.run_full_analysis_pipeline \
  --config new_user/examples/sensor_13_31.yaml \
  --accuracy all

echo ""
echo "========================================"
echo " Analysis Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - Sensor: derivatives/sensor/sensor_13_31/"
echo "  - Source: derivatives/source/source_*_13_31/"
echo "  - Reports: derivatives/reports/"
echo ""
