#!/bin/bash
# ============================================================================
# Quick-run script for new users (macOS/Linux)
#
# Usage:
#   ./run_analysis.sh <path-to-sensor-config.yaml>
#
# Example:
#   ./run_analysis.sh new_user/configs/13_31/sensor_13_31.yaml
# ============================================================================

if [ -z "$1" ]; then
    echo "Error: No config file specified!"
    echo ""
    echo "Usage: ./run_analysis.sh <path-to-sensor-config.yaml>"
    echo ""
    echo "Example:"
    echo "  ./run_analysis.sh new_user/configs/13_31/sensor_13_31.yaml"
    echo ""
    exit 1
fi

echo ""
echo "========================================"
echo " Running Analysis Pipeline"
echo "========================================"
echo ""
echo "Config: $1"
echo ""
echo "This will run the full pipeline:"
echo "  1. Sensor-space analysis"
echo "  2. Source-space analyses (if significant)"
echo ""

# Activate the conda environment
# Note: You may need to run 'conda init bash' first if conda commands don't work
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate numbers_eeg_source

# Set the output directory to new_user/derivatives/ (keeps training outputs separate)
export DERIVATIVES_ROOT=new_user/derivatives

# Run the full analysis pipeline with the specified sensor config
python -m code.run_full_analysis_pipeline \
  --config "$1" \
  --accuracy all

echo ""
echo "========================================"
echo " Analysis Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - Sensor: new_user/derivatives/sensor/sensor_13_31/"
echo "  - Source: new_user/derivatives/source/source_*_13_31/"
echo "  - Reports: new_user/derivatives/reports/"
echo ""
