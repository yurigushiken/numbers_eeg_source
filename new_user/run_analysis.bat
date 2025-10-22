@echo off
REM ============================================================================
REM Quick-run script for new users
REM Double-click this file or run from PowerShell to execute the example analysis
REM ============================================================================

echo.
echo ========================================
echo  Running Example Analysis (13_31)
echo ========================================
echo.
echo This will run the full pipeline:
echo   1. Sensor-space analysis
echo   2. Source-space analyses (if significant)
echo.

REM Activate the conda environment
call conda activate numbers_eeg_source

REM Run the full analysis pipeline with the example sensor config
python -m code.run_full_analysis_pipeline ^
  --config configs/new_user/examples/sensor_13_31.yaml ^
  --accuracy all

echo.
echo ========================================
echo  Analysis Complete!
echo ========================================
echo.
echo Results saved to:
echo   - Sensor: derivatives/sensor/sensor_13_31/
echo   - Source: derivatives/source/source_*_13_31/
echo   - Reports: derivatives/reports/
echo.
pause
