@echo off
REM ============================================================================
REM Quick-run script for new users
REM
REM Usage:
REM   run_analysis.bat <path-to-sensor-config.yaml>
REM
REM Example:
REM   run_analysis.bat new_user/configs/13_31/sensor_13_31.yaml
REM ============================================================================

if "%~1"=="" (
    echo Error: No config file specified!
    echo.
    echo Usage: run_analysis.bat ^<path-to-sensor-config.yaml^>
    echo.
    echo Example:
    echo   run_analysis.bat new_user/configs/13_31/sensor_13_31.yaml
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Running Analysis Pipeline
echo ========================================
echo.
echo Config: %~1
echo.
echo This will run the full pipeline:
echo   1. Sensor-space analysis
echo   2. Source-space analyses (if significant)
echo.

REM Activate the conda environment
call conda activate numbers_eeg_source

REM Set the output directory to new_user/derivatives/ (keeps training outputs separate)
set DERIVATIVES_ROOT=new_user/derivatives

REM Run the full analysis pipeline with the specified sensor config
python -m code.run_full_analysis_pipeline ^
  --config %~1 ^
  --accuracy all

echo.
echo ========================================
echo  Analysis Complete!
echo ========================================
echo.
echo Results saved to:
echo   - Sensor: new_user/derivatives/sensor/sensor_13_31/
echo   - Source: new_user/derivatives/source/source_*_13_31/
echo   - Reports: new_user/derivatives/reports/
echo.
pause
