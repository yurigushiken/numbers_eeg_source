# EEG Analysis Pipeline

A reproducible workflow for **sensor-space** and **source-space** EEG analyses using MNE-Python, with non-parametric cluster-based permutation statistics and transparent preprocessing defaults. This project follows the methods described in:

**Jas, M., Larson, E., Engemann, D. A., Leppäkangas, J., Taulu, S., Hämäläinen, M., & Gramfort, A. (2018).** *A reproducible MEG/EEG group study with the MNE software: Recommendations, quality assessments, and good practices.* https://pmc.ncbi.nlm.nih.gov/articles/PMC6088222/


## How it Works

You can control the analysis with a single YAML (`.yaml`) configuration file and execute with a single Python script. The pipeline performs the following steps automatically:

1.  **Load Config:** Parses the specified `.yaml` file.
2.  **Load Data:** Gathers all required subject epoch files.
3.  **Compute Contrasts:** For each subject, calculates the difference wave between the two conditions of interest (e.g., "Change" vs. "No Change").
4.  **(Source Analysis Only) Inverse Operators:** The source step expects a precomputed inverse operator file (`<sub-XX>-inv.fif`) in each subject's derivatives folder. Subjects missing this file are skipped in the source step. Use the helper `code.build_inverse_solutions` to create them (fsaverage‑based) before running source statistics.
5.  **Run Group Statistics:** Performs a spatio-temporal cluster permutation test on the contrasts from all subjects.
6.  **Generate Outputs:** Creates a dedicated output directory containing:
    *   A detailed statistical report (`..._report.txt`).
    *   Visualizations of the results (ERP plots and topomaps for sensor space; brain surface plots for source space).
    *   The grand average contrast file (`...-ave.fif` or `...-stc.h5`).

## Project Structure

-   `code/`: Contains all Python analysis scripts.
    -   `run_sensor_analysis_pipeline.py`: Main entrypoint for sensor-space analyses.
    -   `run_source_analysis_pipeline.py`: Main entrypoint for source-space analyses.
    -   `run_full_analysis_pipeline.py`: Runs sensor analysis, then auto-discovers and runs matching source analyses.
    -   `utils/`: Helper modules for data loading, statistics, plotting, and reporting.
-   `configs/`: Contains all analysis configuration files.
    -   `sensor_roi_definitions.yaml`: Central repository for sensor-space ROI definitions (channel groups).
    -   Individual analysis folders with sensor and source YAML configs.
-   `derivatives/`: The output directory for all generated figures and reports, organized by analysis name and domain (sensor/source).

