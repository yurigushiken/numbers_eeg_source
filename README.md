# EEG Analysis Pipeline

A reproducible workflow for **sensor-space** and **source-space** EEG analyses using MNE-Python, with non-parametric cluster-based permutation statistics and transparent preprocessing defaults. This project follows the methods described in:

**Jas, M., Larson, E., Engemann, D. A., Leppäkangas, J., Taulu, S., Hämäläinen, M., & Gramfort, A. (2018).** *A reproducible MEG/EEG group study with the MNE software: Recommendations, quality assessments, and good practices.* https://pmc.ncbi.nlm.nih.gov/articles/PMC6088222/


## Overview

The core philosophy is to let the data reveal statistically significant differences without a priori assumptions about when or where effects will occur. This method controls for multiple comparisons across all time points, sensors, and source vertices.

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
    -   `utils/`: Helper modules for data loading, statistics, plotting, and reporting.
-   `configs/`: Contains all analysis configuration files.
-   `derivatives/`: The output directory for all generated figures and reports, organized by analysis name and domain (sensor/source).

## How to Run an Analysis

PLease execute all scripts as Python modules from the **root directory of the project** (e.g., `D:/numbers_eeg/`). Ensure the `numbers_eeg_source` conda environment is active.

**Command Structure:**

```bash
# General format for sensor space
conda activate numbers_eeg_source; python -m code.run_sensor_analysis_pipeline --config <path_to_config> --accuracy <dataset>

# General format for source space
conda activate numbers_eeg_source; python -m code.run_source_analysis_pipeline --config <path_to_config> --accuracy <dataset>
```

**Example Analyses:**

```bash
# Example 1: Run the 'change vs. no-change' sensor-space analysis
conda activate numbers_eeg_source; python -m code.run_sensor_analysis_pipeline --config configs/sensor_change_vs_no-change.yaml --accuracy all

# Example 2: Run the 'prime 1 vs prime 3' source-space analysis
conda activate numbers_eeg_source; python -m code.run_source_analysis_pipeline --config configs/source_prime1-land3_vs_prime3-land1.yaml --accuracy all
```

**Arguments:**

-   `--config`: The path to the `.yaml` file defining the entire analysis from contrast to statistics.
-   `--accuracy`: The dataset to use (`acc1` for correct trials, `all` for all trials).
-   `--data-source`: Data source selection. `new` (default) uses combined preprocessed files under `data/data_preprocessed/<preprocessing>`; `old` uses legacy split condition folders under `data/<accuracy>/sub-XX`; you may also pass a custom path (e.g., `data/data_preprocessed/hpf_1.5_lpf_40_baseline-on`).

> Note: On Windows/PowerShell use semicolons to chain commands (e.g., `conda activate ...; python ...`).

### Full pipeline behavior and config pairing

-   Always pass a sensor-space YAML to the full pipeline.
-   The full pipeline runs sensor analysis first and parses the sensor stats report.
-   If at least one significant sensor cluster is found, it then runs source analysis.
-   The source config is inferred by replacing "sensor" with "source" in the sensor config path and in the analysis name.
    -   Example pair: `configs/sensor_any_landing1_vs_any_landing3.yaml` ↔ `configs/source_any_landing1_vs_any_landing3.yaml`
-   If the matching source YAML is missing, source analysis is skipped and the report is still created with sensor-only results.
-   Note: The YAML key `domain` is informational; the full pipeline uses filename pairing (sensor→source), not the `domain` value, to decide which analyses to run.

### Output locations

-   Sensor outputs: `derivatives/sensor/<analysis_name>/`
-   Source outputs: `derivatives/source/<analysis_name_with_sensor→source>/`
-   Combined report (neutral name): `derivatives/reports/<base_name>_report.html` and `.pdf`
    -   `<base_name>` strips any leading `sensor_`/`source_` (e.g., `any_landing1_vs_any_landing3`).
-   If sensor is non‑significant or matching source YAML is missing, the Source section in the HTML shows a brief “null” message (sensor results are still fully reported).

### Figures in the combined report

-   Condition ERPs (Figure 0):
    -   Optional ROI ERPs for P1 (Oz), N1 (bilateral), and P3b (midline), shown when both condition grand averages are available.
-   ERP Cluster (left):
    -   Grand‑average contrast over the channels in the most significant sensor cluster (p‑ranked).
    -   Shaded band = that cluster’s time window; legend shows `Cluster #k (p=...)`.
-   T‑Value Topomap (right):
    -   T‑values averaged across the same cluster window.
    -   Only cluster channels are highlighted with hollow markers; peak sensor is annotated. Scalp isocontours may be hidden for clarity.
-   Source Surface (six views):
    -   Summarizes significant source clusters at an informative time around the global peak using signed dSPM with `pick_ori='normal'`.
    -   ROI‑restricted runs plot on the ROI vertex set; a single unified colorbar sits below the grid.

## Key pipeline updates and defaults

-   Signed source estimates: inverse application uses `pick_ori='normal'` so source time courses keep polarity, avoiding magnitude inflation.
-   Preprocessing: baseline correction applied from YAML; average EEG reference is enforced as a projector for robustness; the previous 1 Hz FIR high‑pass on short epochs is not used.
-   Inverse solution parameters: `loose` and `depth` values are now configurable in each source YAML file. The on-the-fly fallback uses EEG-appropriate defaults (`depth=3.0`), and you can specify these values via CLI arguments (`--loose`, `--depth`) when pre-computing operators.
-   Time‑window discipline: source analyses (and label time‑series fallback) crop data to YAML `tmin`/`tmax`; sensor analyses evaluate the full epoch and report/plot the significant window found by clustering.
-   Tail handling and thresholds: both domains honor `tail` (−1/0/+1) in the test; cluster‑forming threshold is tail‑aware in source, while sensor currently uses a two‑sided threshold.
-   ROI‑restricted clustering: optional anatomical restriction (e.g., `aparc` labels). Plotting respects ROI vertices.
-   Label time‑series fallback: include 1D ROI cluster results when full spatio‑temporal stats are null (config via YAML).
-   Combined report naming is neutral: `<base_name>_report.{html,pdf}`; source figure uses six views and a unified colorbar.

## Quickstart (PowerShell examples)

```powershell
# Create and activate a fresh environment (conda)
conda create -n sfn2 python=3.11 -y; conda activate sfn2
conda install -c conda-forge mne numpy scipy matplotlib pandas tqdm imageio nibabel nilearn pyvista pyvistaqt pyqt vtk pyyaml -y
# Optional (PDF export): conda install -c conda-forge weasyprint cairo pango gdk-pixbuf -y

# Activate env and run SENSOR pipeline
python -m code.run_sensor_analysis_pipeline --config configs/sensor_prime1-land3_vs_prime3-land1.yaml --accuracy all

# Activate env and run SOURCE pipeline
python -m code.run_source_analysis_pipeline --config configs/source_prime1-land3_vs_prime3-land1.yaml --accuracy all

# Precompute inverse operators (recommended before source analyses)
# Note: Use --depth 3.0 for an EEG-appropriate depth weighting.
python -m code.build_inverse_solutions --depth 3.0

# Run FULL pipeline (builds combined HTML report)
python -m code.run_full_analysis_pipeline --config configs/sensor_prime1-land3_vs_prime3-land1.yaml --accuracy all

# Aggregated-power example: All landing on 1 vs All landing on 3
python -m code.run_full_analysis_pipeline --config configs/sensor_any_landing1_vs_any_landing3.yaml --accuracy all
```

## Configuration tips (YAML)

-   Stats sensitivity
    -   **p_threshold**: cluster‑forming threshold (e.g., 0.001–0.05). Larger is more sensitive to forming clusters, but raises FWER bar.
    -   **cluster_alpha**: cluster‑level alpha (typically 0.05).
    -   **tail**: use `1` or `-1` for one‑tailed tests if direction is known; `0` for two‑tailed.
    -   **n_permutations**: increase for stable p‑values (e.g., 5000–10000).
-   Time window: set `tmin`/`tmax` to bracket the expected effect (often informed by sensors).
-   ROI restriction (optional):
    ```yaml
    stats:
      roi:
        parc: aparc
        labels: ["cuneus", "lingual", "lateraloccipital"]
    ```
-   Label time‑series fallback (optional):
    ```yaml
    stats:
      label_timeseries:
        parc: aparc
        labels: ["cuneus", "lingual", "lateraloccipital"]
        mode: "mean_flip"
    ```
-   Inverse Solution Parameters (source-space only):
    ```yaml
    inverse:
      loose: 0.2
      depth: 3.0
    ```

## Troubleshooting

-   UnicodeEncodeError (Windows cp1252) while writing HTML
    -   Avoid special hyphens; prefer standard `-`. If needed, set `PYTHONIOENCODING=utf-8` for the session and rerun.
-   No significant source clusters
    -   Narrow the time window (±10–20 ms), use one‑tailed test when justified, restrict ROI, increase permutations, or enable label time‑series fallback.
-   Source plot missing in combined HTML
    -   Run the full pipeline (not just source) to regenerate; hard‑refresh viewer; confirm the PNG at `derivatives/source/<analysis>/<analysis>_source_cluster.png`.
-   PDF not produced
    -   Install a converter (see below) and rerun the full pipeline; HTML will always be created even if PDF fails.

## Optional: PDF export dependencies

The report generator can export a PDF alongside HTML if one of the following is installed and on PATH:

-   wkhtmltopdf CLI (recommended on Windows), or
-   WeasyPrint (pure Python; requires Cairo/Pango/GDK‑PixBuf).

Example (conda‑forge):

```powershell
conda activate numbers_eeg_source; conda install -c conda-forge weasyprint cairo pango gdk-pixbuf
```

If no converter is available, the pipeline logs a warning and continues with HTML only.


Notes:
- On first HTTPS push, Windows Git Credential Manager opens a browser sign‑in. Log in as the repo owner and approve access. Future pushes will succeed without prompts.
- If you previously signed in as the wrong user or have stale tokens, open Windows “Credential Manager” → Windows Credentials → remove entries for `git:https://github.com`, then push again to re‑authenticate.
- We intentionally ignore `data/` and `derivatives/` via `.gitignore` so the repository only contains code, configs, and docs.


conda activate numbers_eeg_source

## Project-wide Defaults (common.yaml)

- `configs/common.yaml` provides minimal, centralized defaults to improve reproducibility and make the default data choice explicit.
- Current defaults:
  - `data.source: new` and `data.preprocessing: hpf_1.0_lpf_35_baseline-on`
  - `epoch_defaults.baseline: [-0.2, 0.0]`
- Behavior:
  - When loading an analysis config, the loader fills in a missing `epoch_window.baseline` from `epoch_defaults.baseline` (does not overwrite if present).
  - Subject discovery for `--data-source new` uses the preprocessing variant from `common.yaml` (no hardcoded path).
  - CLI `--data-source` always overrides the default (you can pass `new`, `old`, or a custom path).

Example `configs/common.yaml`:

```
data:
  source: "new"
  preprocessing: "hpf_1.0_lpf_35_baseline-on"

epoch_defaults:
  baseline: [-0.2, 0.0]

montage: "GSN-HydroCel-129"
reference: "average"
```

Testing notes (TDD): lightweight unit tests for the config merge and data-source resolution live under `tests/test_config_utils.py`. They avoid importing heavy EEG libraries and validate that defaults merge correctly and path resolution behaves as expected.
