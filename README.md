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
-   `--accuracy` (optional): Override trial filtering for both conditions at once. Use `acc1` for correct trials, `acc0` for incorrect trials, or `all` for all trials. If omitted, the pipeline honors per‑condition `accuracy` fields in the YAML (falling back to `all` per side). The loader filters using the metadata column `Target.ACC` (≥0.5 treated as accurate); if that column is missing or filtering removes everything, it falls back to all trials for the affected condition and logs a warning.

> Note: On Windows/PowerShell use semicolons to chain commands (e.g., `conda activate ...; python ...`).

### Full pipeline behavior and config pairing

-   Always pass a sensor-space YAML to the full pipeline.
-   The full pipeline runs sensor analysis first and parses the sensor stats report.
-   If at least one significant sensor cluster is found, the pipeline searches the sensor config directory for every YAML that (a) ends with the same contrast slug (for example `_cardinality1_vs_cardinality2.yaml`) and (b) declares `domain: "source"`.
    -   Keep the sensor file named `sensor_<slug>.yaml`. Name each source method `source_<method>_<slug>.yaml` (e.g., `source_dspm_cardinality1_vs_cardinality2.yaml`, `source_loreta_cardinality1_vs_cardinality2.yaml`).
    -   All discovered source configs are executed in alphabetical order, so you can run multiple inverse methods without touching the CLI command.
-   If no companion source configs are found (or none produce significant clusters), the combined report still includes the full sensor results and notes that the corresponding source analysis was skipped.

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
    -   Shaded band = that cluster's time window; legend shows `Cluster #k (p=...)`.
-   T‑Value Topomap (right):
    -   T‑values averaged across the same cluster window.
    -   Only cluster channels are highlighted with hollow markers; peak sensor is annotated. Scalp isocontours may be hidden for clarity.
    -   ROI-restricted analyses automatically adapt the topomap to show only tested channels; title includes "(ROI-restricted)" notation.
-   Source Surface (six views):
    -   Summarizes significant source clusters at an informative time around the global peak using signed dSPM with `pick_ori='normal'`.
    -   ROI‑restricted runs plot on the ROI vertex set; a single unified colorbar sits below the grid.
-   Statistical Reports:
    -   Text reports (`..._report.txt`) document all analysis parameters, including ROI restrictions when used.
    -   ROI-restricted analyses include dedicated "ROI Restriction" section listing channel groups, specific channels tested, and total vs. tested channel counts.

## Key pipeline updates and defaults

-   Signed source estimates: inverse application uses `pick_ori='normal'` so source time courses keep polarity, avoiding magnitude inflation.
-   Preprocessing: baseline correction applied from YAML; average EEG reference is enforced as a projector for robustness; the previous 1 Hz FIR high‑pass on short epochs is not used.
-   Inverse solution parameters: `loose` and `depth` values are now configurable in each source YAML file. The on-the-fly fallback uses EEG-appropriate defaults (`depth=3.0`), and you can specify these values via CLI arguments (`--loose`, `--depth`) when pre-computing operators.
-   Time‑window discipline: source analyses (and label time‑series fallback) crop data to YAML `tmin`/`tmax`; sensor analyses evaluate the full epoch and report/plot the significant window found by clustering.
-   Tail handling and thresholds: both domains honor `tail` (−1/0/+1) in the test; cluster‑forming threshold is tail‑aware in source, while sensor currently uses a two‑sided threshold.
-   **ROI‑restricted clustering**: Both sensor and source space support optional ROI restriction to increase statistical power:
    -   **Sensor space**: Uses centralized channel group definitions from [configs/sensor_roi_definitions.yaml](configs/sensor_roi_definitions.yaml) (e.g., `N1_bilateral`, `P1_Oz`, `P3b_midline`, `posterior_visual_parietal`).
    -   **Source space**: Uses anatomical labels from FreeSurfer parcellation (e.g., `aparc` labels like `cuneus`, `lingual`, `lateraloccipital`).
    -   Plotting and reporting automatically adapt to ROI-restricted analyses, showing only tested channels/vertices and documenting the restriction.
-   Label time‑series fallback: include 1D ROI cluster results when full spatio‑temporal stats are null (config via YAML).
-   Combined report naming is neutral: `<base_name>_report.{html,pdf}`; source figure uses six views and a unified colorbar.

## Centralized Sensor ROI System

The pipeline now includes a **centralized sensor ROI system** for clean, maintainable channel group definitions:

-   **Central definitions**: All sensor ROIs defined in [configs/sensor_roi_definitions.yaml](configs/sensor_roi_definitions.yaml)
-   **Available ROIs**:
    -   `N1_bilateral` (14 channels): Bilateral occipital-temporal regions
    -   `P1_Oz` (8 channels): Occipital midline (parieto-occipital)
    -   `P3b_midline` (9 channels): Centro-parietal midline
    -   `posterior_visual_parietal` (25 unique channels): Combined N1 + P1 + P3b regions
-   **Usage**: Reference ROIs by name in sensor YAML configs (see Configuration tips section below)
-   **Benefits**:
    -   Reduces config file size by ~80% (from ~50 lines to ~10 lines per config)
    -   Single source of truth for channel groups
    -   Same channels used for both statistics and ERP plotting
    -   Easy to maintain and update
-   **Documentation**: See [SENSOR_ROI_SYSTEM_README.md](SENSOR_ROI_SYSTEM_README.md) for complete documentation, including scientific rationale, testing, and maintenance guidelines.

## Quickstart (PowerShell examples)

```powershell
# Create and activate a fresh environment (conda)
conda create -n sfn2 python=3.11 -y; conda activate sfn2
conda install -c conda-forge mne numpy scipy matplotlib pandas tqdm imageio nibabel nilearn pyvista pyvistaqt pyqt vtk pyyaml -y
# Optional (PDF export): conda install -c conda-forge weasyprint cairo pango gdk-pixbuf -y

# Activate env and run SENSOR pipeline
python -m code.run_sensor_analysis_pipeline --config configs/cardinality1_vs_cardinality2/sensor_cardinality1_vs_cardinality2.yaml --accuracy all

# Activate env and run SOURCE pipeline (single method)
python -m code.run_source_analysis_pipeline --config configs/cardinality1_vs_cardinality2/source_dspm_cardinality1_vs_cardinality2.yaml --accuracy all

# Precompute inverse operators (recommended before source analyses)
# Note: Use --depth 3.0 for an EEG-appropriate depth weighting.
python -m code.build_inverse_solutions --depth 3.0

# Run FULL pipeline (sensor + all matching source configs)
python -m code.run_full_analysis_pipeline --config configs/cardinality1_vs_cardinality2/sensor_cardinality1_vs_cardinality2.yaml

# Optional global override examples
python -m code.run_full_analysis_pipeline --config configs/change_vs_no-change/sensor_change_vs_no-change.yaml --accuracy acc1
python -m code.run_full_analysis_pipeline --config configs/change_vs_no-change/sensor_change_vs_no-change.yaml --accuracy acc0
```

## Configuration tips (YAML)

-   Stats sensitivity
    -   **p_threshold**: cluster‑forming threshold (e.g., 0.001–0.05). Larger is more sensitive to forming clusters, but raises FWER bar.
    -   **cluster_alpha**: cluster‑level alpha (typically 0.05).
    -   **tail**: use `1` or `-1` for one‑tailed tests if direction is known; `0` for two‑tailed.
    -   **n_permutations**: increase for stable p‑values (e.g., 5000–10000).
-   Time window: set `tmin`/`tmax` to bracket the expected effect (often informed by sensors).
-   **ROI restriction (sensor-space, optional):**
    ```yaml
    stats:
      roi:
        channel_groups:
          - N1_bilateral         # 14 channels over bilateral occipital regions
          - P1_Oz                # 8 channels around Oz
          - P3b_midline          # 9 channels over centro-parietal midline
        # Or use the combined posterior ROI:
        # channel_groups:
        #   - posterior_visual_parietal  # All N1 + P1 + P3b regions (25 unique channels)
    ```
    **Note:** Sensor ROI definitions are centrally managed in [configs/sensor_roi_definitions.yaml](configs/sensor_roi_definitions.yaml). See [SENSOR_ROI_SYSTEM_README.md](SENSOR_ROI_SYSTEM_README.md) for complete documentation.
-   **ROI restriction (source-space, optional):**
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

- `configs/common.yaml` provides minimal, centralized defaults to improve reproducibility.
- Current defaults:
  - `data.preprocessing: hpf_1.0_lpf_35_baseline-on`
  - `epoch_defaults.baseline: [-0.2, 0.0]`
- Behavior:
  - When loading an analysis config, the loader fills in a missing `epoch_window.baseline` from `epoch_defaults.baseline` (does not overwrite if present).
  - Subject discovery uses the preprocessing variant from `common.yaml` to locate combined data under `data/data_preprocessed/<preprocessing>`.

Example `configs/common.yaml`:

```
data:
  preprocessing: "hpf_1.0_lpf_35_baseline-on"

epoch_defaults:
  baseline: [-0.2, 0.0]

montage: "GSN-HydroCel-129"
reference: "average"
```

Testing notes (TDD): lightweight unit tests validate that defaults merge correctly and path resolution behaves as expected.


python -m code.run_full_analysis_pipeline --config configs/23_32_22_33/sensor_23_32_22_33.yaml --accuracy acc1
>>




intraparietal sulcus 

