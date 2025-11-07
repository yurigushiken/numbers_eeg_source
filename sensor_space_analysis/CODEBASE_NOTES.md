Codebase Notes for Numbers EEG Source (for Brain–Behavior Correlation Project)

Summary
- Combined preprocessed epochs (baseline −0.2 to 0.0 s; avg reference projector present) are stored under:
  - `data/data_preprocessed/hpf_1.5_lpf_35_baseline-on` as `sub-XX_preprocessed-epo.fif`.
- Sampling rate: 250 Hz. Epoch time range: −0.200 s to 0.496 s (175 samples; 4 ms resolution).
- Channels: EGI HydroCel GSN-129 layout; channel names use `E1`..`E128` (non‑scalp/belt channels exist and are often removed for topomaps). Average reference is applied (as a projection) when needed.
- Metadata (per-trial) is embedded in `Epochs.metadata` with at least these columns:
  - `SubjectID` (e.g., "02")
  - `Block`, `Trial`, `Procedure`
  - `Condition` (prime→oddball code as int, e.g., 23 for 2→3; cardinalities are 11, 22, 33, 44, 55, 66)
  - `Target.ACC` (float: ≥0.5 means correct, <0.5 incorrect)
  - `Target.RT` (float ms; can contain NaN when no response was expected, e.g., cardinality trials)
  - Convenience factors: `direction` (I/D), `change_group` (e.g., dLS), `size` (small/large/cross)

Relevant Modules
- `code/utils/data_loader.py`
  - Locates combined preprocessed files and loads `mne.Epochs` with metadata (`load_subject_combined_epochs`).
  - Retrieves per-condition epoch subsets using metadata filters (`get_evoked_for_condition`).
  - Applies baseline and average reference; enforces a standard EGI-128 montage from `assets/`.
  - Accuracy filtering: uses `Target.ACC` with `acc1` (≥0.5) vs `acc0` (<0.5) if requested.
- `code/run_sensor_analysis_pipeline.py`
  - Orchestrates sensor-space contrasts across subjects, then cluster-based permutation tests.
  - Saves grand-average evoked and statistical reports/figures to `derivatives/sensor/...`.
- `code/utils/cluster_stats.py`
  - Builds channel adjacency (MNE adjacency or distance-based) and optional ROI restriction.
  - Runs `mne.stats.spatio_temporal_cluster_1samp_test` on stacked evoked contrasts.

Configs Referenced
- Example Acc0 vs Acc1 (sensor) config lives under `configs/Acc0_vs_Acc1/`.
  - Defines the contrast between condition sets at the sensor level.
  - Not strictly required for this trial-level extraction, but mirrors how groups were compared.

Key Answers for the Consultant Brief
- Data location: `data/data_preprocessed/hpf_1.5_lpf_35_baseline-on/sub-XX_preprocessed-epo.fif`.
- Metadata structure: Columns include `Condition`, `Target.ACC`, `Target.RT` (ms). Cardinality trials have valid metadata but often NaN RT since no button press is expected.
- Electrode naming: EEG channels named `E1`..`E128` (EGI HydroCel GSN-129). Left anterior–temporal Cluster #2 channels are: `E32, E33, E34, E38, E39, E43, E44, E48, E49, E56, E57, E127, E128`.
- Time sampling: 250 Hz; epoch `times` from −0.200 to 0.496 s (step ≈ 0.004 s). Windows are indexed by comparing `epochs.times` to `[tmin, tmax]`.
- No‑change trials: Cardinalities are represented by `Condition` in {11, 22, 33, 44, 55, 66}. These must be filtered out before RT analyses (no button press expected).

