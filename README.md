- Label summaries live in `derivatives/source/<timestamp>-*/aux/label_cluster_summary.txt`; PI-facing answer comes from that file.

## Source / ROI Naming (13_31)

- Whole-brain discovery: `source_eloreta_wholebrain_magnitude_90_300.yaml`.
- Data-driven ROI (2-label annotation): `source_eloreta_corelabel_c13_31_120_280.yaml`, ROI files `c13_31_core-{lh,rh}.label`, annotation `c13_31`.
- Atlas follow-up: `source_eloreta_aparc_c13_31_180_300.yaml`.
- Alternative solver (same contrast): `source_dspm_wholebrain_signed_90_300.yaml`.
- Future contrasts (e.g., `c11_22`) should mirror this: create a `c11_22` annotation with `c11_22_core-{lh,rh}.label` and a corresponding `source_eloreta_corelabel_c11_22_<tmin>_<tmax>.yaml`.
# Numbers EEG Source Pipeline (2025 status)

This repository hosts a reproducible MNE-Python workflow for EEG source analysis using cluster-based permutation statistics. The notes below document exactly what the current code and configs do so that the PI, future RAs, and collaborators can answer “what did we run?” without digging into scripts.

## Subjects & Preprocessed Data

- **N = 24 subjects** in the current cohort.
- Preprocessed epochs (already baseline-corrected −0.2 s to 0.0 s; average reference projector applied) live at:
  ```
  data/data_preprocessed/hpf_1.5_lpf_35_baseline-on/sub-XX/sub-XX_preprocessed-epo.fif
  ```
- `code/utils/data_loader.py` reads those combined epoch files, filters conditions via metadata, applies baseline if needed, and returns per-condition evokeds before forming contrasts.

## Anatomy / fsaverage / Inverse Operators

- **SUBJECTS_DIR:** `D:/numbers_eeg_source/data/fs_subjects_dir` (fsaverage only; no individual MRIs).
- All subject STCs are morphed to this fsaverage space (see `compute_subject_source_contrast`).
- YAML’s `inverse.source_spacing: ico-4` → pipeline loads/generates `fsaverage-ico-4-src.fif` and builds adjacency from it.
- All current inverses were computed with MNE’s make_inverse_operator using eLORETA, depth=0.8, loose=0.2, pick_ori=normal on fsaverage (ico-4). Any on-the-fly inverse uses the same parameters.
- Subject inverse operators are precomputed and reused every run:
  ```
  data/data_preprocessed/hpf_1.5_lpf_35_baseline-on/sub-XX/sub-XX-inv.fif
  ```
  (generated earlier via `python -m code.build_inverse_solutions`).

## Inverse / Source Parameters (contrasts)

Current whole-brain discovery YAML (`configs/13_31/source_eloreta_wholebrain_magnitude_90_300.yaml`):
- `method: eLORETA`
- `snr: 2.0`
- `pick_ori: normal` (polarity-preserving)
- `inverse.loose: 0.2`, `inverse.depth: 0.8`, `inverse.source_spacing: ico-4`
- We apply the inverse to the **contrast evoked** (A – B), not to each condition separately. Morph happens **once per subject after forming the contrast**.

Contrast definition:
- Name: “1↔3 Transition: Increasing vs. Decreasing”
- Condition A: `TRANSITION_13`
- Condition B: `TRANSITION_31`
- Weights: `[1, -1]`
- Contrasts are built **after** baseline/reference → we never mix pre-/post-baseline data in stats.

## Polarity & Post-process Policy

- `postprocess.align_sign: false` → no whole-brain sign flip (avoids flattening dipoles).
- `postprocess.make_magnitude: false` in the current “dipole” run, but the code supports `true` for orientation-invariant vertex stats (report logs “Orientation-invariant … ” when enabled and sign becomes uninterpretable).
- Label time-series always operate on signed STCs with `mode="mean_flip"`, preventing intra-label cancellation while preserving direction.

## Vertex vs. Label Branches (how they split)

Vertex and label branches are independent. From `code/run_source_analysis_pipeline.py`:
```python
if vertex_enabled:
    stats_results = cluster_stats.run_source_cluster_test(
        all_source_contrasts_for_stats,
        fsaverage_src,
        config,
        make_magnitude=make_magnitude,
    )
else:
    log.info("Vertex-level clustering is disabled for this analysis.")
    stats_results = (np.empty((0, 0)), [], np.empty((0,)), None)

if lt_cfg is not None and lt_enabled:
    label_results, lt_times, label_mean_ts, label_names = (
        cluster_stats.run_label_timecourse_cluster_test(
            label_source_contrasts, fsaverage_src, config
        )
    )
    aux_dir.mkdir(parents=True, exist_ok=True)
    np.save(aux_dir / "label_times.npy", lt_times)
    np.save(aux_dir / "label_mean_ts.npy", label_mean_ts)
    (aux_dir / "labels.txt").write_text("\n".join(label_names))
    (aux_dir / "label_cluster_summary.txt").write_text("\n".join(lines))
```
- Vertex enabled → stack `(n_subjects, n_times, n_vertices)`, crop to `stats.analysis_window`, optionally `np.abs(...)`, build adjacency from the fsaverage ico-4 source, run `mne.stats.spatio_temporal_cluster_1samp_test` with YAML thresholds and `tail`.
- Vertex disabled → we **skip adjacency entirely**.
- Label enabled → extract signed label time courses (mean-flip), crop to the **same window**, run a 1D permutation per label, and write results to `aux/`.

## Label Restrict Behaviour

Label extraction logs available aparc names and matches `restrict_to` case/hemisphere-insensitively. From `code/utils/cluster_stats.py`:
```python
available_names = sorted({lab.name for lab in all_labels})
log.debug(f"Loaded {len(available_names)} labels from {parc} (fsaverage): {available_names}")
...
selected_labels = [lab for lab in all_labels if _base_label(lab.name) in wanted_set]
missing = sorted(wanted_set - { _base_label(l.name) for l in selected_labels })
if missing:
    log.warning("Restrict-to labels not found in %s annot: %s", parc, ", ".join(missing))
if not selected_labels:
    raise ValueError("Selected labels not found on fsaverage. Check parc and label names.")
```
- Requested labels missing from fsaverage → warning.
- All requested labels missing → hard error (run stops).
- When some match, the label permutation still runs for the ones found.
- Cropping: `X_c = X[:, :, time_mask]` uses the **same** `analysis_window` as vertex.
- Label p-values are **uncorrected** across labels; summary reminds you (“Label-wise … uncorrected”).

## Outputs (per run)

Example directory:
```
derivatives/
  source/
    20251101_000946-source_eloreta_wholebrain_magnitude_90_300/
      source_eloreta_wholebrain_magnitude_90_300_report.txt
      inverse_provenance.json
      source_eloreta_wholebrain_magnitude_90_300_grand_average-stc.h5-lh.stc
      source_eloreta_wholebrain_magnitude_90_300_grand_average-stc.h5-rh.stc
      aux/
        label_cluster_summary.txt
        label_times.npy
        label_mean_ts.npy
        labels.txt
```
- `aux/` is always created when label stats run, independent of STC/window-average toggles.
- Grand-average STCs are **always** written for QC/visualization even when vertex is disabled.

## Visualization

- To export synchronized source/sensor movies, set `visualization.movie.enabled: true` in the source YAML. The pipeline will write `movie_source.mp4`, `movie_topo.mp4`, and (when ffmpeg is available) `movie_source_topo.mp4` into the source output directory using the requested window, codec, and frame count.
- Movies are generated with standard MNE workflows: `SourceEstimate.plot(...).save_movie(...)` as in “Visualize source time courses (stcs)” and `evoked.animate_topomap(...)` from “Plotting topographic maps of evoked data”.

## Current PI Run (Nov 2025)

- Whole-brain discovery (orientation-invariant): `configs/13_31/source_eloreta_wholebrain_magnitude_90_300.yaml`.
- Data-driven core ROI (c13_31 annotation): `configs/13_31/source_eloreta_corelabel_c13_31_120_280.yaml` → tests `c13_31_core-{lh,rh}` between 120–280 ms (vertex disabled).
- Atlas follow-up (expanded medial/parietal set): `configs/13_31/source_eloreta_aparc_c13_31_180_300.yaml`.
- Solver cross-check: `configs/13_31/source_dspm_wholebrain_signed_90_300.yaml`.
- All runs use the signed contrast, no global sign alignment, and 5000 permutations (tail=0).
- Label summaries live in `derivatives/source/<timestamp>-*/aux/label_cluster_summary.txt`; PI-facing answer comes from that file.

## How to Run This Analysis

On Windows PowerShell / Cmd (with conda on PATH):
```powershell
conda activate numbers_eeg_source
python -m code.run_source_analysis_pipeline `
    --config configs\13_31\source_eloreta_wholebrain_magnitude_90_300.yaml
```
- The pipeline logs the 24 inverses it loads, the labels being tested, and whether any clusters survive.
- Watch the log for messages such as “Restrict-to labels not found …” or errors inside the label branch—those are the only reasons aux files would be missing.

## Key Implementation Notes

- Subject loop (
`compute_subject_source_contrast`
) morphs once per subject, returns two independent STCs; label flipping never modifies vertex data.
- Vertex stats call `run_source_cluster_test(stcs, fsaverage_src, config, make_magnitude=False)`; label stats call `run_label_timecourse_cluster_test` which now enforces `restrict_to` gracefully.
- `postprocess.save_subject_stc` / `dump_window_average` default to false; they are true QC toggles only (they no longer affect label outputs).
- Reports include a “Polarity Handling” section plus dipole context:
  > “Vertex-wise analyses run without a global sign flip; label time-series handle polarity locally via mode="mean_flip". … Label-level tests were restricted to caudalanteriorcingulate/rostralanteriorcingulate/superiorparietal/precuneus … to probe the anterior–posterior dipole suggested by sensor-space clusters.”
- When magnitude mode is enabled for a different experiment, the report explicitly states the vertex branch is orientation-invariant and that sign is not interpretable.

## Keeping This Doc Accurate

- If you change the YAML defaults (e.g., add other parcels, enable vertex stats, turn on `make_magnitude`, switch depth), update both the README and the YAML comments.
- Whenever aux files or report sections change, verify that the README still matches the folder layout.
- For new cohorts, update subject count and data paths here first.

With this README, anyone reviewing the project can reconstruct the pipeline: where data sit, how contrasts are formed, how vertex vs. label stats diverge, and what files to inspect for signed ROI answers. Keep it synced with the configs and log outputs as the project evolves. 




