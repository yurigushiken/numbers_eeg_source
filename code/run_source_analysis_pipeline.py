"""
Source-Space Analysis Pipeline
"""
import argparse
import logging
import os
from pathlib import Path
import mne
import numpy as np
from tqdm import tqdm
from datetime import datetime

from code.utils import data_loader, cluster_stats, plotting, reporter
from code.utils.anatomical_reporter import generate_anatomical_report, generate_hs_summary_table

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()


def main(config_path=None, accuracy=None):
    if config_path is None:
        parser = argparse.ArgumentParser(description="Run Source-Space Analysis Pipeline")
        parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file.")
        parser.add_argument("--accuracy", type=str, required=False, default=None, choices=['all', 'acc1', 'acc0'],
                            help=("Optional override for trial filtering. Use 'all', 'acc1' (correct only), or 'acc0' (incorrect only). "
                                  "If omitted, per-condition YAML 'accuracy' fields are used (fallback 'all')."))
        # Data source flag removed; combined preprocessed data is the standard
        args = parser.parse_args()
        config_path = args.config
        accuracy = args.accuracy
    

    # --- 1. Load Config and Setup ---
    config = data_loader.load_config(config_path)
    analysis_name = config['analysis_name']

    # Generate a timestamped name for the analysis folder and report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_analysis_name = f"{timestamp}-{analysis_name}"

    # Allow custom derivatives root via environment variable (default: "derivatives")
    derivatives_root = os.environ.get("DERIVATIVES_ROOT", "derivatives")
    output_dir = Path(derivatives_root) / "source" / timestamped_analysis_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory created at: {output_dir}")

    # --- 2. Load Data and Compute Source Contrasts ---
    subject_dirs = data_loader.get_subject_dirs(accuracy)
    fsaverage_src = data_loader.get_fsaverage_src(config)

    all_source_contrasts = []
    provenance = []  # Track inverse operator provenance per subject
    project_root = Path(__file__).resolve().parents[1]
    log.info("Processing subjects for source analysis...")
    for subject_dir in tqdm(subject_dirs, desc="Processing subjects (source)"):
        contrast_evoked, _ = data_loader.create_subject_contrast(
            subject_dir, config, accuracy=accuracy
        )
        if contrast_evoked is None:
            continue

        try:
            inv_operator = data_loader.get_inverse_operator(subject_dir)
            used = "precomputed"
            inv_path = subject_dir / f"{subject_dir.name}-inv.fif"
        except FileNotFoundError:
            log.warning(f"Inverse operator not found for {subject_dir.name}. "
                        f"Generating a template inverse on-the-fly.")
            # We need the concatenated epochs for this subject to compute covariance.
            # Note: This re-loads data, but it's an exceptional case.
            _, all_epochs = data_loader.create_subject_contrast(
                subject_dir, config, accuracy=accuracy
            )
            if all_epochs is None:
                log.error(f"Could not load epochs for {subject_dir.name} to generate "
                          f"template inverse. Skipping.")
                continue
            # Now call the fallback function, passing the config
            inv_operator = data_loader.generate_template_inverse_operator_from_epochs(
                all_epochs, subject_dir, config
            )
            used = "template"
            inv_path = subject_dir / f"{subject_dir.name}-inv.fif"

        # Record provenance for this subject
        try:
            inv_path_abs = inv_path.resolve()
            path_rel = inv_path_abs
            try:
                path_rel = inv_path_abs.relative_to(project_root.resolve())
            except Exception:
                from os import path as _path
                path_rel = _path.relpath(str(inv_path_abs), start=str(project_root.resolve()))
            stat = inv_path_abs.stat()
            from datetime import datetime as _dt
            provenance.append({
                "subject": subject_dir.name,
                "used": used,
                "path": str(path_rel),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": _dt.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            })
        except Exception:
            pass

        stc = data_loader.compute_subject_source_contrast(
            contrast_evoked, inv_operator, config
        )
        all_source_contrasts.append(stc)

    if not all_source_contrasts:
        log.error("No valid source data found for any subject. Cannot proceed.")
        return
    log.info(f"Successfully created source contrasts for {len(all_source_contrasts)} subjects.")

    # --- 3. Use full temporal resolution for cluster statistics ---
    # Keep the original sampling frequency to increase the number of time samples in the tested window.
    sfreq_orig = all_source_contrasts[0].sfreq
    log.info(f"Using original sampling frequency ({sfreq_orig} Hz) for source cluster statistics.")
    all_source_contrasts_for_stats = all_source_contrasts


    # --- 4. Compute Grand Average Source Estimate (from full-resolution data) ---
    log.info("Computing grand average source estimate from full-resolution data...")
    # Sum all STCs and divide by N for the grand average
    stc_grand_average = np.sum(all_source_contrasts) / len(all_source_contrasts)
    ga_fname = output_dir / f"{analysis_name}_grand_average-stc.h5"
    stc_grand_average.save(ga_fname, overwrite=True)

    # Create a copy for reporting/plotting timebase alignment (no resampling)
    stc_ga_for_reporting = stc_grand_average.copy()
    # Crop the reporting STC to stats.analysis_window to match clustering
    aw = (config.get('stats') or {}).get('analysis_window')
    if aw and len(aw) == 2:
        rep_tmin, rep_tmax = float(aw[0]), float(aw[1])
        try:
            stc_ga_for_reporting.crop(tmin=rep_tmin, tmax=rep_tmax)
            log.info(f"Reporting STC cropped to {rep_tmin:.3f}s–{rep_tmax:.3f}s")
        except Exception:
            log.warning("Failed to crop reporting STC; proceeding without cropping.")
    else:
        log.info("No analysis_window provided for reporting STC; using full range.")

    # --- 5. Run Group-Level Cluster Statistics (on full-resolution data) ---
    stats_cfg = config.get('stats') or {}
    vertex_cfg = (stats_cfg.get('vertex') or {})
    vertex_enabled = bool(vertex_cfg.get('enabled', True))

    lt_cfg = stats_cfg.get('label_timeseries')
    lt_enabled = False
    lt_mode = 'disabled'
    if lt_cfg is not None:
        lt_enabled = bool(lt_cfg.get('enabled', True))
        lt_mode = str(lt_cfg.get('mode', 'fallback')).lower()
        if lt_mode not in ('fallback', 'always', 'only'):
            lt_mode = 'fallback'
    if lt_enabled and lt_mode == 'only':
        if vertex_enabled:
            log.info("Label time-series mode set to 'only'; skipping vertex-level clustering.")
        vertex_enabled = False

    stats_results = None
    clusters = []
    cluster_p_values = np.empty((0,))
    vertex_has_significant_cluster = False

    if vertex_enabled:
        stats_results = cluster_stats.run_source_cluster_test(all_source_contrasts_for_stats, fsaverage_src, config)
        try:
            _, clusters, cluster_p_values, _ = stats_results
            vertex_has_significant_cluster = bool((cluster_p_values < stats_cfg['cluster_alpha']).any())
        except Exception as exc:
            log.warning(f"Failed to evaluate vertex-level significance: {exc}")
    else:
        log.info("Vertex-level clustering is disabled for this analysis.")
        stats_results = (np.empty((0, 0)), [], np.empty((0,)), None)

    # Decide whether to run label time-series clustering
    lt_results = None
    aux_dir = output_dir / "aux"
    summary_path = aux_dir / "label_cluster_summary.txt"
    aux_artifacts = [
        aux_dir / "label_times.npy",
        aux_dir / "label_mean_ts.npy",
        aux_dir / "labels.txt",
        summary_path,
    ]

    if lt_cfg is not None and lt_enabled:
        run_label_ts = False
        if lt_mode in ('always', 'only'):
            run_label_ts = True
        elif lt_mode == 'fallback':
            run_label_ts = (not vertex_enabled) or (not vertex_has_significant_cluster)
        if run_label_ts:
            log.info(f"Running label time-series clustering (mode={lt_mode})...")
            label_results, lt_times, label_mean_ts, label_names = cluster_stats.run_label_timecourse_cluster_test(
                all_source_contrasts_for_stats, fsaverage_src, config
            )
            aux_dir = output_dir / "aux"
            aux_dir.mkdir(exist_ok=True)
            np.save(aux_dir / "label_times.npy", lt_times)
            np.save(aux_dir / "label_mean_ts.npy", label_mean_ts)
            (aux_dir / "labels.txt").write_text("\n".join(label_names))

            resolved = (stats_cfg.get('label_timeseries') or {}).get('_resolved_params', {})
            lt_alpha = float(resolved.get('cluster_alpha', stats_cfg.get('cluster_alpha')))
            lt_p_threshold = float(resolved.get('p_threshold', stats_cfg.get('p_threshold')))
            lt_tail = int(resolved.get('tail', stats_cfg.get('tail', 0)))
            lt_n_perm = int(resolved.get('n_permutations', stats_cfg.get('n_permutations')))

            try:
                lines = [
                    f"mode={lt_mode}; n_subjects={len(all_source_contrasts_for_stats)}; n_labels={len(label_names)}; "
                    f"window={lt_times[0]:.3f}-{lt_times[-1]:.3f}s; tail={lt_tail}; "
                    f"threshold from p_threshold={lt_p_threshold}; n_permutations={lt_n_perm}"
                ]
                sig_any = False
                for label_name, stats in zip(label_names, label_results):
                    t_obs, clusters, cluster_p_values, _ = stats
                    if len(cluster_p_values) == 0:
                        lines.append(f"{label_name}: no clusters formed (check threshold).")
                        continue
                    label_sig = False
                    for idx, pval in enumerate(cluster_p_values):
                        if pval < lt_alpha:
                            label_sig = True
                            sig_any = True
                            mask = clusters[idx]
                            t_inds = np.where(mask)[0]
                            if t_inds.size == 0:
                                continue
                            tmin_ms = float(lt_times[t_inds.min()] * 1000.0)
                            tmax_ms = float(lt_times[t_inds.max()] * 1000.0)
                            vals = t_obs[t_inds]
                            peak_i = int(np.abs(vals).argmax())
                            peak_ms = float(lt_times[t_inds[peak_i]] * 1000.0)
                            lines.append(
                                f"{label_name}: SIGNIFICANT cluster p={pval:.4f} at {tmin_ms:.1f}-{tmax_ms:.1f} ms (peak {peak_ms:.1f} ms)"
                            )
                    if not label_sig:
                        lines.append(f"{label_name}: no significant clusters at cluster_alpha={lt_alpha:.3f}")
                if not sig_any:
                    lines.append("No label time-series clusters reached significance.")
                (aux_dir / "label_cluster_summary.txt").write_text("\n".join(lines))
            except Exception as e:
                log.warning(f"Failed to write label time-series summary: {e}")
        else:
            log.info(f"Skipping label time-series analysis (mode={lt_mode}) because vertex clustering already produced significant results.")
            for artifact in aux_artifacts:
                if artifact.exists():
                    try:
                        artifact.unlink()
                    except Exception:
                        pass
    else:
        for artifact in aux_artifacts:
            if artifact.exists():
                try:
                    artifact.unlink()
                except Exception:
                    pass

    # Persist provenance JSON next to outputs
    try:
        import json as _json
        (output_dir / "inverse_provenance.json").write_text(_json.dumps(provenance, indent=2))
    except Exception:
        pass

    # --- 6. Generate Report and Visualizations ---
    log.info("Generating source report and plots...")
    reporter.generate_source_report(stats_results, stc_ga_for_reporting, config, output_dir, len(all_source_contrasts_for_stats))
    if vertex_enabled:
        plotting.plot_source_clusters(stats_results, stc_ga_for_reporting, config, output_dir)
    else:
        log.info("Skipping vertex-level plotting because vertex clustering was disabled.")

    # --- 7. Generate and Save Anatomical Report ---
    # This report provides detailed anatomical labels for any significant clusters.
    log.info("Generating detailed anatomical report for significant clusters...")
    try:
        project_root = Path(__file__).resolve().parents[1]
        candidates = [
            project_root / 'data' / 'fs_subjects_dir',
            project_root / 'data' / 'all' / 'fs_subjects_dir',
        ]
        subjects_dir = next((cand for cand in candidates if cand.exists()), candidates[0])
        if subjects_dir:
            anatomical_df = generate_anatomical_report(
                stats_results, fsaverage_src, "fsaverage", str(subjects_dir), config
            )
            # Prepare output paths and proactively remove stale files
            report_fname = output_dir / f"{analysis_name}_anatomical_report.csv"
            hs_fname = output_dir / f"{analysis_name}_anatomical_summary_hs.csv"
            try:
                if report_fname.exists():
                    report_fname.unlink()
            except Exception:
                pass
            try:
                if hs_fname.exists():
                    hs_fname.unlink()
            except Exception:
                pass

            if not anatomical_df.empty:
                anatomical_df.to_csv(report_fname, index=False)
                log.info(f"Anatomical report saved to: {report_fname}")
            # Cortical Cluster Localization per-cluster summary
            hs_df = generate_hs_summary_table(
                stats_results, fsaverage_src, "fsaverage", str(subjects_dir), config
            )
            if not hs_df.empty:
                hs_df.to_csv(hs_fname, index=False)
                log.info(f"Cortical Cluster Localization summary saved to: {hs_fname}")
        else:
            log.warning("FS subjects_dir not found under data; skipping anatomical report.")
    except Exception as e:
        log.error(f"An error occurred during anatomical report generation: {e}")


    log.info("-" * 80)
    log.info(f"Source pipeline finished successfully for '{analysis_name}'.")
    log.info(f"All outputs are saved in: {output_dir}")
    log.info("-" * 80)
    return output_dir


if __name__ == "__main__":
    main()

