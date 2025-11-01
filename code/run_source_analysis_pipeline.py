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

from code.utils import data_loader, cluster_stats, plotting, reporter, movie_utils
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
    label_source_contrasts = []
    provenance = []  # Track inverse operator provenance per subject
    project_root = Path(__file__).resolve().parents[1]
    log.info("Processing subjects for source analysis...")
    stats_cfg = config.get('stats') or {}
    source_cfg = config.get('source') or {}
    pick_ori_cfg = str(source_cfg.get('pick_ori', 'normal') or 'normal').lower()

    postproc_cfg = (config.get('postprocess') or {})
    hs_normalize = bool(postproc_cfg.get('normalize_log_total_current', False))
    hs_power = float(postproc_cfg.get('normalization_power', 1.0))
    hs_take_abs = bool(postproc_cfg.get('take_abs', False))
    align_sign = bool(postproc_cfg.get('align_sign', False)) and pick_ori_cfg == 'normal'
    save_subject_stc = bool(postproc_cfg.get('save_subject_stc', False))
    dump_window_avg = bool(postproc_cfg.get('dump_window_average', False))
    make_magnitude = bool(postproc_cfg.get('make_magnitude', False))
    if postproc_cfg.get('align_sign', False):
        # Project policy: global sign alignment is disabled for vertex-wise stats.
        # Sign handling should occur per-label via mode='mean_flip'.
        if pick_ori_cfg != 'normal':
            log.warning("postprocess.align_sign requested but pick_ori is not 'normal'; ignoring.")
        else:
            log.info("Global sign alignment is disabled for vertex-wise clustering; sign will be handled per-label via mode='mean_flip'.")
    if hs_normalize:
        config['_hs_normalized'] = True
        config['_hs_normalization_power'] = hs_power
        if hs_take_abs:
            config['_hs_take_abs'] = True

    lt_cfg = stats_cfg.get('label_timeseries')
    lt_enabled = False
    if lt_cfg is not None:
        lt_enabled = bool(lt_cfg.get('enabled', True))
    # --- Branch switches (YAML is the single source of truth) ---
    vertex_cfg = (stats_cfg.get('vertex') or {})
    vertex_enabled = bool(vertex_cfg.get('enabled', False))
    do_label = bool((stats_cfg.get('label_timeseries') or {}).get('enabled', False))

    # --- Sanity logs at startup ---
    aw_log = (stats_cfg.get('analysis_window') or [None, None])
    try:
        aw_log = (float(aw_log[0]), float(aw_log[1])) if len(aw_log) == 2 else (None, None)
    except Exception:
        aw_log = (None, None)
    log.info(
        f"Branches → vertex: {vertex_enabled}, label_ts: {do_label}; "
        f"analysis_window: {aw_log[0]}–{aw_log[1]}; align_sign: {align_sign}; "
        f"orientation_invariant: {make_magnitude}; save_subject_stc: {save_subject_stc}; "
        f"dump_window_average: {dump_window_avg}"
    )

    flipped_subjects = []
    aux_base_dir = output_dir / "aux"
    need_aux_dir = bool(save_subject_stc or dump_window_avg or (lt_cfg is not None and lt_enabled))
    if need_aux_dir:
        aux_base_dir.mkdir(parents=True, exist_ok=True)

    stc_save_dir = aux_base_dir / "subject_stcs" if save_subject_stc else None
    if stc_save_dir is not None:
        stc_save_dir.mkdir(parents=True, exist_ok=True)

    window_avg_dir = aux_base_dir / "window_averages" if dump_window_avg else None
    if window_avg_dir is not None:
        window_avg_dir.mkdir(parents=True, exist_ok=True)

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

        stc_vertex, stc_label = data_loader.compute_subject_source_contrast(
            contrast_evoked, inv_operator, config
        )

        if hs_normalize:
            aw = stats_cfg.get('analysis_window') or [stc_vertex.tmin, stc_vertex.times[-1]]
            window = (float(aw[0]), float(aw[1])) if len(aw) == 2 else (stc_vertex.tmin, stc_vertex.times[-1])
            stc_vertex = data_loader.normalize_and_log_stc(
                stc_vertex,
                window=window,
                power=hs_power,
                take_abs=hs_take_abs,
            )
            if lt_enabled:
                stc_label = data_loader.normalize_and_log_stc(
                    stc_label,
                    window=window,
                    power=hs_power,
                    take_abs=hs_take_abs,
                )

        # Note: global sign flipping removed per project policy

        stc_save_base = None
        if save_subject_stc and stc_save_dir is not None:
            stc_save_base = stc_save_dir / f"{subject_dir.name}_contrast"
            lh_path = Path(f"{stc_save_base}-lh.stc")
            if lh_path.exists():
                log.info(f"Skipping subject STC write for {subject_dir.name} (save_subject_stc=true, already exists).")
            else:
                log.info(f"Writing subject STC for {subject_dir.name} (save_subject_stc=true)...")
                try:
                    stc_vertex.save(str(stc_save_base), overwrite=False)
                except FileExistsError:
                    log.info(f"Subject STC for {subject_dir.name} already present; keeping existing file.")
                except Exception as exc:
                    log.warning(f"Failed to save subject STC for {subject_dir.name}: {exc}")
        else:
            log.debug(f"Skipping subject STC write for {subject_dir.name} (save_subject_stc=false).")

        if dump_window_avg and window_avg_dir is not None:
            aw_avg = stats_cfg.get('analysis_window')
            if aw_avg and len(aw_avg) == 2:
                try:
                    tmin_avg = float(aw_avg[0])
                    tmax_avg = float(aw_avg[1])
                except Exception:
                    tmin_avg = float(stc_vertex.tmin)
                    tmax_avg = float(stc_vertex.times[-1])
            else:
                tmin_avg = float(stc_vertex.tmin)
                tmax_avg = float(stc_vertex.times[-1])
            try:
                avg_base = window_avg_dir / f"{subject_dir.name}_window_mean"
                lh_avg = Path(f"{avg_base}-lh.stc")
                if lh_avg.exists():
                    log.info(f"Skipping window-average write for {subject_dir.name} (dump_window_average=true, already exists).")
                else:
                    stc_window = stc_vertex.copy().crop(tmin=tmin_avg, tmax=tmax_avg)
                    if stc_window.data.size:
                        data_mean = stc_window.data.mean(axis=1, keepdims=True)
                        mid_time = 0.5 * (tmin_avg + tmax_avg)
                        stc_avg = mne.SourceEstimate(
                            data_mean,
                            vertices=stc_vertex.vertices,
                            tmin=mid_time,
                            tstep=1.0,
                            subject=stc_vertex.subject,
                        )
                        log.info(f"Writing window-average STC for {subject_dir.name} (dump_window_average=true)...")
                        stc_avg.save(str(avg_base), overwrite=False)
            except FileExistsError:
                log.info(f"Window-average STC for {subject_dir.name} already present; keeping existing file.")
            except Exception as exc:
                log.warning(f"Failed to save window-averaged STC for {subject_dir.name}: {exc}")
        else:
            log.debug(f"Skipping window-average write for {subject_dir.name} (dump_window_average=false).")
        all_source_contrasts.append(stc_vertex)
        label_source_contrasts.append(stc_label)

    if not all_source_contrasts:
        log.error("No valid source data found for any subject. Cannot proceed.")
        return
    log.info(f"Successfully created source contrasts for {len(all_source_contrasts)} subjects.")
    # No global sign alignment summary; flipping is disabled

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

    # Honour vertex-level ROI restriction requests that were previously ignored.
    if vertex_enabled:
        restrict_cfg = {}
        if bool(vertex_cfg.get('restrict_to_roi')):
            parc = vertex_cfg.get('parc')
            labels = vertex_cfg.get('labels') or []
            if parc:
                restrict_cfg['parc'] = parc
            if labels:
                restrict_cfg['labels'] = labels
        # Only update stats.roi if the caller provided something explicit.
        if restrict_cfg:
            stats_cfg['roi'] = restrict_cfg

    stats_results = None
    clusters = []
    cluster_p_values = np.empty((0,))
    vertex_has_significant_cluster = False

    if vertex_enabled:
        if make_magnitude:
            log.info("Orientation-invariant source stats: testing current magnitude (sign discarded).")
        stats_results = cluster_stats.run_source_cluster_test(
            all_source_contrasts_for_stats,
            fsaverage_src,
            config,
            make_magnitude=make_magnitude,
        )
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
    aux_dir = aux_base_dir if need_aux_dir else output_dir / "aux"
    summary_path = aux_dir / "label_cluster_summary.txt"
    aux_artifacts = [
        aux_dir / "label_times.npy",
        aux_dir / "label_mean_ts.npy",
        aux_dir / "labels.txt",
        summary_path,
    ]

    if lt_cfg is not None and lt_enabled:
        # Run label TS strictly based on YAML 'enabled', no fallback logic
        try:
            log.info("Running label time-series clustering (per YAML enable=true)...")
            label_results, lt_times, label_mean_ts, label_names = cluster_stats.run_label_timecourse_cluster_test(
                label_source_contrasts, fsaverage_src, config
            )
            aux_dir.mkdir(parents=True, exist_ok=True)
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
                    "Label-wise cluster permutation results (p-values uncorrected across labels).",
                    f"n_subjects={len(label_source_contrasts)}; n_labels={len(label_names)}; "
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
                            cluster_values = t_obs[t_inds]
                            cluster_t_sum = float(cluster_values.sum())
                            direction = "positive" if cluster_t_sum > 0 else ("negative" if cluster_t_sum < 0 else "mixed")
                            tmin_ms = float(lt_times[t_inds.min()] * 1000.0)
                            tmax_ms = float(lt_times[t_inds.max()] * 1000.0)
                            peak_i = int(np.abs(cluster_values).argmax())
                            peak_ms = float(lt_times[t_inds[peak_i]] * 1000.0)
                            lines.append(
                                f"{label_name}: SIGNIFICANT cluster p={pval:.4f} ({direction}, t-sum={cluster_t_sum:.2f}) "
                                f"at {tmin_ms:.1f}-{tmax_ms:.1f} ms (peak {peak_ms:.1f} ms)"
                            )
                    if not label_sig:
                        lines.append(f"{label_name}: no significant clusters at cluster_alpha={lt_alpha:.3f}")
                if not sig_any:
                    lines.append("No label time-series clusters reached significance.")
                (aux_dir / "label_cluster_summary.txt").write_text("\n".join(lines))
            except Exception as e:
                log.warning(f"Failed to write label time-series summary: {e}")
        except Exception as e:
            log.warning(f"Label time-series analysis failed: {e}")
    else:
        if need_aux_dir and aux_dir.exists():
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
    plotting.plot_grand_average_snapshot(stc_ga_for_reporting, config, output_dir)
    if vertex_enabled:
        plotting.plot_source_clusters(stats_results, stc_ga_for_reporting, config, output_dir)
    else:
        log.info("Vertex clustering disabled; skipping vertex-level plotting.")

    if lt_cfg is not None and lt_enabled:
        plotting.plot_label_top_highlight(
            stc_ga_for_reporting,
            config,
            output_dir,
            summary_path,
        )

    plotting.plot_grand_average_peak(
        stc_ga_for_reporting,
        config,
        output_dir,
        stats_results if vertex_enabled else stats_results,
        summary_path if summary_path.exists() else None,
    )

    movie_artifacts = {}
    try:
        movie_artifacts = movie_utils.generate_movies(
            config=config,
            output_dir=output_dir,
            analysis_name=analysis_name,
            derivatives_root=Path(derivatives_root),
            logger=log,
        )
        if movie_artifacts:
            log.info("Visualization movies exported: %s", ", ".join(movie_artifacts.values()))
    except Exception as exc:
        log.warning("Movie generation failed: %s", exc, exc_info=True)

    reporter.generate_source_report(
        stats_results,
        stc_ga_for_reporting,
        config,
        output_dir,
        len(all_source_contrasts_for_stats),
        movie_artifacts=movie_artifacts,
    )

    # --- 7. Generate and Save Anatomical Report ---
    # This report provides detailed anatomical labels for any significant clusters.
    log.info("Generating detailed anatomical report for significant clusters...")
    project_root = Path(__file__).resolve().parents[1]
    subjects_dir = project_root / 'data' / 'fs_subjects_dir'
    if not subjects_dir.exists():
        raise FileNotFoundError(
            f"FreeSurfer subjects directory not found at {subjects_dir}. "
            "Expected fsaverage under data/fs_subjects_dir/."
        )

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
    # Generate the HS-style cortical summary ONLY when HS normalization is explicitly enabled
    if bool((config.get('postprocess') or {}).get('normalize_log_total_current', False)):
        hs_df = generate_hs_summary_table(
            stats_results, fsaverage_src, "fsaverage", str(subjects_dir), config
        )
        if not hs_df.empty:
            hs_df.to_csv(hs_fname, index=False)
            log.info(f"Cortical Cluster Localization summary saved to: {hs_fname}")


    log.info("-" * 80)
    log.info(f"Source pipeline finished successfully for '{analysis_name}'.")
    log.info(f"All outputs are saved in: {output_dir}")
    log.info("-" * 80)
    return output_dir


if __name__ == "__main__":
    main()

