"""
Source-Space Analysis Pipeline
"""
import argparse
import logging
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


def main(config_path=None, accuracy=None, data_source=None):
    if config_path is None or accuracy is None:
        parser = argparse.ArgumentParser(description="Run Source-Space Analysis Pipeline")
        parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file.")
        parser.add_argument("--accuracy", type=str, required=True, choices=['all', 'acc1'], help="Dataset to use.")
        parser.add_argument("--data-source", type=str, default="new",
                            help="Data source: 'new' (default), 'old', or custom path.")
        args = parser.parse_args()
        config_path = args.config
        accuracy = args.accuracy
        data_source = args.data_source

    # Use default if data_source still None
    if data_source is None:
        data_source = "new"

    # --- 1. Load Config and Setup ---
    config = data_loader.load_config(config_path)
    analysis_name = config['analysis_name']

    # Generate a timestamped name for the analysis folder and report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_analysis_name = f"{timestamp}-{analysis_name}"

    output_dir = Path("derivatives/source") / timestamped_analysis_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory created at: {output_dir}")

    # --- 2. Load Data and Compute Source Contrasts ---
    subject_dirs = data_loader.get_subject_dirs(accuracy, data_source=data_source)
    fsaverage_src = data_loader.get_fsaverage_src()

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
    stats_results = cluster_stats.run_source_cluster_test(all_source_contrasts_for_stats, fsaverage_src, config)

    # If no significant clusters, optionally run label time-course clustering for sensitivity
    try:
        t_obs, clusters, cluster_p_values, _ = stats_results
        if not (cluster_p_values < config['stats']['cluster_alpha']).any():
            if 'label_timeseries' in config['stats']:
                log.info("Primary source clustering null; running label time-course 1D clustering...")
                lt_results = cluster_stats.run_label_timecourse_cluster_test(
                    all_source_contrasts_for_stats, fsaverage_src, config
                )
                # Store auxiliary results for potential report use
                aux_dir = output_dir / "aux"
                aux_dir.mkdir(exist_ok=True)
                np.save(aux_dir / "label_times.npy", lt_results[1])
                np.save(aux_dir / "label_mean_ts.npy", lt_results[2])
                (aux_dir / "labels.txt").write_text("\n".join(lt_results[3]))

                # Also write a concise textual summary of significant label time clusters
                try:
                    lt_t_obs, lt_clusters, lt_cluster_p, _ = lt_results[0]
                    lt_times = lt_results[1]
                    alpha = float(config['stats']['cluster_alpha'])
                    lines = [
                        f"n_subjects={len(all_source_contrasts_for_stats)}; window={lt_times[0]:.3f}-{lt_times[-1]:.3f}s; "
                        f"tail={int(config['stats']['tail'])}; threshold computed from p_threshold={config['stats']['p_threshold']}"
                    ]
                    sig_any = False
                    for idx, pval in enumerate(lt_cluster_p):
                        if pval < alpha:
                            sig_any = True
                            mask = lt_clusters[idx]
                            t_inds = np.where(mask)[0]
                            if t_inds.size == 0:
                                continue
                            tmin_ms = float(lt_times[t_inds.min()] * 1000.0)
                            tmax_ms = float(lt_times[t_inds.max()] * 1000.0)
                            vals = lt_t_obs[t_inds]
                            peak_i = int(np.abs(vals).argmax())
                            peak_ms = float(lt_times[t_inds[peak_i]] * 1000.0)
                            lines.append(
                                f"ROI combined: SIGNIFICANT time cluster p={pval:.4f} at {tmin_ms:.1f}-{tmax_ms:.1f} ms, "
                                f"peak at {peak_ms:.1f} ms"
                            )
                    if not sig_any:
                        lines.append("No significant label time-series clusters at cluster_alpha=%.3f" % alpha)
                    (aux_dir / "label_cluster_summary.txt").write_text("\n".join(lines))
                except Exception as e:
                    log.warning(f"Failed to write label time-series summary: {e}")
    except Exception as e:
        log.warning(f"Label time-course fallback failed: {e}")

    # Persist provenance JSON next to outputs
    try:
        import json as _json
        (output_dir / "inverse_provenance.json").write_text(_json.dumps(provenance, indent=2))
    except Exception:
        pass

    # --- 6. Generate Report and Visualizations ---
    log.info("Generating source report and plots...")
    reporter.generate_source_report(stats_results, stc_ga_for_reporting, config, output_dir)
    plotting.plot_source_clusters(stats_results, stc_ga_for_reporting, config, output_dir)

    # --- 7. Generate and Save Anatomical Report ---
    # This report provides detailed anatomical labels for any significant clusters.
    log.info("Generating detailed anatomical report for significant clusters...")
    try:
        subjects_dir = Path(__file__).resolve().parents[1] / 'data' / 'all' / 'fs_subjects_dir'
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

