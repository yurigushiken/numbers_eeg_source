"""
SFN2 Sensor-Space Analysis Pipeline

This script serves as the main entrypoint for running the data-driven,
cluster-based permutation testing analysis on sensor-space EEG data.
"""
import argparse
import logging
from pathlib import Path
import mne
from tqdm import tqdm
from datetime import datetime

# It's crucial to import the utility modules we've created.
from code.utils import data_loader, cluster_stats, plotting, reporter

# Setup basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()


def main(config_path=None, accuracy=None):
    """
    Main function to orchestrate the sensor-space analysis pipeline.
    """
    if config_path is None:
        # If not called with args, parse from command line
        parser = argparse.ArgumentParser(description="Run Sensor-Space Analysis Pipeline")
        parser.add_argument("--config", type=str, required=True,
                            help="Path to the analysis configuration YAML file.")
        parser.add_argument("--accuracy", type=str, required=False, default=None, choices=['all', 'acc1', 'acc0'],
                            help=("Optional override for trial filtering. "
                                  "Use 'all', 'acc1' (correct only), or 'acc0' (incorrect only). "
                                  "If omitted, per-condition YAML 'accuracy' fields are used (fallback 'all')."))
        # Data source flag removed; combined preprocessed data is the standard
        args = parser.parse_args()
        config_path = args.config
        accuracy = args.accuracy
    

    # --- 1. Load Configuration and Setup ---
    config = data_loader.load_config(config_path)
    analysis_name = config['analysis_name']

    # Generate a timestamped name for the analysis folder and report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_analysis_name = f"{timestamp}-{analysis_name}"
    
    output_dir = Path("derivatives/sensor") / timestamped_analysis_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory created at: {output_dir}")

    # --- 2. Load Data and Compute Contrasts for Each Subject ---
    subject_dirs = data_loader.get_subject_dirs(accuracy)
    if not subject_dirs:
        log.error("No subject directories found. Exiting.")
        return

    log.info("Creating contrasts for each subject...")
    epoch_cfg = config.get('epoch_window', {})
    baseline = epoch_cfg.get('baseline')

    # Resolve per-condition accuracy with YAML taking precedence; CLI acts as a global fallback
    contrast_cfg = (config.get('contrast') or {})
    acc_A = ((contrast_cfg.get('condition_A') or {}).get('accuracy')) or accuracy
    acc_B = ((contrast_cfg.get('condition_B') or {}).get('accuracy')) or accuracy
    contrasts = []
    condA_evokeds_per_subject = []
    condB_evokeds_per_subject = []
    for subject_dir in tqdm(subject_dirs, desc="Processing subjects (sensor)"):
        # Unpack the tuple; contrast + all epochs
        contrast_evoked, _ = data_loader.create_subject_contrast(
            subject_dir, config, accuracy=accuracy
        )
        if contrast_evoked is not None:
            contrasts.append(contrast_evoked)

        # Also collect condition-specific evokeds to build group ERPs
        try:
            evoked_A, _ = data_loader.get_evoked_for_condition(
                subject_dir,
                config['contrast']['condition_A'],
                baseline=baseline,
                accuracy=acc_A,
            )
            evoked_B, _ = data_loader.get_evoked_for_condition(
                subject_dir,
                config['contrast']['condition_B'],
                baseline=baseline,
                accuracy=acc_B,
            )
            if evoked_A:
                condA_evokeds_per_subject.append(mne.grand_average(evoked_A))
            if evoked_B:
                condB_evokeds_per_subject.append(mne.grand_average(evoked_B))
        except Exception:
            pass

    if not contrasts:
        log.error("No valid contrasts could be created. Aborting analysis.")
        return
    log.info(f"Successfully created contrasts for {len(contrasts)} subjects.")

    # --- 3a. Optionally crop data for stats to analysis_window ---
    analysis_window = (config.get('stats') or {}).get('analysis_window')
    if analysis_window and len(analysis_window) == 2:
        try:
            aw0, aw1 = float(analysis_window[0]), float(analysis_window[1])
            log.info(f"Cropping subject contrasts to analysis_window {aw0:.3f}-{aw1:.3f}s for sensor stats")
            for c in contrasts:
                c.crop(tmin=aw0, tmax=aw1)
        except Exception:
            log.warning("Failed to crop contrasts to analysis_window; proceeding without cropping.")

    # --- 3b. Compute Grand Average ---
    log.info("Computing grand average contrast...")
    grand_average = mne.grand_average(contrasts)
    ga_fname = output_dir / f"{analysis_name}_grand_average-ave.fif"
    grand_average.save(ga_fname, overwrite=True)
    log.info(f"Grand average saved to {ga_fname}")

    # Group ERPs for each condition (if available)
    ga_cond_A = mne.grand_average(condA_evokeds_per_subject) if condA_evokeds_per_subject else None
    ga_cond_B = mne.grand_average(condB_evokeds_per_subject) if condB_evokeds_per_subject else None

    # --- 4. Run Group-Level Cluster Statistics ---
    stats_results, ch_names = cluster_stats.run_sensor_cluster_test(contrasts, config)

    # --- 5. Generate Report and Visualizations ---
    log.info("Generating report and plots...")
    # The `times` vector is needed for the report and plots
    times = grand_average.times

    # Generate text report
    reporter.generate_report(stats_results, times, ch_names, config, output_dir, grand_average, len(contrasts))

    # Generate ERP plot
    plotting.plot_contrast_erp(grand_average, stats_results, config, output_dir, ch_names)

    # Generate topomap plot
    plotting.plot_t_value_topomap(grand_average, stats_results, config, output_dir, ch_names)

    # Generate ROI condition ERP plots (P1, N1, P3b)
    if ga_cond_A is not None and ga_cond_B is not None:
        plotting.plot_condition_erps_rois(ga_cond_A, ga_cond_B, config, output_dir)

    log.info("-" * 80)
    log.info(f"Pipeline finished successfully for '{analysis_name}'.")
    log.info(f"All outputs are saved in: {output_dir}")
    log.info("-" * 80)
    return output_dir


if __name__ == "__main__":
    main()
