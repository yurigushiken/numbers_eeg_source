"""
SFN2 Reporting Utilities
"""
import logging
import numpy as np

log = logging.getLogger()


def generate_report(stats_results, times, ch_names, config, output_dir):
    """
    Generates a text report summarizing the cluster statistics results.

    Args:
        stats_results (tuple): The output from the MNE cluster test.
        times (np.ndarray): The time vector.
        ch_names (list): The list of channel names.
        config (dict): The analysis configuration dictionary.
        output_dir (Path): The directory to save the report in.
    """
    t_obs, clusters, cluster_p_values, _ = stats_results
    alpha = config['stats']['cluster_alpha']

    report_path = output_dir / f"{config['analysis_name']}_report.txt"
    log.info(f"Generating statistical report at: {report_path}")

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Cluster Analysis Report: {config['analysis_name']}\n")
        f.write("=" * 80 + "\n\n")

        epoch_cfg = config.get('epoch_window', {})
        ep_tmin, ep_tmax = epoch_cfg.get('tmin'), epoch_cfg.get('tmax')
        baseline = epoch_cfg.get('baseline')
        analysis_window = config.get('stats', {}).get('analysis_window')

        f.write("Analysis Parameters:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Contrast: {config['contrast']['name']}\n")
        f.write(f"  Condition A set: {config['contrast']['condition_A']['condition_set_name']}\n")
        f.write(f"  Condition B set: {config['contrast']['condition_B']['condition_set_name']}\n")
        f.write(f"Epoch Window (loaded): {ep_tmin}s to {ep_tmax}s\n")
        if baseline:
            f.write(f"Baseline: {baseline[0]}s to {baseline[1]}s\n")
        if analysis_window:
            f.write(f"Statistical Window (tested): {analysis_window[0]}s to {analysis_window[1]}s\n")
        f.write("\n")

        f.write("Statistical Parameters:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Cluster-forming p-value (initial): {config['stats']['p_threshold']}\n")
        f.write(f"Cluster significance alpha: {alpha}\n")
        f.write(f"Number of permutations: {config['stats']['n_permutations']}\n")
        f.write(f"Test tail: {'two-sided' if config['stats']['tail'] == 0 else ('positive' if config['stats']['tail'] == 1 else 'negative')}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 80 + "\n\n")

        sig_cluster_indices = np.where(cluster_p_values < alpha)[0]
        if not sig_cluster_indices.size:
            f.write("No significant clusters found.\n")
            log.info("Reported no significant clusters.")
        else:
            f.write(f"Found {len(sig_cluster_indices)} significant cluster(s).\n\n")
            log.info(f"Reporting on {len(sig_cluster_indices)} significant cluster(s).")
            
            # Sort clusters by p-value for reporting
            sorted_indices = sig_cluster_indices[np.argsort(cluster_p_values[sig_cluster_indices])]

            for i, idx in enumerate(sorted_indices):
                p_val = cluster_p_values[idx]
                mask = clusters[idx]
                
                # --- Calculate enriched stats ---
                t_values_in_cluster = t_obs[mask]
                cluster_mass = np.sum(t_values_in_cluster)
                
                # Find peak and its location
                peak_idx_flat = np.argmax(np.abs(t_values_in_cluster))
                peak_t_val = t_values_in_cluster[peak_idx_flat]
                
                # Convert flat index back to 2D (time, channel)
                time_indices, ch_indices = np.where(mask)
                peak_time_idx = time_indices[peak_idx_flat]
                peak_ch_idx = ch_indices[peak_idx_flat]
                
                peak_time_ms = times[peak_time_idx] * 1000
                peak_ch_name = ch_names[peak_ch_idx]

                # Get time window of the cluster
                time_mask = mask.any(axis=1)
                cluster_times = times[time_mask]
                tmin, tmax = cluster_times[0], cluster_times[-1]

                # Get channels in the cluster
                ch_mask = mask.any(axis=0)
                cluster_ch_names = [ch_names[c_idx] for c_idx, in_cluster in enumerate(ch_mask) if in_cluster]

                f.write("-" * 40 + "\n")
                f.write(f"Cluster #{i+1} (p-value = {p_val:.4f})\n")
                f.write("-" * 40 + "\n")
                f.write(f"  - Cluster mass (sum of t-values): {cluster_mass:.2f}\n")
                f.write(f"  - Peak t-value: {peak_t_val:.3f} at {peak_time_ms:.1f} ms on channel {peak_ch_name}\n")
                f.write(f"  - Time window: {tmin*1000:.1f} ms to {tmax*1000:.1f} ms\n")
                f.write(f"  - Number of channels: {len(cluster_ch_names)}\n")
                f.write(f"  - Channels involved: {', '.join(cluster_ch_names)}\n\n")

    log.info("Report generation complete.")


def generate_source_report(stats_results, stc_grand_average, config, output_dir):
    """
    Generates a text report summarizing the source-space cluster results.
    """
    t_obs, clusters, cluster_p_values, _ = stats_results
    alpha = config['stats']['cluster_alpha']
    times = stc_grand_average.times

    report_path = output_dir / f"{config['analysis_name']}_report.txt"
    log.info(f"Generating source statistical report at: {report_path}")

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Source Cluster Analysis Report: {config['analysis_name']}\n")
        f.write("=" * 80 + "\n\n")

        # Analysis parameters
        epoch_cfg = config.get('epoch_window', {})
        baseline = epoch_cfg.get('baseline')
        analysis_window = config.get('stats', {}).get('analysis_window')

        f.write("Analysis Parameters:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Contrast: {config['contrast']['name']}\n")
        f.write(f"  Condition A set: {config['contrast']['condition_A']['condition_set_name']}\n")
        f.write(f"  Condition B set: {config['contrast']['condition_B']['condition_set_name']}\n")
        if baseline:
            f.write(f"Baseline: {baseline[0]}s to {baseline[1]}s\n")
        if analysis_window:
            f.write(f"Statistical Window (tested): {analysis_window[0]}s to {analysis_window[1]}s\n")
        else:
            # If no analysis window, show the effective window from the STC object
            f.write(f"Statistical Window (tested): {stc_grand_average.times[0]:.3f}s to {stc_grand_average.times[-1]:.3f}s\n")
        f.write(f"Source Method: {config['source']['method']}\n")
        f.write(f"Source SNR: {config['source']['snr']}\n")
        f.write("\n")

        f.write("Statistical Parameters:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Cluster-forming p-value (initial): {config['stats']['p_threshold']}\n")
        f.write(f"Cluster significance alpha: {alpha}\n")
        f.write(f"Number of permutations: {config['stats']['n_permutations']}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 80 + "\n\n")

        sig_cluster_indices = np.where(cluster_p_values < alpha)[0]
        if not sig_cluster_indices.size:
            f.write("No significant clusters found.\n")
        else:
            f.write(f"Found {len(sig_cluster_indices)} significant cluster(s).\n\n")
            
            # --- Filter and sort clusters for reporting (CH's suggestion) ---
            alpha = float(config['stats']['cluster_alpha'])
            sig_clusters_with_p = [(clusters[i], cluster_p_values[i]) for i in sig_cluster_indices]

            # Sort by p-value
            sorted_clusters = sorted(sig_clusters_with_p, key=lambda x: x[1])

            # Cap the number of reported clusters
            max_report = int(config['stats'].get('max_report_clusters', 25))
            if len(sorted_clusters) > max_report:
                f.write(f"Displaying the top {max_report} most significant clusters (out of {len(sorted_clusters)} found):\n\n")
                sorted_clusters = sorted_clusters[:max_report]

            for i, (clu, p_val) in enumerate(sorted_clusters):
                t_inds, v_inds = clu

                # --- Filter out tiny clusters (fixed logic) ---
                min_cells = int(config['stats'].get('min_cluster_cells', 60))
                if len(t_inds) < min_cells:
                    continue

                # --- Calculate enriched stats (consultant's correct pairwise indexing) ---
                vals = t_obs[(t_inds, v_inds)] # Correctly select only cluster elements

                cluster_mass  = float(vals.sum())
                peak_i        = int(np.abs(vals).argmax())
                peak_t_val    = float(vals[peak_i])
                peak_time_ms  = float(times[t_inds[peak_i]] * 1000.0)
                peak_vertex   = int(v_inds[peak_i])

                tmin_ms = float(times[int(t_inds.min())] * 1000.0)
                tmax_ms = float(times[int(t_inds.max())] * 1000.0)
                n_verts = int(np.unique(v_inds).size)

                f.write("-" * 40 + "\n")
                f.write(f"Cluster #{i+1} (p-value = {p_val:.4f})\n")
                f.write("-" * 40 + "\n")
                f.write(f"  - Cluster mass (sum of t-values): {cluster_mass:.2f}\n")
                f.write(f"  - Peak t-value: {peak_t_val:.3f} at {peak_time_ms:.1f} ms (vertex #{peak_vertex})\n")
                f.write(f"  - Time window: {tmin_ms:.1f} ms to {tmax_ms:.1f} ms\n")
                f.write(f"  - Number of vertices: {n_verts}\n\n")
    
    log.info("Source report generation complete.")
