"""
SFN2 Reporting Utilities
"""
import logging
import numpy as np
from pathlib import Path
import json as _json
from scipy.stats import t as t_dist

log = logging.getLogger()


def _compute_peak_descriptives(mean_value, peak_t, n_subjects):
    """
    Compute mean difference CI and Cohen's d (with CI) for a one-sample contrast.
    mean_value is already expressed in the desired units (e.g., µV).
    """
    if n_subjects is None or n_subjects < 2:
        return None
    if np.isclose(peak_t, 0.0):
        return None

    se = mean_value / peak_t  # standard error in same units as mean_value
    # If standard error is (near) zero, CIs collapse to the mean; do not drop the estimate.
    # Proceed with CI at the mean and carry forward d from t.
    # This avoids spurious 'insufficient data' when mean is tiny or t is large.
    if np.isclose(se, 0.0):
        se = 0.0

    dof = n_subjects - 1
    t_crit = t_dist.ppf(0.975, dof)

    ci_mean_lower = mean_value - t_crit * se
    ci_mean_upper = mean_value + t_crit * se
    # Sort in case of negative means
    ci_mean = (min(ci_mean_lower, ci_mean_upper), max(ci_mean_lower, ci_mean_upper))

    # Cohen's d for one-sample (difference wave) is t / sqrt(n)
    cohen_d = peak_t / np.sqrt(n_subjects)
    sd = se * np.sqrt(n_subjects)  # SD of the contrast in same units as mean_value
    if np.isclose(sd, 0.0):
        return {
            "mean": mean_value,
            "ci_mean": ci_mean,
            "cohen_d": cohen_d,
            "ci_d": (np.nan, np.nan),
        }

    ci_d_lower = ci_mean[0] / sd
    ci_d_upper = ci_mean[1] / sd
    ci_d = (min(ci_d_lower, ci_d_upper), max(ci_d_lower, ci_d_upper))

    return {
        "mean": mean_value,
        "ci_mean": ci_mean,
        "cohen_d": cohen_d,
        "ci_d": ci_d,
    }


def generate_report(stats_results, times, ch_names, config, output_dir, grand_average, n_subjects):
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

    with open(report_path, 'w', encoding='utf-8') as f:
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

        # Report ROI restriction if used
        roi_cfg = config.get('stats', {}).get('roi')
        if roi_cfg:
            # Check if ROI was actually applied (ch_names will be subset if ROI used)
            total_channels = len(grand_average.info['ch_names'])
            tested_channels = len(ch_names)
            roi_restricted = tested_channels < total_channels

            if roi_restricted:
                f.write(f"\nROI Restriction:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Analysis restricted to {tested_channels} of {total_channels} channels\n")

                # Report which channel groups were used
                if 'channel_groups' in roi_cfg:
                    groups = roi_cfg['channel_groups']
                    f.write(f"Channel groups: {', '.join(groups)}\n")

                # List the specific channels
                f.write(f"Channels tested: {', '.join(ch_names)}\n")
                f.write("\nNOTE: Statistical inference is restricted to the specified ROI.\n")
                f.write("Clusters can only form within these channels.\n")

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

                # Mean difference at the peak (convert to µV for interpretability)
                mean_diff_uv = float(grand_average.data[peak_ch_idx, peak_time_idx] * 1e6)
                descriptives = _compute_peak_descriptives(mean_diff_uv, peak_t_val, n_subjects)

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

                if descriptives is not None:
                    ci_mean_low, ci_mean_high = descriptives["ci_mean"]
                    ci_d_low, ci_d_high = descriptives["ci_d"]
                    f.write(f"    Mean difference: {descriptives['mean']:.2f} µV ")
                    f.write(f"(95% CI [{ci_mean_low:.2f}, {ci_mean_high:.2f}])\n")
                    f.write(f"    Cohen's d: {descriptives['cohen_d']:.2f} ")
                    f.write(f"(95% CI [{ci_d_low:.2f}, {ci_d_high:.2f}])\n")
                else:
                    f.write("    Mean difference / effect size could not be estimated (insufficient data).\n")

                f.write(f"  - Time window: {tmin*1000:.1f} ms to {tmax*1000:.1f} ms\n")
                f.write(f"  - Number of channels: {len(cluster_ch_names)}\n")
                f.write(f"  - Channels involved: {', '.join(cluster_ch_names)}\n\n")

    log.info("Report generation complete.")


def generate_source_report(stats_results, stc_grand_average, config, output_dir, n_subjects):
    """
    Generates a text report summarizing the source-space cluster results.
    """
    if stats_results is None:
        t_obs = np.empty((0, 0))
        clusters = []
        cluster_p_values = np.empty((0,))
    else:
        t_obs, clusters, cluster_p_values, _ = stats_results
    alpha = config['stats']['cluster_alpha']
    times = stc_grand_average.times
    vertex_cfg = (config.get('stats', {}).get('vertex') or {})
    vertex_enabled = bool(vertex_cfg.get('enabled', True))

    report_path = output_dir / f"{config['analysis_name']}_report.txt"
    log.info(f"Generating source statistical report at: {report_path}")

    with open(report_path, 'w', encoding='utf-8') as f:
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

        # Inverse operator provenance (if available)
        try:
            prov_path = Path(output_dir) / "inverse_provenance.json"
            if prov_path.exists():
                rows = _json.loads(prov_path.read_text())
                used_types = [r.get('used', '') for r in rows]
                pre_count = sum(1 for u in used_types if u == 'precomputed')
                tpl_count = sum(1 for u in used_types if u == 'template')
                f.write("Inverse Operators:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Precomputed: {pre_count}, Template: {tpl_count}\n")
                preview = rows[:min(5, len(rows))]
                for r in preview:
                    f.write(f"  - {r.get('subject','?')}: {r.get('used','?')} ({r.get('size_mb','?')} MB) -> {r.get('path','')}\n")
                if len(rows) > len(preview):
                    f.write(f"  ... ({len(rows) - len(preview)} more; see inverse_provenance.json)\n")
                f.write("\n")
        except Exception:
            pass

        f.write("=" * 80 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 80 + "\n\n")

        sig_cluster_indices = np.where(cluster_p_values < alpha)[0]
        if not vertex_enabled:
            f.write("Vertex-space clustering was disabled in this analysis.\n")
        if not sig_cluster_indices.size:
            if vertex_enabled:
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
                peak_time_idx = int(t_inds[peak_i])
                peak_vertex_local = int(v_inds[peak_i])

                keep_idx = config.get('stats', {}).get('_keep_idx')
                peak_vertex_global = None
                if keep_idx is not None:
                    keep_idx = np.asarray(keep_idx, dtype=int)
                    if peak_vertex_local >= keep_idx.size:
                        log.warning("Peak vertex index exceeds ROI mapping; skipping descriptives for this cluster.")
                    else:
                        peak_vertex_global = int(keep_idx[peak_vertex_local])
                else:
                    peak_vertex_global = peak_vertex_local

                vertices_lh, vertices_rh = stc_grand_average.vertices
                n_lh = len(vertices_lh)
                total_vertices = n_lh + len(vertices_rh)
                amplitude_info = None

                if peak_vertex_global is None:
                    log.warning("Peak vertex mapping unavailable; skipping descriptives for this cluster.")
                elif peak_vertex_global >= total_vertices:
                    log.warning("Peak vertex index out of bounds for grand average STC; skipping descriptives.")
                else:
                    hemi_label = "LH" if peak_vertex_global < n_lh else "RH"
                    if hemi_label == "LH":
                        vertex_id = int(vertices_lh[peak_vertex_global])
                        data_row = peak_vertex_global
                    else:
                        offset = peak_vertex_global - n_lh
                        vertex_id = int(vertices_rh[offset])
                        data_row = peak_vertex_global

                    mean_diff_value = float(stc_grand_average.data[data_row, peak_time_idx])
                    method = (config.get('source', {}).get('method') or '').lower()
                    if method == 'mne':
                        scale = 1e9
                        unit = "nAm"
                    else:
                        scale = 1.0
                        unit = "a.u."
                    mean_diff_scaled = mean_diff_value * scale
                    amplitude_info = {
                        "vertex_label": f"{vertex_id} ({hemi_label})",
                        "unit": unit,
                        "mean_scaled": mean_diff_scaled,
                        "descriptives": _compute_peak_descriptives(mean_diff_scaled, peak_t_val, n_subjects),
                    }

                tmin_ms = float(times[int(t_inds.min())] * 1000.0)
                tmax_ms = float(times[int(t_inds.max())] * 1000.0)
                n_verts = int(np.unique(v_inds).size)

                f.write("-" * 40 + "\n")
                f.write(f"Cluster #{i+1} (p-value = {p_val:.4f})\n")
                f.write("-" * 40 + "\n")
                f.write(f"  - Cluster mass (sum of t-values): {cluster_mass:.2f}\n")
                if amplitude_info is not None:
                    f.write(f"  - Peak t-value: {peak_t_val:.3f} at {peak_time_ms:.1f} ms (vertex {amplitude_info['vertex_label']})\n")
                    desc = amplitude_info["descriptives"]
                    if desc is not None:
                        ci_mean_low, ci_mean_high = desc["ci_mean"]
                        ci_d_low, ci_d_high = desc["ci_d"]
                        f.write(f"    Mean difference: {desc['mean']:.3f} {amplitude_info['unit']} ")
                        f.write(f"(95% CI [{ci_mean_low:.3f}, {ci_mean_high:.3f}])\n")
                        f.write(f"    Cohen's d: {desc['cohen_d']:.2f} ")
                        f.write(f"(95% CI [{ci_d_low:.2f}, {ci_d_high:.2f}])\n")
                    else:
                        f.write("    Mean difference / effect size could not be estimated (insufficient data).\n")
                else:
                    f.write(f"  - Peak t-value: {peak_t_val:.3f} at {peak_time_ms:.1f} ms (vertex index unavailable)\n")
                    f.write("    Mean difference / effect size could not be estimated (mapping issue).\n")
                f.write(f"  - Time window: {tmin_ms:.1f} ms to {tmax_ms:.1f} ms\n")
                f.write(f"  - Number of vertices: {n_verts}\n\n")
    
    log.info("Source report generation complete.")
