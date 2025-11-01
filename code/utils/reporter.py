"""
SFN2 Reporting Utilities
"""
import logging
from pathlib import Path

import mne
import numpy as np
from scipy.stats import t as t_dist
import json as _json

log = logging.getLogger()
_APARC_LABEL_CACHE = {}


def _get_aparc_lookup(parc: str, subjects_dir: Path) -> dict:
    key = (parc, Path(subjects_dir).resolve())
    if key not in _APARC_LABEL_CACHE:
        lookup = {'lh': [], 'rh': []}
        for hemi in ('lh', 'rh'):
            labels = mne.read_labels_from_annot(
                subject='fsaverage',
                parc=parc,
                hemi=hemi,
                subjects_dir=str(subjects_dir),
                verbose=False,
            )
            lookup[hemi] = [(lab.name, np.asarray(lab.vertices, dtype=int)) for lab in labels]
        _APARC_LABEL_CACHE[key] = lookup
    return _APARC_LABEL_CACHE[key]


def _describe_cluster_with_aparc(vertex_ids_by_hemi: dict, subjects_dir: Path, parc: str = 'aparc', top_n: int = 3) -> dict:
    lookup = _get_aparc_lookup(parc, subjects_dir)
    out: dict[str, list[tuple[str, int]]] = {}
    for hemi in ('lh', 'rh'):
        verts = np.asarray(vertex_ids_by_hemi.get(hemi, []), dtype=int)
        if verts.size == 0:
            continue
        hits = []
        for name, lab_vertices in lookup[hemi]:
            count = int(np.intersect1d(verts, lab_vertices, assume_unique=False).size)
            if count:
                hits.append((name, count))
        if hits:
            hits.sort(key=lambda x: x[1], reverse=True)
            out[hemi] = hits[:top_n]
    return out


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

        f.write("\nPolarity Handling:\n")
        f.write("-" * 20 + "\n")
        align_requested = bool((config.get('postprocess') or {}).get('align_sign', False))
        make_magnitude = bool((config.get('postprocess') or {}).get('make_magnitude', False))
        magnitude_note = " Vertex-level stats were run on |current|; sign is not interpretable." if make_magnitude else ""
        if align_requested:
            f.write(
                "Config requested postprocess.align_sign=true. Vertex-wise analyses run without a global flip; "
                "label time-series handle polarity locally via mode=\"mean_flip\"." + magnitude_note + "\n\n"
            )
        else:
            f.write(
                "Vertex-wise analyses run without a global sign flip; label time-series handle polarity locally via "
                "mode=\"mean_flip\"." + magnitude_note + "\n\n"
            )
        restrict_labels = (config.get('stats', {}).get('label_timeseries') or {}).get('restrict_to') or []
        if restrict_labels:
            f.write(
                "This run used signed data (no orientation-invariant transform). Label tests were restricted to "
                f"{', '.join(restrict_labels)} to probe the anterior–posterior dipole suggested by sensor-space clusters.\n\n"
            )

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


def generate_source_report(
    stats_results,
    stc_grand_average,
    config,
    output_dir,
    n_subjects,
    *,
    movie_artifacts: dict | None = None,
):
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

    movie_artifacts = movie_artifacts or {}

    report_path = output_dir / f"{config['analysis_name']}_report.txt"
    log.info(f"Generating source statistical report at: {report_path}")

    project_root = Path(__file__).resolve().parents[2]
    subjects_dir = project_root / 'data' / 'fs_subjects_dir'

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
            f.write(f"Statistical Window (tested): {stc_grand_average.times[0]:.3f}s to {stc_grand_average.times[-1]:.3f}s\n")
        f.write(f"Source Method: {config['source']['method']}\n")
        f.write(f"Source SNR: {config['source']['snr']}\n")
        f.write("\n")

        if config.get('_hs_normalized'):
            power = config.get('_hs_normalization_power', 1.0)
            norm_desc = "total current" if abs(power - 1.0) < 1e-6 else "total power"
            f.write("Post-processing:\n")
            f.write("-" * 20 + "\n")
            f.write(
                f"Source values were normalized to unit {norm_desc} per subject and "
                "log-transformed following Hyde & Spelke (2012) and the LORETA-KEY documentation.\n"
            )
            f.write("\n")
        else:
            norm_desc = None

        f.write("Statistical Parameters:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Cluster-forming p-value (initial): {config['stats']['p_threshold']}\n")
        f.write(f"Cluster significance alpha: {alpha}\n")
        f.write(f"Number of permutations: {config['stats']['n_permutations']}\n")
        f.write("\n")

        if movie_artifacts:
            f.write("Visualization Movies:\n")
            f.write("-" * 20 + "\n")
            labels = {
                "source": "Source movie",
                "topo": "Sensor topomap movie",
                "stitched": "Side-by-side movie",
            }
            for key, path in movie_artifacts.items():
                label = labels.get(key, key.capitalize())
                f.write(f"{label}: {path}\n")
            f.write("\n")

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

            alpha = float(config['stats']['cluster_alpha'])
            sig_clusters_with_p = [(clusters[i], cluster_p_values[i]) for i in sig_cluster_indices]

            sorted_clusters = sorted(sig_clusters_with_p, key=lambda x: x[1])

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
                        # Use compact scientific formatting for arbitrary units to avoid misleading zeros
                        f.write(f"    Mean difference: {desc['mean']:.3g} {amplitude_info['unit']} ")
                        f.write(f"(95% CI [{ci_mean_low:.3g}, {ci_mean_high:.3g}])\n")
                        f.write(f"    Cohen's d: {desc['cohen_d']:.2f} ")
                        f.write(f"(95% CI [{ci_d_low:.2f}, {ci_d_high:.2f}])\n")
                    else:
                        f.write("    Mean difference / effect size could not be estimated (insufficient data).\n")
                else:
                    f.write(f"  - Peak t-value: {peak_t_val:.3f} at {peak_time_ms:.1f} ms (vertex index unavailable)\n")
                    f.write("    Mean difference / effect size could not be estimated (mapping issue).\n")

                cluster_vertices_global = np.unique(v_inds)
                lh_vertex_ids = np.array([], dtype=int)
                rh_vertex_ids = np.array([], dtype=int)
                if cluster_vertices_global.size:
                    cluster_vertices_global = cluster_vertices_global.astype(int)
                    if keep_idx is not None:
                        try:
                            keep_idx_arr = np.asarray(keep_idx, dtype=int)
                            cluster_vertices_global = keep_idx_arr[cluster_vertices_global]
                        except Exception:
                            pass
                    cluster_vertices_global = np.asarray(cluster_vertices_global, dtype=int)
                    lh_mask = cluster_vertices_global < n_lh
                    if lh_mask.any():
                        lh_vertex_ids = vertices_lh[cluster_vertices_global[lh_mask]]
                    if (~lh_mask).any():
                        rh_indices = cluster_vertices_global[~lh_mask] - n_lh
                        rh_vertex_ids = vertices_rh[rh_indices]

                aparc_hits = _describe_cluster_with_aparc(
                    {
                        'lh': lh_vertex_ids,
                        'rh': rh_vertex_ids,
                    },
                    subjects_dir,
                )
                if aparc_hits:
                    f.write("    Anatomy (aparc):\n")
                    for hemi_label in ('lh', 'rh'):
                        hits = aparc_hits.get(hemi_label)
                        if not hits:
                            continue
                        hit_str = ", ".join(f"{name} ({count})" for name, count in hits)
                        f.write(f"      {hemi_label}: {hit_str}\n")
                else:
                    f.write("    Anatomy (aparc): no label coverage detected.\n")

                f.write(f"  - Time window: {tmin_ms:.1f} ms to {tmax_ms:.1f} ms\n")
                f.write(f"  - Number of vertices: {n_verts}\n\n")
    
        label_cfg = (config.get('stats', {}).get('label_timeseries') or {})
        label_summary_path = Path(output_dir) / "aux" / "label_cluster_summary.txt"
        if bool(label_cfg.get('enabled', True)):
            if label_summary_path.exists():
                try:
                    rel_label_path = label_summary_path.relative_to(output_dir)
                except Exception:
                    rel_label_path = label_summary_path
                f.write(
                    "\nLabel-level tests were also run; see "
                    f"{rel_label_path} for detailed results (p-values uncorrected across labels).\n"
                )
            else:
                f.write(
                    "\nLabel-level tests were configured but no label summary file was generated. "
                    "Check logs for label clustering errors.\n"
                )
        else:
            f.write(
                "\nLabel-level tests were not run (stats.label_timeseries.enabled = false).\n"
            )

    log.info("Source report generation complete.")
