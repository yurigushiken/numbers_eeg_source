"""
Plotting Utilities for Cluster Analysis
"""
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import mne
from pathlib import Path
import re
from typing import Optional

from code.utils.electrodes import ELECTRODE_GROUPS
from code.utils.label_summary import parse_label_summary


log = logging.getLogger()

# Robust import for MNE private helper across versions
try:
    from mne.channels.layout import _find_topomap_coords  # canonical in newer MNE
except Exception:
    try:
        from mne.viz._topomap import _find_topomap_coords  # fallback for older wheels
    except Exception:
        _find_topomap_coords = None  # type: ignore


def _check_fsaverage_path():
    """Resolve fsaverage directory; fail fast if not under data/fs_subjects_dir."""
    project_root = Path(__file__).resolve().parents[2]
    subjects_dir = project_root / "data" / "fs_subjects_dir"
    if not subjects_dir.exists():
        raise FileNotFoundError(
            f"FreeSurfer subjects directory not found at {subjects_dir}. "
            "Expected fsaverage under data/fs_subjects_dir/."
        )
    fsaverage_path = subjects_dir / "fsaverage"
    return str(subjects_dir), fsaverage_path

def _split_vertices_for_fsaverage(vertices_like):
    """Return [lh, rh] vertex lists. Accepts a flat array or a [lh, rh] list."""
    if isinstance(vertices_like, (list, tuple)) and len(vertices_like) == 2:
        return [np.array(vertices_like[0]), np.array(vertices_like[1])]
    verts = np.array(vertices_like)
    n_total = len(verts)
    
    # Heuristic to find split point for fsaverage standard spaces like ico-5
    # A more robust solution would pass n_lh from the source space
    lh_verts = [v for v in verts if v < 10242] # ico-5 lh max is 10241
    rh_verts = [v for v in verts if v >= 10242]
    if len(lh_verts) + len(rh_verts) != n_total:
         log.warning("Vertex splitting heuristic may have failed.")
         n_hemi = n_total // 2
         return [verts[:n_hemi], verts[n_hemi:]]
    return [np.array(lh_verts), np.array(rh_verts)]


def _create_blank_image(path, message):
    """Creates a blank placeholder image with a text message."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12, wrap=True)
    ax.axis('off')
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _compose_source_figure(
    stc_data,
    vertices,
    tmin,
    tstep,
    colormap,
    clim,
    cbar_label,
    output_path,
    analysis_name,
    config,
    cluster_p_val,
    time_label,
    peak_info,
    subjects_dir,
):
    """Helper to generate and save a 6-view source plot composite."""
    stc_to_plot = mne.SourceEstimate(
        stc_data, vertices=vertices, tmin=tmin, tstep=tstep, subject='fsaverage'
    )

    # Render each view to a temporary tile
    views = ['lateral', 'medial', 'frontal', 'parietal', 'dorsal', 'ventral']
    tile_paths = []
    output_dir = Path(output_path).parent
    peak_vertno, hemi_code = peak_info['vertno'], peak_info['hemi_code']

    for view in views:
        brain = None
        try:
            brain = stc_to_plot.plot(
                subject='fsaverage',
                subjects_dir=subjects_dir,
                surface='white',
                hemi='both',
                views=[view],
                size=(800, 600),
                background='white',
                foreground='black',
                time_viewer=False,
                show_traces=False,
                colorbar=False,
                colormap=colormap,
                cortex='high_contrast',
                smoothing_steps=10,
                clim=clim,
                initial_time=float(stc_to_plot.times[0]),
                transparent=False,
                alpha=1.0,
                title="",
                silhouette=True,
            )
        except TypeError:
            log.debug("silhouette argument not supported; rendering without it.")
            brain = stc_to_plot.plot(
                subject='fsaverage', subjects_dir=subjects_dir, surface='white',
                hemi='both', views=[view], size=(800, 600), background='white',
                foreground='black', time_viewer=False, show_traces=False,
                colorbar=False, colormap=colormap, cortex='high_contrast',
                smoothing_steps=10, clim=clim,
                initial_time=float(stc_to_plot.times[0]), transparent=False,
                alpha=1.0, title=""
            )
        if peak_vertno is not None:
            brain.add_foci(
                [peak_vertno], coords_as_verts=True,
                hemi=('lh' if hemi_code == 0 else 'rh'),
                scale_factor=0.7, color='black'
            )
        tmp_path = output_dir / f"{analysis_name}_tile_{view}.png"
        try:
            brain.save_image(tmp_path)
        except Exception:
            img = brain.screenshot()
            from matplotlib import image as mpimg
            mpimg.imsave(tmp_path, img)
        finally:
            brain.close()
        tile_paths.append(tmp_path)

    if not tile_paths:
        raise RuntimeError("Failed to render any source views.")

    # Compose tiles into a 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=300)
    for ax, img_path in zip(axes.ravel(), tile_paths):
        img = plt.imread(str(img_path))
        ax.imshow(img)
        ax.axis('off')
        img_path.unlink() # Clean up temporary tile

    # Add unified colorbar
    cmap = plt.get_cmap(colormap)
    vmax_cb = clim['pos_lims'][-1] if 'pos_lims' in clim else clim['lims'][-1]
    vmin_cb = -vmax_cb if 'pos_lims' in clim else clim['lims'][0]
    norm = Normalize(vmin=vmin_cb, vmax=vmax_cb)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.05, pad=0.10)
    cbar.set_label(cbar_label, fontsize=10)

    # --- Compose and add metadata strings ---
    title_main = f"{config['contrast']['name']}"
    if cluster_p_val is not None:
        subtitle_1 = (
            f"Most significant cluster (p={cluster_p_val:.3f}), averaged within {time_label} | "
            f"{peak_info['subtitle_1_stats']}"
        )
        subtitle_2 = (
            f"Peak t={peak_info['t']:.2f} at MNI {peak_info['mni_str']}, "
            f"region {peak_info['region']}, cluster size={peak_info['size']} vertices"
        )
    else:
        subtitle_text = peak_info.get('subtitle_label')
        if subtitle_text is None:
            stats_info = peak_info.get('subtitle_1_stats', '')
            subtitle_text = f"Label-only snapshot | {stats_info}" if stats_info else "Label-only snapshot"
        detail_text = peak_info.get('subtitle_label_detail')
        if detail_text is None:
            detail_text = (
                f"Peak |t|={peak_info['t']:.2f} at MNI {peak_info['mni_str']} (region {peak_info['region']})"
            )
        subtitle_1 = f"{subtitle_text}"
        subtitle_2 = detail_text
    fig.text(0.02, 0.995, title_main, ha='left', va='top', fontsize=22)
    fig.text(0.02, 0.955, subtitle_1, ha='left', va='top', fontsize=10)
    fig.text(0.02, 0.925, subtitle_2, ha='left', va='top', fontsize=10)

    fig.text(0.02, 0.06, peak_info['footer'], ha='left', va='bottom', fontsize=9)

    view_labels = ['Lateral', 'Medial', 'Frontal', 'Parietal', 'Dorsal', 'Ventral']
    for ax, lbl in zip(axes.ravel(), view_labels):
        bbox = ax.get_position()
        fig.text(bbox.x0 + 0.005, bbox.y0 - 0.02, lbl, ha='left', va='top', fontsize=9)

    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    log.info(f"Saved source cluster snapshot to {output_path}")


def plot_source_clusters(stats_results, stc_grand_average, config, output_dir):
    """
    Plots the full spatial extent of all significant source clusters, saving
    a separate figure for each.
    """
    try:
        analysis_name = config['analysis_name']
        subjects_dir, _ = _check_fsaverage_path()
        if subjects_dir is None:
            raise RuntimeError("subjects_dir could not be resolved for fsaverage")

        if stats_results is None:
            log.info("No vertex-level statistics provided; skipping source cluster plotting.")
            return

        t_obs, clusters, cluster_p_values, _ = stats_results
        alpha = float(config['stats']['cluster_alpha'])
        if clusters is None or cluster_p_values is None or len(clusters) == 0 or cluster_p_values.size == 0:
            msg = f"No significant source clusters at alpha={alpha}."
            path = output_dir / f"{analysis_name}_source_cluster.png"
            _create_blank_image(path, msg)
            log.info(msg)
            return

        # --- 1. Find ALL Significant Clusters ---
        sig_idx = np.where(cluster_p_values < alpha)[0]
        if sig_idx.size == 0:
            # If no significant clusters, create a single placeholder image and return
            msg = f"No significant source clusters at alpha={alpha}."
            path = output_dir / f"{analysis_name}_source_cluster.png"
            _create_blank_image(path, msg)
            log.info(msg)
            return

        # Order them by p-value for consistent numbering
        ordered_indices = sig_idx[np.argsort(cluster_p_values[sig_idx])]

        # Prepare storage for combined view
        full_vertices = [np.array(stc_grand_average.vertices[0]), np.array(stc_grand_average.vertices[1])]
        full_n = int(np.sum([len(v) for v in full_vertices]))
        combined_masks = []
        combined_p_values = []

        # --- Loop through each significant cluster ---
        n_sig = len(ordered_indices)
        for rank, cluster_idx in enumerate(ordered_indices, start=1):
            cl = clusters[cluster_idx]

            time_inds = np.unique(np.asarray(cl[0], dtype=int))
            n_times_grand_avg = stc_grand_average.times.size
            time_inds = time_inds[(time_inds >= 0) & (time_inds < n_times_grand_avg)]
            if time_inds.size == 0:
                log.warning(f"Skipping cluster #{rank} due to out-of-bounds time indices.")
                continue

            # --- 2. Prepare Data Array for the Current Cluster ---
            t_avg_all = t_obs[time_inds, :].mean(axis=0)
            v_inds_cluster_local = np.unique(np.asarray(cl[1], dtype=int))

            t_map_local = np.full_like(t_avg_all, np.nan, dtype=float)
            t_map_local[v_inds_cluster_local] = t_avg_all[v_inds_cluster_local]

            keep_idx = config.get('stats', {}).get('_keep_idx')
            t_map_full = t_map_local
            if keep_idx is not None and int(t_map_local.size) != full_n:
                keep_idx = np.asarray(keep_idx, dtype=int)
                full_vec = np.full(full_n, np.nan, dtype=float)
                full_vec[keep_idx] = t_map_local
                t_map_full = full_vec

            combined_masks.append(~np.isnan(t_map_full))
            combined_p_values.append(float(cluster_p_values[cluster_idx]))
            
            # --- 3. Calculate Metadata for the Current Cluster ---
            cluster_times = stc_grand_average.times[time_inds]
            time_label = f"{cluster_times.min()*1000:.0f}-{cluster_times.max()*1000:.0f} ms"
            
            peak_global_idx = int(np.nanargmax(np.abs(t_map_full)))
            peak_t = float(t_map_full[peak_global_idx])
            
            lh_n = len(full_vertices[0])
            hemi_code = 0 if peak_global_idx < lh_n else 1
            peak_vertno = int(full_vertices[0][peak_global_idx] if hemi_code == 0 else full_vertices[1][peak_global_idx - lh_n])
            
            peak_mni = mne.vertex_to_mni([peak_vertno], hemis=hemi_code, subject='fsaverage', subjects_dir=subjects_dir)[0]
            
            parc = (config.get('stats', {}).get('roi', {}) or {}).get('parc', 'aparc')
            labels = mne.read_labels_from_annot('fsaverage', parc=parc, subjects_dir=subjects_dir, verbose=False)
            peak_region = "Unknown"
            for lbl in labels:
                if lbl.hemi == ('lh' if hemi_code == 0 else 'rh') and peak_vertno in lbl.vertices:
                    peak_region = lbl.name.replace('-lh','').replace('-rh','')
                    break

            method = str(config.get('source', {}).get('method', 'dSPM')).upper()
            tail = int(config.get('stats', {}).get('tail', 0))
            pthr = config.get('stats', {}).get('p_threshold')
            perms = config.get('stats', {}).get('n_permutations')
            roi_cfg = config.get('stats', {}).get('roi', {}) or {}
            roi_labels = (
                ", ".join(roi_cfg.get('labels') or roi_cfg.get('channels', []))
                or 'whole brain'
            )

            peak_info = {
                't': peak_t,
                'vertno': peak_vertno,
                'hemi_code': hemi_code,
                'mni_str': f"({peak_mni[0]:.1f}, {peak_mni[1]:.1f}, {peak_mni[2]:.1f})",
                'region': peak_region,
                'size': int(np.count_nonzero(~np.isnan(t_map_full))),
                'subtitle_1_stats': f"method={method} -> cluster stats | tail={tail}, p_thr={pthr}, perms={perms} | ROI: {roi_labels}",
                'footer': f"fsaverage | pick_ori=normal | vertices kept {t_map_local.size}/{full_n}"
            }

            stc_tmin = float(cluster_times.mean())
            stc_tstep = 0.0

            # --- 4. Generate and Save the Plot for the Current Cluster ---
            vmax_abs = np.nanmax(np.abs(t_map_full))
            if not np.isfinite(vmax_abs) or vmax_abs <= 0:
                pos_lims_full = [1e-6, 2e-6, 3e-6]
            else:
                pos_lims_full = [vmax_abs * 0.25, vmax_abs * 0.5, vmax_abs * 0.95]
            
            eps_full = max(1e-6, 1e-6 * (pos_lims_full[-1] if np.isfinite(pos_lims_full[-1]) else 1.0))
            if not (pos_lims_full[1] > pos_lims_full[0]):
                pos_lims_full[1] = pos_lims_full[0] + eps_full
            if not (pos_lims_full[2] > pos_lims_full[1]):
                pos_lims_full[2] = pos_lims_full[1] + eps_full
            clim_full = dict(kind='value', pos_lims=pos_lims_full)
            
            # Determine the output path based on the rank of the cluster
            if rank == 1:
                # The most significant cluster's plot is the main one, without a number suffix
                path = output_dir / f"{analysis_name}_source_cluster.png"
            else:
                # Subsequent clusters get a numbered suffix
                path = output_dir / f"{analysis_name}_source_cluster_{rank}.png"
            
            _compose_source_figure(
                t_map_full.reshape(-1, 1), full_vertices, stc_tmin, stc_tstep, 'RdBu_r', clim_full,
                f'T-value (Full Range, Cluster #{rank})', path, analysis_name, config,
                cluster_p_values[cluster_idx], time_label, peak_info, subjects_dir
            )

        # Combined cluster overview
        if combined_masks:
            combined_data = np.zeros(full_n, dtype=float)
            for idx, mask in enumerate(combined_masks, start=1):
                combined_data[mask] = float(idx)
            max_idx = float(len(combined_masks))
            if max_idx <= 0:
                max_idx = 1.0
            combined_stc = combined_data.reshape(-1, 1)
            combined_clim = dict(kind='value', lims=[0.0, max_idx * 0.5, max_idx])
            combined_path = output_dir / f"{analysis_name}_source_clusters_combined.png"
            min_p = min(combined_p_values) if combined_p_values else 1.0
            combined_info = {
                't': 0.0,
                'vertno': None,
                'hemi_code': 0,
                'mni_str': '',
                'region': f"{n_sig} clusters",
                'size': int(np.count_nonzero(combined_data)),
                'subtitle_1_stats': f"Combined cluster view | k={n_sig}, min p={min_p:.4f}",
                'footer': "fsaverage | clusters tinted by rank"
            }
            _compose_source_figure(
                combined_stc,
                full_vertices,
                float(stc_grand_average.times[0]) if stc_grand_average.times.size else 0.0,
                0.0,
                'tab20',
                combined_clim,
                'Cluster index',
                combined_path,
                analysis_name,
                config,
                None,
                'combined',
                combined_info,
                subjects_dir,
            )

    except Exception as e:
        log.error(f"Failed to generate source cluster plots: {e}", exc_info=True)
        # Do not abort pipeline on plotting failure
        return


def plot_grand_average_snapshot(stc_grand_average, config, output_dir):
    """Render a continuous (time-averaged) grand-average STC preview."""
    try:
        analysis_name = config['analysis_name']
        subjects_dir, _ = _check_fsaverage_path()

        stc = stc_grand_average.copy()
        stats_cfg = config.get('stats', {}) or {}
        aw = stats_cfg.get('analysis_window')

        if aw and len(aw) == 2:
            tmin, tmax = float(aw[0]), float(aw[1])
            stc = stc.copy().crop(tmin=tmin, tmax=tmax)
            if stc.times.size == 0:
                stc = stc_grand_average.copy()
                data_vector = stc.data.mean(axis=1, keepdims=True)
                time_label = "full window"
                stc_tmin = float(stc.times[0]) if stc.times.size else 0.0
            else:
                data_vector = stc.data.mean(axis=1, keepdims=True)
                time_label = f"{tmin*1000:.0f}-{tmax*1000:.0f} ms"
                stc_tmin = tmin
        else:
            idx = int(np.argmax(np.abs(stc.data)))
            vert_idx, time_idx = np.unravel_index(idx, stc.data.shape)
            data_vector = stc.data[:, [time_idx]]
            time_label = f"{stc.times[time_idx]*1000:.1f} ms"
            stc_tmin = float(stc.times[time_idx])

        data_vec_flat = data_vector.reshape(-1)
        abs_vals = np.abs(data_vec_flat)
        vmax_abs = float(np.nanpercentile(abs_vals, 98)) if abs_vals.size else 0.0
        if not np.isfinite(vmax_abs) or vmax_abs <= 0:
            vmax_abs = float(np.nanmax(abs_vals)) if abs_vals.size else 0.0
        output_path = output_dir / f"{analysis_name}_grand_average_continuous.png"

        if not np.isfinite(vmax_abs) or vmax_abs <= 0:
            _create_blank_image(output_path, "Continuous STC preview unavailable (flat signal).")
            return

        clim = dict(kind='value', lims=[-vmax_abs, 0.0, vmax_abs])

        lh_n = len(stc_grand_average.vertices[0])
        peak_idx = int(np.nanargmax(np.abs(data_vec_flat)))
        hemi_code = 0 if peak_idx < lh_n else 1
        if hemi_code == 0:
            peak_vertno = int(stc_grand_average.vertices[0][peak_idx])
        else:
            peak_vertno = int(stc_grand_average.vertices[1][peak_idx - lh_n])
        peak_mni = mne.vertex_to_mni([peak_vertno], hemis=hemi_code, subject='fsaverage', subjects_dir=subjects_dir)[0]

        lt_cfg = stats_cfg.get('label_timeseries') or {}
        parc = lt_cfg.get('parc', 'aparc')
        try:
            labels = mne.read_labels_from_annot('fsaverage', parc=parc, subjects_dir=subjects_dir, verbose=False)
        except Exception:
            labels = []
        peak_region = "Unknown"
        for lbl in labels:
            if lbl.hemi == ('lh' if hemi_code == 0 else 'rh') and peak_vertno in lbl.vertices:
                peak_region = lbl.name.replace('-lh', '').replace('-rh', '')
                break

        method = str(config.get('source', {}).get('method', 'eLORETA')).upper()
        peak_info = {
            't': vmax_abs,
            'vertno': peak_vertno,
            'hemi_code': hemi_code,
            'mni_str': f"({peak_mni[0]:.1f}, {peak_mni[1]:.1f}, {peak_mni[2]:.1f})",
            'region': peak_region,
            'size': int(np.count_nonzero(np.abs(data_vec_flat) > 0)),
            'subtitle_1_stats': f"Continuous preview | method={method} | window={time_label}",
            'footer': "fsaverage | continuous preview",
        }

        _compose_source_figure(
            data_vector,
            stc_grand_average.vertices,
            stc_tmin,
            0.0,
            'RdBu_r',
            clim,
            'Contrast amplitude (a.u.)',
            output_path,
            analysis_name,
            config,
            None,
            time_label,
            peak_info,
            subjects_dir,
        )
    except Exception as e:
        log.error(f"Failed to render grand-average snapshot: {e}", exc_info=True)
        placeholder = output_dir / f"{config['analysis_name']}_grand_average_continuous.png"
        _create_blank_image(placeholder, f"Failed to render continuous preview: {e}")


def plot_label_top_highlight(stc_grand_average, config, output_dir, summary_path):
    """Render a highlight figure for the top significant label in label-only runs."""
    try:
        if not summary_path or not Path(summary_path).exists():
            return

        summary = parse_label_summary(Path(summary_path))
        if not summary.significant:
            log.info("No significant labels found for highlight plot.")
            return

        top_label = summary.significant[0]
        label_name = top_label.name
        if not label_name:
            return

        stats_cfg = config.get('stats', {}) or {}
        lt_cfg = stats_cfg.get('label_timeseries') or {}
        parc = lt_cfg.get('parc', 'aparc')

        subjects_dir, _ = _check_fsaverage_path()
        labels = mne.read_labels_from_annot(
            'fsaverage', parc=parc, subjects_dir=subjects_dir, verbose=False
        )
        if not labels:
            log.warning("Unable to load labels for parc %s; skipping label highlight", parc)
            return

        # Match label case-insensitively
        target = label_name.lower()
        label_obj = None
        for lbl in labels:
            if lbl.name.lower() == target:
                label_obj = lbl
                break

        if label_obj is None:
            log.warning("Top label %s not found in parc %s", label_name, parc)
            return

        analysis_name = config['analysis_name']
        output_path = output_dir / f"{analysis_name}_label_top.png"

        lh_vertices = np.array(stc_grand_average.vertices[0])
        rh_vertices = np.array(stc_grand_average.vertices[1])
        lh_count = lh_vertices.size
        rh_count = rh_vertices.size

        if lh_count + rh_count == 0:
            log.warning("Grand average STC appears empty; skipping label highlight plot.")
            return

        data = np.zeros((lh_count + rh_count, 1), dtype=float)
        hemi_code = 0 if label_obj.hemi == 'lh' else 1

        if hemi_code == 0:
            mask = np.isin(lh_vertices, label_obj.vertices)
            data[:lh_count, 0][mask] = 1.0
        else:
            mask = np.isin(rh_vertices, label_obj.vertices)
            data[lh_count:, 0][mask] = 1.0

        if not np.any(data):
            log.warning("Label %s has no vertices in fsaverage source space; skipping highlight.", label_name)
            return

        vertno_array = (lh_vertices if hemi_code == 0 else rh_vertices)
        peak_vertno = int(vertno_array[np.where(mask)[0][0]])
        peak_mni = mne.vertex_to_mni(
            [peak_vertno], hemis=hemi_code, subject='fsaverage', subjects_dir=subjects_dir
        )[0]

        method = str(config.get('source', {}).get('method', 'eLORETA')).upper()
        time_label = "Label highlight"
        if top_label.time_start_ms is not None and top_label.time_end_ms is not None:
            time_label = f"{top_label.time_start_ms:.1f}-{top_label.time_end_ms:.1f} ms"

        peak_info = {
            't': 1.0,
            'vertno': peak_vertno,
            'hemi_code': hemi_code,
            'mni_str': f"({peak_mni[0]:.1f}, {peak_mni[1]:.1f}, {peak_mni[2]:.1f})",
            'region': label_name,
            'size': int(np.count_nonzero(data)),
            'subtitle_label': f"Label highlight | {label_name} (p={top_label.p_value:.4f})",
            'subtitle_label_detail': (
                f"Window {time_label} | solver {method}"
            ),
            'footer': "fsaverage | label highlight snapshot",
        }

        clim = dict(kind='value', lims=[0.0, 0.5, 1.0])

        _compose_source_figure(
            data,
            stc_grand_average.vertices,
            0.0,
            0.0,
            'Reds',
            clim,
            'Label membership (a.u.)',
            output_path,
            analysis_name,
            config,
            None,
            time_label,
            peak_info,
            subjects_dir,
        )
    except Exception as e:
        log.error(f"Failed to render label highlight snapshot: {e}", exc_info=True)


def plot_grand_average_peak(stc_grand_average, config, output_dir, stats_results=None, summary_path=None):
    """Render a single-time-point peak snapshot derived from label or vertex stats."""
    try:
        analysis_name = config['analysis_name']
        subjects_dir, _ = _check_fsaverage_path()
        stats_cfg = config.get('stats', {}) or {}
        method = str(config.get('source', {}).get('method', 'eLORETA')).upper()

        summary = None
        if summary_path and Path(summary_path).exists():
            summary = parse_label_summary(Path(summary_path))

        time_s: Optional[float] = None
        caption_info = {}

        if summary and summary.significant:
            top_label = summary.significant[0]
            if top_label.peak_ms is not None:
                time_s = top_label.peak_ms / 1000.0
            elif top_label.time_start_ms is not None:
                time_s = top_label.time_start_ms / 1000.0
            caption_info.update({
                'reason': 'label',
                'label_name': top_label.name,
                'label_p': top_label.p_value,
                'label_time': (top_label.time_start_ms, top_label.time_end_ms),
                'label_peak_ms': top_label.peak_ms,
            })

        if time_s is None and stats_results is not None:
            try:
                t_obs, clusters, cluster_p_values, _ = stats_results
                if clusters is not None and cluster_p_values is not None and len(cluster_p_values):
                    alpha = float(stats_cfg.get('cluster_alpha', 0.05))
                    sig_idx = np.where(cluster_p_values < alpha)[0]
                    if sig_idx.size:
                        best_idx = sig_idx[np.argmin(cluster_p_values[sig_idx])]
                        cluster_mask = clusters[best_idx]
                        coords = np.where(cluster_mask)
                        if coords[0].size:
                            abs_vals = np.abs(t_obs[coords])
                            peak_pos = int(np.argmax(abs_vals))
                            time_index = int(coords[0][peak_pos])
                            time_s = float(stc_grand_average.times[time_index])
                            caption_info.update({
                                'reason': 'vertex',
                                'cluster_p': float(cluster_p_values[best_idx]),
                            })
            except Exception as exc:
                log.debug("Vertex peak selection failed: %s", exc)

        if time_s is None:
            aw = stats_cfg.get('analysis_window') or []
            if isinstance(aw, (list, tuple)) and len(aw) == 2:
                try:
                    start = float(aw[0])
                    end = float(aw[1])
                    time_s = (start + end) / 2.0
                    caption_info.update({'reason': 'window', 'window': (start, end)})
                except Exception:
                    pass

        if time_s is None and stc_grand_average.times.size:
            idx = int(np.nanargmax(np.abs(stc_grand_average.data)))
            _, time_idx = np.unravel_index(idx, stc_grand_average.data.shape)
            time_s = float(stc_grand_average.times[time_idx])
            caption_info.update({'reason': 'global'})

        if time_s is None:
            log.info("Unable to determine peak time; skipping peak snapshot.")
            return

        times = stc_grand_average.times
        if times.size == 0:
            return
        nearest_idx = int(np.argmin(np.abs(times - time_s)))
        time_s = float(times[nearest_idx])

        data_vector = stc_grand_average.data[:, [nearest_idx]].copy()
        data_vec_flat = data_vector.reshape(-1)
        abs_vals = np.abs(data_vec_flat)
        vmax_abs = float(np.nanpercentile(abs_vals, 98)) if abs_vals.size else 0.0
        if not np.isfinite(vmax_abs) or vmax_abs <= 0:
            vmax_abs = float(np.nanmax(abs_vals)) if abs_vals.size else 0.0
        if not np.isfinite(vmax_abs) or vmax_abs <= 0:
            log.info("Peak snapshot skipped due to flat data at selected time.")
            return

        clim = dict(kind='value', lims=[-vmax_abs, 0.0, vmax_abs])

        lh_n = len(stc_grand_average.vertices[0])
        peak_idx = int(np.nanargmax(np.abs(data_vec_flat)))
        hemi_code = 0 if peak_idx < lh_n else 1
        peak_vertno = int(
            stc_grand_average.vertices[0][peak_idx]
            if hemi_code == 0
            else stc_grand_average.vertices[1][peak_idx - lh_n]
        )
        peak_mni = mne.vertex_to_mni(
            [peak_vertno], hemis=hemi_code, subject='fsaverage', subjects_dir=subjects_dir
        )[0]

        reason = caption_info.get('reason', 'window')
        subtitle = "Peak snapshot"
        detail = f"Time={time_s*1000:.1f} ms | solver {method}"

        if reason == 'label' and caption_info.get('label_name'):
            label_name = caption_info['label_name']
            p_val = caption_info.get('label_p')
            subtitle = f"Peak snapshot | label {label_name} (p={p_val:.4f})"
            if caption_info.get('label_time'):
                start_ms, end_ms = caption_info['label_time']
                detail = (
                    f"Peak={caption_info.get('label_peak_ms', time_s*1000):.1f} ms | window {start_ms:.1f}-{end_ms:.1f} ms | solver {method}"
                )
        elif reason == 'vertex' and caption_info.get('cluster_p') is not None:
            p_val = caption_info['cluster_p']
            subtitle = f"Peak snapshot | vertex cluster (p={p_val:.4f})"
            detail = f"Time={time_s*1000:.1f} ms | solver {method}"
        elif reason == 'global':
            subtitle = "Peak snapshot | global max"
        else:
            subtitle = "Peak snapshot | analysis window midpoint"

        peak_info = {
            't': vmax_abs,
            'vertno': peak_vertno,
            'hemi_code': hemi_code,
            'mni_str': f"({peak_mni[0]:.1f}, {peak_mni[1]:.1f}, {peak_mni[2]:.1f})",
            'region': caption_info.get('label_name', 'N/A'),
            'size': int(np.count_nonzero(np.abs(data_vec_flat) > 0)),
            'subtitle_label': subtitle,
            'subtitle_label_detail': detail,
            'footer': "fsaverage | peak-moment snapshot",
        }

        output_path = output_dir / f"{analysis_name}_grand_average_peak.png"
        _compose_source_figure(
            data_vector,
            stc_grand_average.vertices,
            time_s,
            0.0,
            'RdBu_r',
            clim,
            'Contrast amplitude (a.u.)',
            output_path,
            analysis_name,
            config,
            None,
            f"{time_s*1000:.1f} ms",
            peak_info,
            subjects_dir,
        )

        meta = {
            'time_ms': time_s * 1000.0,
            'reason': reason,
            'method': method,
        }
        if caption_info.get('label_name'):
            meta['label_name'] = caption_info['label_name']
            meta['label_p'] = caption_info.get('label_p')
            meta['label_peak_ms'] = caption_info.get('label_peak_ms')
            meta['label_window_ms'] = caption_info.get('label_time')
        if caption_info.get('cluster_p') is not None:
            meta['cluster_p'] = caption_info['cluster_p']
        if caption_info.get('window'):
            window = caption_info['window']
            meta['analysis_window_ms'] = (float(window[0]) * 1000.0, float(window[1]) * 1000.0)

        meta_path = output_dir / f"{analysis_name}_grand_average_peak.json"
        try:
            import json as _json

            meta_path.write_text(_json.dumps(meta, indent=2))
        except Exception as exc:
            log.debug("Failed to write peak snapshot metadata: %s", exc)
    except Exception as e:
        log.error(f"Failed to render peak snapshot: {e}", exc_info=True)

def _choose_time_and_clim(stc_sum):
    """Choose a time point and clims for plotting the summary STC."""
    # Find the time point corresponding to the global max |activation|
    flat_idx = np.argmax(np.abs(stc_sum.data))
    vert_idx, time_idx = np.unravel_index(flat_idx, stc_sum.data.shape)
    initial_time = stc_sum.times[time_idx]

    # Use robust quantiles for color limits
    data_abs = np.abs(stc_sum.data)
    non_zero_data = data_abs[data_abs > 0]
    if len(non_zero_data) > 0:
        clim = dict(kind="value", lims=np.quantile(non_zero_data, [0.95, 0.975, 0.999]))
    else:
        clim = dict(kind="value", lims=[0, 0.5, 1]) # Fallback for empty data
        
    return initial_time, clim


def plot_contrast_erp(grand_average, stats_results, config, output_dir, ch_names):
    """
    Plot ERP figures for all significant clusters; preserve legacy single-file
    output for the top-ranked cluster.
    """
    t_obs, clusters, cluster_p_values, _ = stats_results
    alpha = config['stats']['cluster_alpha']

    sig_cluster_indices = np.where(cluster_p_values < alpha)[0]
    if not sig_cluster_indices.size:
        log.info("No significant clusters found. Skipping ERP plot.")
        return

    ordered = sig_cluster_indices[np.argsort(cluster_p_values[sig_cluster_indices])]
    p_order = np.argsort(cluster_p_values)

    for rank, idx in enumerate(ordered, start=1):
        mask = clusters[idx]
        ch_mask = mask.any(axis=0)
        time_mask = mask.any(axis=1)

        cluster_ch_names = [ch_names[i] for i, in_cluster in enumerate(ch_mask) if in_cluster]
        cluster_times = grand_average.times[time_mask]
        tmin, tmax = cluster_times[0], cluster_times[-1]

        picks = mne.pick_channels(grand_average.info['ch_names'], include=cluster_ch_names)
        roi_data = grand_average.get_data(picks=picks).mean(axis=0)

        cluster_rank = int(np.where(p_order == idx)[0][0]) + 1

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(grand_average.times * 1000, roi_data * 1e6, lw=2, label='Grand Average Contrast')
        ax.axvspan(tmin * 1000, tmax * 1000, alpha=0.2, color='red',
                   label=f'Cluster #{cluster_rank} (p={cluster_p_values[idx]:.3f})')

        ax.axhline(0, ls='--', color='black', lw=1)
        ax.axvline(0, ls='-', color='black', lw=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(
            f"Contrast: {config['contrast']['name']}\n(Channels averaged over Cluster #{cluster_rank})"
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.legend(loc='lower left')
        plt.tight_layout()

        out_num = output_dir / f"{config['analysis_name']}_erp_cluster_{cluster_rank}.png"
        fig.savefig(out_num, dpi=300)
        if rank == 1:
            legacy = output_dir / f"{config['analysis_name']}_erp_cluster.png"
            fig.savefig(legacy, dpi=300)
        log.info(f"Saved ERP cluster plot to {out_num}")
        plt.close(fig)


def plot_t_value_topomap(grand_average, stats_results, config, output_dir, ch_names):
    """
    Plot topomap figures for all significant clusters; preserve legacy single-file
    output for the top-ranked cluster.

    Handles both full-scalp and ROI-restricted analyses.
    """
    t_obs, clusters, cluster_p_values, _ = stats_results
    alpha = config['stats']['cluster_alpha']

    sig_cluster_indices = np.where(cluster_p_values < alpha)[0]
    if not sig_cluster_indices.size:
        log.info("No significant clusters found. Skipping topomap plot.")
        return

    ordered = sig_cluster_indices[np.argsort(cluster_p_values[sig_cluster_indices])]
    p_order = np.argsort(cluster_p_values)

    # Reshape t-values to (n_times, n_channels)
    t_obs_tc = t_obs.reshape(len(grand_average.times), len(ch_names))

    # Check if ROI restriction was used
    roi_restricted = len(ch_names) != len(grand_average.info['ch_names'])

    if roi_restricted:
        # Create a subset info object that matches the ROI channels
        roi_picks = mne.pick_channels(grand_average.info['ch_names'], include=ch_names, ordered=True)
        roi_info = mne.pick_info(grand_average.info, roi_picks)
        log.info(f"ROI-restricted analysis: plotting {len(ch_names)} channels on topomap")
    else:
        roi_info = grand_average.info

    for rank, idx in enumerate(ordered, start=1):
        time_mask = clusters[idx].any(axis=1)
        cluster_times = grand_average.times[time_mask]
        tmin, tmax = cluster_times[0], cluster_times[-1]

        t_topo = t_obs_tc[time_mask, :].mean(axis=0)
        ch_mask = clusters[idx].any(axis=0)

        # Use p-rank for cluster numbering to match text report
        cluster_rank = int(np.where(p_order == idx)[0][0]) + 1
        fig, ax = plt.subplots(figsize=(6.5, 5.5))

        # Add ROI info to title if restricted
        roi_note = " (ROI-restricted)" if roi_restricted else ""
        title = (f"T-Values ({tmin*1000:.0f} - {tmax*1000:.0f} ms){roi_note}\n"
                 f"Cluster #{cluster_rank}, p = {cluster_p_values[idx]:.4f}")

        mask_params = dict(marker='o', markerfacecolor='none', markeredgecolor='k',
                           linewidth=1.5, markersize=7)
        im, _ = mne.viz.plot_topomap(
            t_topo, roi_info, axes=ax, show=False, cmap='RdBu_r',
            contours=6, sensors=False, mask=ch_mask, mask_params=mask_params,
        )

        # Annotate the peak channel within the cluster (largest |t|)
        try:
            if _find_topomap_coords is None:
                raise RuntimeError("no _find_topomap_coords available")

            all_names = grand_average.info['ch_names']
            picks_for_pos = [all_names.index(nm) for nm in ch_names]
            pos = _find_topomap_coords(grand_average.info, picks_for_pos)

            t_abs = np.abs(t_topo.copy()); t_abs[~ch_mask] = 0.0
            peak_idx = int(np.argmax(t_abs)); peak_name = ch_names[peak_idx]
            ax.scatter(pos[peak_idx, 0], pos[peak_idx, 1], s=140,
                       facecolors='yellow', edgecolors='k', linewidths=2, zorder=11)
            ax.text(pos[peak_idx, 0] + 0.02, pos[peak_idx, 1] + 0.02, peak_name,
                    fontsize=9, color='k', weight='bold', zorder=12)
        except Exception as e:
            log.warning(f"Peak sensor annotation skipped ({e})")

        cbar = fig.colorbar(im, ax=ax); cbar.set_label("T-Value"); ax.set_title(title)

        out_num = output_dir / f"{config['analysis_name']}_topomap_cluster_{rank}.png"
        fig.savefig(out_num, dpi=300)
        if rank == 1:
            legacy = output_dir / f"{config['analysis_name']}_topomap_cluster.png"
            fig.savefig(legacy, dpi=300)
        log.info(f"Saved T-value topomap to {out_num}")
        plt.close(fig)


def plot_condition_erps_rois(ga_cond_A, ga_cond_B, config, output_dir):
    """
    Plot condition ERPs separately for P1 (Oz), N1 (bilateral), and P3b (midline).
    Saves three PNGs in output_dir:
      <analysis_name>_roi_P1_erp.png
      <analysis_name>_roi_N1_erp.png
      <analysis_name>_roi_P3b_erp.png
    """
    try:
        import mne
        analysis_name = config["analysis_name"]

        def _pair_label(cond_obj: dict, fallback: str) -> str:
            # Try to derive compact digit-pair label like '14' from config
            try:
                csn = cond_obj.get("condition_set_name", "")
                m = re.search(r"PRIME_(\d{1,2})_LAND_(\d{1,2})", csn)
                if m:
                    return f"{m.group(1)}{m.group(2)}"
                name_str = cond_obj.get("name", fallback)
                digits = re.findall(r"(?<!\d)(\d{1,2})(?!\d)", name_str)
                if len(digits) >= 2:
                    return f"{digits[0]}{digits[1]}"
            except Exception:
                pass
            return fallback

        cond_A_fallback = config.get("contrast", {}).get("condition_A", {}).get("name", "Condition A")
        cond_B_fallback = config.get("contrast", {}).get("condition_B", {}).get("name", "Condition B")
        cond_A_label = _pair_label(config.get("contrast", {}).get("condition_A", {}), cond_A_fallback)
        cond_B_label = _pair_label(config.get("contrast", {}).get("condition_B", {}), cond_B_fallback)

        rois = [
            ("P1", "Oz", "P1 (Oz)", "P1"),
            ("N1", "bilateral", "N1 (bilateral)", "N1"),
            ("P3b", "midline", "P3b (midline)", "P3b")
        ]

        times_ms = ga_cond_A.times * 1000.0
        colors = ["#1f77b4", "#d62728"]  # blue/red

        for comp, region, title, tag in rois:
            group = ELECTRODE_GROUPS.get(comp, {}).get(region, {})
            chs = group.get("electrodes", [])
            if not chs:
                continue
            picks_A = mne.pick_channels(ga_cond_A.info["ch_names"], include=chs)
            picks_B = mne.pick_channels(ga_cond_B.info["ch_names"], include=chs)
            if len(picks_A) == 0 or len(picks_B) == 0:
                continue
            yA = ga_cond_A.get_data(picks=picks_A).mean(axis=0) * 1e6
            yB = ga_cond_B.get_data(picks=picks_B).mean(axis=0) * 1e6

            fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=150)
            ax.plot(times_ms, yA, lw=1.8, color=colors[0], label=cond_A_label)
            ax.plot(times_ms, yB, lw=1.8, color=colors[1], label=cond_B_label)
            ax.axhline(0, ls='--', color='black', lw=0.8)
            ax.axvline(0, ls='-', color='black', lw=0.8)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Time (ms)", fontsize=9)
            ax.set_ylabel("Amplitude (µV)", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.legend(loc='upper right', fontsize=8, frameon=False)

            out_path = Path(output_dir) / f"{analysis_name}_roi_{tag}_erp.png"
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            log.info(f"Saved ROI condition ERP plot to {out_path}")
    except Exception as e:
        log.warning(f"Failed to plot ROI condition ERPs: {e}")
