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
from code.utils.electrodes import ELECTRODE_GROUPS


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
    """Resolve fsaverage strictly from data (no fallbacks)."""
    project_root = Path(__file__).resolve().parents[2]
    subjects_dir = project_root / "data" / "all" / "fs_subjects_dir"
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
    stc_data, vertices, tmin, tstep, colormap, clim, cbar_label, output_path,
    analysis_name, config, cluster_p_val, time_label, peak_info, subjects_dir
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
        brain = stc_to_plot.plot(
            subject='fsaverage', subjects_dir=subjects_dir, surface='inflated',
            hemi='both', views=[view], size=(800, 600), background='white',
            foreground='black', time_viewer=False, show_traces=False,
            colorbar=False, colormap=colormap, cortex='classic',
            smoothing_steps=10, clim=clim,
            initial_time=float(stc_to_plot.times[0]), transparent=False,
            alpha=1.0, title=""
        )
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
    subtitle_1 = (
        f"Most significant cluster (p={cluster_p_val:.3f}), averaged within {time_label} | "
        f"{peak_info['subtitle_1_stats']}"
    )
    subtitle_2 = (
        f"Peak t={peak_info['t']:.2f} at MNI {peak_info['mni_str']}, "
        f"region {peak_info['region']}, cluster size={peak_info['size']} vertices"
    )
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

        t_obs, clusters, cluster_p_values, _ = stats_results
        if not clusters or cluster_p_values is None:
            raise RuntimeError("No clusters returned from statistical analysis.")

        # --- 1. Find ALL Significant Clusters ---
        alpha = float(config['stats']['cluster_alpha'])
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
            full_vertices = [np.array(stc_grand_average.vertices[0]), np.array(stc_grand_average.vertices[1])]
            full_n = int(np.sum([len(v) for v in full_vertices]))
            
            t_map_full = t_map_local
            if keep_idx is not None and int(t_map_local.size) != full_n:
                keep_idx = np.asarray(keep_idx, dtype=int)
                full_vec = np.full(full_n, np.nan, dtype=float)
                full_vec[keep_idx] = t_map_local
                t_map_full = full_vec
            
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
            roi_labels = ", ".join(config.get('stats', {}).get('roi', {}).get('labels', [])) or 'whole brain'

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

    except Exception as e:
        log.error(f"Failed to generate source cluster plots: {e}", exc_info=True)
        # Do not abort pipeline on plotting failure
        return


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

    for rank, idx in enumerate(ordered, start=1):
        time_mask = clusters[idx].any(axis=1)
        cluster_times = grand_average.times[time_mask]
        tmin, tmax = cluster_times[0], cluster_times[-1]

        t_topo = t_obs_tc[time_mask, :].mean(axis=0)
        ch_mask = clusters[idx].any(axis=0)

        # Use p-rank for cluster numbering to match text report
        cluster_rank = int(np.where(p_order == idx)[0][0]) + 1
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        title = (f"T-Values ({tmin*1000:.0f} - {tmax*1000:.0f} ms)\n"
                 f"Cluster #{cluster_rank}, p = {cluster_p_values[idx]:.4f}")

        mask_params = dict(marker='o', markerfacecolor='none', markeredgecolor='k',
                           linewidth=1.5, markersize=7)
        im, _ = mne.viz.plot_topomap(
            t_topo, grand_average.info, axes=ax, show=False, cmap='RdBu_r',
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
