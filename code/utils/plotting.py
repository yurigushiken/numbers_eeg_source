"""
Plotting Utilities for Cluster Analysis
"""
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import mne
from pathlib import Path
import imageio.v2 as iio
from nilearn import plotting
import nibabel as nib
import re
from code.utils.electrodes import ELECTRODE_GROUPS


log = logging.getLogger()


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


def export_mask_to_glass_brain(mask_stc, inv_fname, output_dir, analysis_name, subjects_dir):
    try:
        import mne
        import nibabel as nib
        from nilearn import plotting
        inv = mne.minimum_norm.read_inverse_operator(str(inv_fname))
        src = inv["src"]

        # Only volumetric STCs can be turned into NIfTI directly
        if not isinstance(mask_stc, mne.VolSourceEstimate):
            log.warning("mask_stc is a surface SourceEstimate; skipping glass-brain NIfTI export.")
            return

        # Volumetric path:
        img = mask_stc.as_volume(src=src, dest="mri", mri_resolution=True)
        out_nii = output_dir / f"{analysis_name}_union_mask.nii.gz"
        nib.save(img, out_nii)
        log.info(f"Saved union-mask NIfTI to {out_nii}")

        display = plotting.plot_glass_brain(
            str(out_nii), display_mode="lyrz", threshold=0.5, colorbar=True,
            plot_abs=False, title="Significant Clusters (union mask)",
        )
        out_png = output_dir / f"{analysis_name}_glass_brain.png"
        display.savefig(out_png, dpi=200, facecolor="black")
        display.close()
        log.info(f"Saved glass-brain snapshot to {out_png}")

    except Exception as e:
        log.error(f"Failed to generate glass brain plot: {e}")


def plot_source_clusters(stats_results, stc_grand_average, config, output_dir):
    """
    Builds a summary STC of significant clusters and plots it on the surface.
    """
    try:
        from mne.stats import summarize_clusters_stc
        analysis_name = config['analysis_name']
        subjects_dir, _ = _check_fsaverage_path()
        if subjects_dir is None:
            return

        log.info("Summarizing significant clusters for plotting...")
        
        # If ROI restriction was used during clustering, the returned cluster vertex
        # indices refer to the ROI-subset vertex list, not the full fsaverage space.
        # Use the ROI vertices to avoid shape mismatches; otherwise fall back to full.
        roi_vertices = None
        try:
            roi_vertices = config.get('stats', {}).get('_roi_vertices')
            if roi_vertices is not None and len(roi_vertices) == 2:
                roi_vertices = [np.array(roi_vertices[0]), np.array(roi_vertices[1])]
        except Exception:
            roi_vertices = None

        vertices_for_summary = roi_vertices if roi_vertices is not None else stc_grand_average.vertices

        stc_sum = summarize_clusters_stc(
            stats_results,
            p_thresh=config['stats']['cluster_alpha'],
            tstep=stc_grand_average.tstep,
            tmin=stc_grand_average.tmin,
            subject="fsaverage",
            vertices=vertices_for_summary,
        )

        # Note: summarize_clusters_stc returns vertices matching the input 'vertices'.
        # For ROI-restricted clustering, this results in an STC defined on the ROI mesh.
        # This is acceptable for plotting and prevents vertex mismatch errors.

        if stc_sum.data.max() == 0:
            log.warning("No significant clusters found to plot after summarizing. Skipping plot generation.")
            return

        log.info(f"Plotting summary of significant clusters.")
        
        # Use helper to pick an informative time point and robust color limits
        initial_time, clim = _choose_time_and_clim(stc_sum)

        # Render six tiles (one per view) and compose into a 2x3 grid for consistent spacing
        views = ["lateral", "medial", "frontal", "parietal", "dorsal", "ventral"]
        tile_paths = []
        for idx, view in enumerate(views):
            try:
                brain_tile = stc_sum.plot(
                    subject="fsaverage",
                    subjects_dir=subjects_dir,
                    hemi="both",
                    views=[view],
                    surface="inflated",
                    cortex="classic",
                    time_viewer=False,
                    colorbar=False,  # we'll draw a unified colorbar in the footer
                    show_traces=False,
                    background="white",
                    size=(900, 700),
                    time_label="",  # suppress on-tile time text to avoid occlusion
                    clim=clim,
                    initial_time=initial_time,
                )
                tmp_path = output_dir / f"{analysis_name}_tile_{view}.png"
                brain_tile.save_image(tmp_path)
                tile_paths.append(tmp_path)
                try:
                    brain_tile.close()
                except Exception:
                    pass
            except Exception as e_tile:
                log.warning(f"Failed to render view '{view}': {e_tile}")

        if not tile_paths:
            log.warning("No tiles were rendered; skipping composite image.")
            return

        # Compose tiles into a grid canvas with header/footer space for text
        # Load tiles and normalize to RGB (3 channels)
        tiles = []
        for p in tile_paths:
            img = iio.imread(str(p))
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            tiles.append(img)

        h, w = tiles[0].shape[:2]
        pad = 20
        grid_h = 2 * h + 3 * pad
        grid_w = 3 * w + 4 * pad
        header_px, footer_px = 260, 120
        full_h = header_px + grid_h + footer_px
        full_w = grid_w
        canvas = np.ones((full_h, full_w, 3), dtype=np.uint8) * 255
        # place tiles after header
        positions = [
            (header_px + pad, pad),
            (header_px + pad, pad + w + pad),
            (header_px + pad, pad + 2 * (w + pad)),
            (header_px + pad + h + pad, pad),
            (header_px + pad + h + pad, pad + w + pad),
            (header_px + pad + h + pad, pad + 2 * (w + pad)),
        ]
        for img, (yy, xx) in zip(tiles, positions):
            hh, ww = img.shape[:2]
            canvas[yy:yy+hh, xx:xx+ww] = img

        # Global annotations via matplotlib overlay
        try:
            n_total = int(np.sum([len(v) for v in stc_grand_average.vertices]))
            roi_vertices = config.get('stats', {}).get('_roi_vertices')
            n_keep = int(np.sum([len(v) for v in roi_vertices])) if roi_vertices is not None else n_total
        except Exception:
            n_total, n_keep = 0, 0
        contrast = config.get('contrast', {}).get('name', analysis_name)
        tail = config.get('stats', {}).get('tail')
        pthr = config.get('stats', {}).get('p_threshold')
        alpha = config.get('stats', {}).get('cluster_alpha')
        n_perm = config.get('stats', {}).get('n_permutations')
        roi_cfg = config.get('stats', {}).get('roi')
        roi_str = ",".join(roi_cfg.get('labels', [])) if roi_cfg else 'whole brain'
        title = f"{contrast} — ROI: {roi_str} — method=dSPM, pick_ori=normal"
        stats_line1 = f"t={initial_time*1000:.1f} ms, tail={tail}"
        stats_line2 = f"p_thr={pthr}, α={alpha}, perms={n_perm}"
        footer = f"fsaverage • dSPM (a.u.) • vertices {n_keep}/{n_total}"

        out_png_fname = output_dir / f"{analysis_name}_source_cluster.png"
        fig, ax = plt.subplots(figsize=(15, 9.5), dpi=300)
        ax.imshow(canvas)
        ax.axis('off')
        # Use figure-level text so it never overlaps the image grid
        fig.text(0.02, 0.99, title, ha='left', va='top', fontsize=20, color='black')
        # Add extra vertical spacing between title and stats lines
        fig.text(0.02, 0.955, stats_line1, ha='left', va='top', fontsize=14, color='black')
        fig.text(0.02, 0.93, stats_line2, ha='left', va='top', fontsize=14, color='black')
        fig.text(0.02, 0.03, footer, ha='left', va='bottom', fontsize=11, color='black')
        # Add clean per-view labels on the composite (avoid tile cropping)
        for view, (yy, xx) in zip(views, positions):
            x_frac = (xx + 8) / full_w
            # place label further down to avoid header/title overlap
            y_frac = 1.0 - (yy + 160) / full_h
            fig.text(x_frac, y_frac, view.title(), ha='left', va='top', fontsize=13, color='black')
        # Add a unified horizontal colorbar in the footer band
        try:
            vmin = float(clim.get('lims', [0, 0, 1])[0]) if isinstance(clim, dict) else None
            vmax = float(clim.get('lims', [0, 0, 1])[-1]) if isinstance(clim, dict) else None
            if vmin is None or vmax is None:
                vmin, vmax = 0.0, 1.0
            norm = Normalize(vmin=vmin, vmax=vmax)
            from matplotlib.cm import get_cmap
            cmap = get_cmap('hot')
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cax = fig.add_axes([0.35, 0.055, 0.3, 0.02])
            cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
            cb.set_label('dSPM (a.u.)', fontsize=10)
        except Exception as e_cb:
            log.warning(f"Failed to draw unified colorbar: {e_cb}")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.04)
        fig.savefig(out_png_fname, dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        log.info(f"Saved source cluster snapshot to {out_png_fname}")

    except Exception as e:
        log.error(f"Failed to generate source cluster plot: {e}")


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
    Plots the grand average contrast ERP, averaged over channels in the most
    significant cluster, with the cluster's time window shaded.
    """
    t_obs, clusters, cluster_p_values, _ = stats_results
    alpha = config['stats']['cluster_alpha']
    
    # Find significant clusters and select the one with the smallest p-value
    sig_cluster_indices = np.where(cluster_p_values < alpha)[0]
    if not sig_cluster_indices.size:
        log.info("No significant clusters found. Skipping ERP plot.")
        return

    # Select the most significant cluster
    most_sig_idx = sig_cluster_indices[cluster_p_values[sig_cluster_indices].argmin()]
    log.info(f"Plotting ERP for most significant cluster (p={cluster_p_values[most_sig_idx]:.4f})")

    # Get the boolean mask for the cluster (shape: n_times, n_channels)
    mask = clusters[most_sig_idx]
    
    # Find the channels and time points belonging to this cluster
    ch_mask = mask.any(axis=0)
    time_mask = mask.any(axis=1)
    
    cluster_ch_names = [ch_names[i] for i, in_cluster in enumerate(ch_mask) if in_cluster]
    cluster_times = grand_average.times[time_mask]
    tmin, tmax = cluster_times[0], cluster_times[-1]

    # Create a spatial selection for the channels in the cluster
    picks = mne.pick_channels(grand_average.info['ch_names'], include=cluster_ch_names)

    # Average the grand average data over the selected channels
    roi_data = grand_average.get_data(picks=picks).mean(axis=0)

    # Determine p-rank for stable cluster numbering by significance
    p_order = np.argsort(cluster_p_values)
    cluster_rank = int(np.where(p_order == most_sig_idx)[0][0]) + 1

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grand_average.times * 1000, roi_data * 1e6, lw=2, label='Grand Average Contrast')
    ax.axvspan(
        tmin * 1000,
        tmax * 1000,
        alpha=0.2,
        color='red',
        label=f'Cluster #{cluster_rank} (p={cluster_p_values[most_sig_idx]:.3f})',
    )
    
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

    # Save figure
    fname = output_dir / f"{config['analysis_name']}_erp_cluster.png"
    fig.savefig(fname, dpi=300)
    log.info(f"Saved ERP cluster plot to {fname}")
    plt.close(fig) # <-- THIS IS THE FIX


def plot_t_value_topomap(grand_average, stats_results, config, output_dir, ch_names):
    """
    Plots a topomap of the t-values, averaged over the time window of the most
    significant cluster.
    """
    t_obs, clusters, cluster_p_values, _ = stats_results
    alpha = config['stats']['cluster_alpha']

    sig_cluster_indices = np.where(cluster_p_values < alpha)[0]
    if not sig_cluster_indices.size:
        log.info("No significant clusters found. Skipping topomap plot.")
        return

    most_sig_idx = sig_cluster_indices[cluster_p_values[sig_cluster_indices].argmin()]
    log.info(f"Plotting topomap for most significant cluster (p={cluster_p_values[most_sig_idx]:.4f})")

    # Reshape t-values to (n_times, n_channels)
    t_obs_tc = t_obs.reshape(len(grand_average.times), len(ch_names))

    # Get time mask for the significant cluster
    time_mask = clusters[most_sig_idx].any(axis=1)
    cluster_times = grand_average.times[time_mask]
    tmin, tmax = cluster_times[0], cluster_times[-1]

    # Average t-values across the cluster's time window
    t_topo = t_obs_tc[time_mask, :].mean(axis=0)
    # Channel membership mask for the selected cluster (for highlighting)
    ch_mask = clusters[most_sig_idx].any(axis=0)

    # Create the plot
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    title = (f"T-Values ({tmin*1000:.0f} - {tmax*1000:.0f} ms)\n"
             f"Cluster #{most_sig_idx+1}, p = {cluster_p_values[most_sig_idx]:.4f}")

    # Highlight only channels belonging to the selected cluster
    mask_params = dict(marker='o', markerfacecolor='none', markeredgecolor='k',
                       linewidth=1.5, markersize=7)
    im, _ = mne.viz.plot_topomap(
        t_topo,
        grand_average.info,
        axes=ax,
        show=False,
        cmap='RdBu_r',
        contours=6,  # restore scalp isocontours
        sensors=False,
        mask=ch_mask,
        mask_params=mask_params,
    )

    # Annotate the peak channel within the cluster (largest |t|)
    try:
        # Map provided ch_names to indices in info
        all_names = grand_average.info['ch_names']
        picks_for_pos = [all_names.index(nm) for nm in ch_names]
        try:
            from mne.viz.topomap import _find_topomap_coords
        except Exception:  # pragma: no cover - version fallback
            from mne.viz.topomap import _find_topomap_coords  # best effort
        pos = _find_topomap_coords(grand_average.info, picks_for_pos)

        t_abs = np.abs(t_topo.copy())
        t_abs[~ch_mask] = 0.0
        peak_idx = int(np.argmax(t_abs))
        peak_name = ch_names[peak_idx]
        ax.scatter(pos[peak_idx, 0], pos[peak_idx, 1], s=140,
                   facecolors='yellow', edgecolors='k', linewidths=2, zorder=11)
        ax.text(pos[peak_idx, 0] + 0.02, pos[peak_idx, 1] + 0.02, peak_name,
                fontsize=9, color='k', weight='bold', zorder=12)
    except Exception as e:
        log.warning(f"Failed to annotate peak sensor: {e}")
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("T-Value")
    ax.set_title(title)
    
    # Save figure
    fname = output_dir / f"{config['analysis_name']}_topomap_cluster.png"
    fig.savefig(fname, dpi=300)
    log.info(f"Saved T-value topomap to {fname}")
    plt.close(fig) # <-- THIS IS THE FIX


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