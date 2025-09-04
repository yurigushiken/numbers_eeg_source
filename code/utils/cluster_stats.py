"""
Cluster Statistics Utilities
"""
import logging
import mne
import numpy as np
from scipy.stats import t as t_dist
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform

log = logging.getLogger()


def _distance_adjacency(info, threshold=0.04):
    """
    Computes adjacency matrix based on channel positions and a distance threshold.
    """
    picks = mne.pick_types(info, meg=False, eeg=True)
    ch_names = [info['ch_names'][p] for p in picks]
    pos = np.array([info['chs'][p]['loc'][:3] for p in picks])

    if not np.isfinite(pos).all() or np.allclose(pos, 0):
        raise RuntimeError(
            "Channel positions are invalid or absent. "
            "Please ensure a montage is set on your data."
        )

    # Calculate pairwise distances and apply threshold
    dist_matrix = squareform(pdist(pos))
    adjacency_matrix = dist_matrix <= threshold
    
    # Convert to sparse matrix format expected by MNE
    adjacency = csr_matrix(adjacency_matrix)
    return adjacency, ch_names


def run_sensor_cluster_test(contrasts, config):
    """
    Runs a sensor-space cluster permutation test on a list of evoked contrasts.
    """
    # 1. Prepare data for MNE stats function
    log.info("Preparing data for sensor-space cluster analysis...")
    X = np.array([c.get_data() for c in contrasts])
    X = X.transpose(0, 2, 1)  # (n_subjects, n_times, n_channels)
    log.info(f"Data stacked into shape: {X.shape}")

    # 2. Get channel adjacency
    log.info("Finding channel adjacency...")
    info_ref = contrasts[0].info
    conn_cfg = config['stats'].get('connectivity', 'eeg')

    if isinstance(conn_cfg, str):
        adjacency, ch_names = mne.channels.find_ch_adjacency(info_ref, ch_type=conn_cfg)
    elif isinstance(conn_cfg, dict) and conn_cfg.get('method') == 'distance':
        adjacency, ch_names = _distance_adjacency(
            info_ref, threshold=conn_cfg.get('threshold', 0.04)
        )
    else:
        raise ValueError(f"Invalid connectivity configuration: {conn_cfg!r}")

    # 3. Define the statistical threshold
    p_threshold = config['stats']['p_threshold']
    t_threshold = t_dist.ppf(1.0 - p_threshold / 2., len(contrasts) - 1)
    log.info(f"Calculated t-threshold for cluster formation: {t_threshold:.3f} (for p < {p_threshold})")

    # 4. Run the cluster permutation test
    log.info(f"Running cluster permutation test with {config['stats']['n_permutations']} permutations...")
    stat_results = mne.stats.spatio_temporal_cluster_1samp_test(
        X,
        adjacency=adjacency,
        n_permutations=config['stats']['n_permutations'],
        threshold=t_threshold,
        tail=config['stats']['tail'],
        n_jobs=-1,
        seed=config['stats'].get('seed', None),
        out_type='mask',  # Explicitly request boolean masks for compatibility
        verbose=True
    )
    log.info("Sensor cluster analysis complete.")

    t_obs, clusters, cluster_p_values, H0 = stat_results
    return (t_obs, clusters, cluster_p_values, H0), ch_names


def run_source_cluster_test(stcs, fsaverage_src, config):
    """
    Runs a spatio-temporal cluster 1-sample t-test on source-space contrasts.
    """
    if len(stcs) < 2:
        raise ValueError("Cannot run source cluster test with fewer than 2 subjects.")

    log.info("Preparing data for source-space cluster analysis...")
    # Stack data into a (n_subjects, n_times, n_vertices) array
    # Data is transposed from (n_vertices, n_times) to (n_times, n_vertices)
    X = np.stack([stc.data.T for stc in stcs], axis=0)
    log.info(f"Data stacked into shape: {X.shape}")

    # Get source space adjacency
    log.info("Calculating source space adjacency for fsaverage...")
    source_adjacency = mne.spatial_src_adjacency(fsaverage_src)

    # Determine thresholding strategy: TFCE or single-threshold
    n_subjects = len(stcs)
    degrees_of_freedom = n_subjects - 1
    method = config['stats'].get('method', 'threshold')
    tail = int(config['stats']['tail'])
    if method == 'tfce':
        # Use TFCE; older MNE versions expect a dict with 'start' and 'step'.
        # The sign of 'step' must match the tail: negative for tail=-1, positive for tail=+1.
        tfce_cfg = config['stats'].get('tfce', {})
        tfce_start = float(tfce_cfg.get('start', 0.0))
        tfce_step = abs(float(tfce_cfg.get('step', 0.2)))
        if tail < 0:
            tfce_step = -abs(tfce_step)
        elif tail > 0:
            tfce_step = abs(tfce_step)
        # For tail == 0, keep positive step; MNE will evaluate both tails internally.
        threshold = dict(start=tfce_start, step=tfce_step)
        log.info(f"Using TFCE thresholding for cluster formation (start={tfce_start}, step={tfce_step}).")
    else:
        log.info("Using single-threshold method for cluster formation.")
        p_threshold = config['stats']['p_threshold']
        if tail == 0:
            t_threshold = t_dist.ppf(1.0 - p_threshold / 2., df=degrees_of_freedom)
        else:
            t_threshold = t_dist.ppf(1.0 - p_threshold, df=degrees_of_freedom)
            if tail < 0:
                t_threshold = -abs(t_threshold)
            else:
                t_threshold = abs(t_threshold)
        threshold = t_threshold
        log.info(
            f"Calculated t-threshold for cluster formation: {t_threshold:.3f} "
            f"(for p < {p_threshold}, tail={tail})"
        )

    # Optionally restrict to ROI vertices from labels (reduces spatial search space)
    roi_cfg = config['stats'].get('roi', None)
    roi_mask = None
    roi_vertices_lh = None
    roi_vertices_rh = None
    keep_idx = None  # mapping from local (ROI-cropped) vertex indices to global src indices

    # Clear any stale mapping from prior runs
    try:
        config['stats'].pop('_keep_idx', None)
    except Exception:
        pass
    if roi_cfg:
        try:
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[2]
            subjects_dir = project_root / 'data' / 'all' / 'fs_subjects_dir'
            parc = roi_cfg.get('parc', 'aparc')
            wanted = set([s.lower() for s in roi_cfg.get('labels', [])])
            if wanted:
                all_labels = mne.read_labels_from_annot('fsaverage', parc=parc, subjects_dir=subjects_dir, verbose=False)
                def base_name(name: str) -> str:
                    name = name.lower()
                    if name.endswith('-lh') or name.endswith('-rh'):
                        return name[:-3]
                    return name
                lh_verts_set = set()
                rh_verts_set = set()
                for lab in all_labels:
                    if base_name(lab.name) in wanted:
                        if lab.hemi == 'lh':
                            lh_verts_set.update(lab.vertices.tolist())
                        elif lab.hemi == 'rh':
                            rh_verts_set.update(lab.vertices.tolist())
                lh_verts = np.array(sorted(lh_verts_set), dtype=int) if lh_verts_set else np.array([], dtype=int)
                rh_verts = np.array(sorted(rh_verts_set), dtype=int) if rh_verts_set else np.array([], dtype=int)

                # Map label vertex ids to src vertex indices order
                src_lh = np.array(fsaverage_src[0]['vertno'])
                src_rh = np.array(fsaverage_src[1]['vertno'])
                lh_sel = np.isin(src_lh, lh_verts)
                rh_sel = np.isin(src_rh, rh_verts)
                roi_mask = np.concatenate([lh_sel, rh_sel])
                # Save selected vertex ids for plotting later
                roi_vertices_lh = src_lh[lh_sel]
                roi_vertices_rh = src_rh[rh_sel]

                n_keep = int(roi_mask.sum())
                log.info(f"ROI restriction active (parc={parc}): keeping {n_keep} of {roi_mask.size} vertices.")
                if n_keep == 0:
                    log.warning("ROI selection resulted in zero vertices; disabling ROI restriction.")
                    roi_mask = None
        except Exception as e:
            log.warning(f"Failed to apply ROI restriction: {e}. Proceeding without ROI.")

    # Optionally crop the time window used for clustering to stats.analysis_window
    # Build time indices from the first STC (all resampled equally upstream)
    times = stcs[0].times
    aw = (config.get('stats') or {}).get('analysis_window')
    if aw and len(aw) == 2:
        tmin = float(aw[0])
        tmax = float(aw[1])
        log.info(f"Restricting source clustering to analysis_window {tmin:.3f}-{tmax:.3f}s")
    else:
        tmin = float(times[0])
        tmax = float(times[-1])
        log.info("No analysis_window provided for source stats; using full range.")
    time_mask = (times >= tmin) & (times <= tmax)
    if time_mask.sum() == 0:
        log.warning("Time mask is empty after applying tmin/tmax; using full window instead.")
        time_mask = slice(None)

    X_cropped = X[:, time_mask, :]
    if roi_mask is not None:
        keep_idx = np.flatnonzero(roi_mask.astype(bool))
        X_cropped = X_cropped[:, :, keep_idx]
        # Convert to CSR before subsetting rows/cols
        source_adjacency = source_adjacency.tocsr()[keep_idx][:, keep_idx]
        # Persist mapping for downstream anatomical reporting
        try:
            config['stats']['_keep_idx'] = keep_idx.tolist()
        except Exception:
            pass
    log.info(f"Clustering over cropped window {tmin:.3f}s to {tmax:.3f}s with data shape {X_cropped.shape}")

    # Run the cluster permutation test on the cropped data
    log.info(f"Running source cluster permutation test with {config['stats']['n_permutations']} permutations...")
    stat_results = mne.stats.spatio_temporal_cluster_1samp_test(
        X_cropped,
        adjacency=source_adjacency,
        threshold=threshold,
        tail=tail,
        n_permutations=config['stats']['n_permutations'],
        out_type='indices',  # Ensure output is indices for plotting
        max_step=1,          # Recommended for spatio-temporal clustering
        n_jobs=int(config['stats'].get('n_jobs', 4)),
        seed=config['stats'].get('seed', None),
        buffer_size=int(config['stats'].get('buffer_size', 1000)),
        verbose=True
    )
    log.info("Source cluster analysis complete.")
    # Persist ROI vertices for plotting if used
    if roi_vertices_lh is not None and roi_vertices_rh is not None:
        try:
            config['stats']['_roi_vertices'] = [roi_vertices_lh.tolist(), roi_vertices_rh.tolist()]
        except Exception:
            pass
    return stat_results


def run_label_timecourse_cluster_test(stcs, fsaverage_src, config):
    """
    Run a 1D time-cluster permutation test on label-averaged time courses.

    - Extract label time courses on fsaverage with mode='mean_flip'.
    - Average across specified labels to one ROI time course per subject.
    - Crop to YAML tmin/tmax and cluster across time.
    """
    import mne
    log.info("Preparing label-averaged time courses for clustering...")

    lt_cfg = config['stats'].get('label_timeseries', {})
    parc = lt_cfg.get('parc', 'aparc')
    labels_wanted = lt_cfg.get('labels') or config['stats'].get('roi', {}).get('labels', [])
    if not labels_wanted:
        raise ValueError("No labels provided for label time-series test (stats.label_timeseries.labels).")
    labels_wanted = [s.lower() for s in labels_wanted]

    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    subjects_dir = project_root / 'data' / 'all' / 'fs_subjects_dir'
    all_labels = mne.read_labels_from_annot('fsaverage', parc=parc, subjects_dir=subjects_dir, verbose=False)
    def base_name(name: str) -> str:
        name = name.lower()
        if name.endswith('-lh') or name.endswith('-rh'):
            return name[:-3]
        return name
    selected_labels = [lab for lab in all_labels if base_name(lab.name) in labels_wanted]
    if not selected_labels:
        raise ValueError("Selected labels not found on fsaverage. Check parc and label names.")

    # Build (n_subjects, n_times) matrix
    X = []
    for stc in stcs:
        ltc = mne.extract_label_time_course(
            stc, selected_labels, src=fsaverage_src, mode='mean_flip', allow_empty=True, verbose=False
        )  # shape (n_labels, n_times)
        ts = ltc.mean(axis=0)
        X.append(ts)
    X = np.vstack(X)  # (n_subjects, n_times)

    # Build time crop
    times = stcs[0].times
    aw = (config.get('stats') or {}).get('analysis_window')
    if aw and len(aw) == 2:
        tmin = float(aw[0])
        tmax = float(aw[1])
        log.info(f"Label time-series: analysis_window {tmin:.3f}-{tmax:.3f}s")
    else:
        tmin = float(times[0])
        tmax = float(times[-1])
    time_mask = (times >= tmin) & (times <= tmax)
    if time_mask.sum() == 0:
        log.warning("Time mask is empty for label TS; using full window.")
        time_mask = slice(None)
        times_c = times
        X_c = X
    else:
        times_c = times[time_mask]
        X_c = X[:, time_mask]

    # Compute signed threshold
    n_subjects = X_c.shape[0]
    dof = n_subjects - 1
    p_threshold = config['stats']['p_threshold']
    tail = int(config['stats']['tail'])
    if tail == 0:
        thr = t_dist.ppf(1.0 - p_threshold / 2., df=dof)
    else:
        thr = t_dist.ppf(1.0 - p_threshold, df=dof)
        thr = -abs(thr) if tail < 0 else abs(thr)
    log.info(
        f"Running 1D time clustering over labels {labels_wanted} with threshold {thr:.3f}, "
        f"n_permutations={config['stats']['n_permutations']}, tail={tail}"
    )

    # 1D cluster test over time
    stat_results = mne.stats.permutation_cluster_1samp_test(
        X_c, threshold=thr, tail=tail, n_permutations=config['stats']['n_permutations'],
        out_type='mask', verbose=True
    )

    return stat_results, times_c, X_c.mean(axis=0), [lab.name for lab in selected_labels]
