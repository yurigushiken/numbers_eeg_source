"""
Cluster Statistics Utilities
"""
import logging
import mne
import numpy as np
import yaml
from pathlib import Path
from scipy.stats import t as t_dist
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform

log = logging.getLogger()

def _resolve_fs_subjects_dir(project_root: Path) -> Path:
    candidates = [
        project_root / 'data' / 'fs_subjects_dir',
        project_root / 'data' / 'all' / 'fs_subjects_dir',
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]

def _load_sensor_roi_definitions():
    """
    Load sensor ROI definitions from the central config file.

    Returns
    -------
    dict
        Dictionary mapping ROI names to their channel lists and metadata.
    """
    project_root = Path(__file__).resolve().parents[2]
    roi_defs_path = project_root / 'configs' / 'sensor_roi_definitions.yaml'

    if not roi_defs_path.exists():
        log.warning(f"Sensor ROI definitions file not found at {roi_defs_path}")
        return {}

    try:
        with open(roi_defs_path, 'r') as f:
            roi_data = yaml.safe_load(f)
        return roi_data.get('sensor_rois', {})
    except Exception as e:
        log.warning(f"Failed to load sensor ROI definitions: {e}")
        return {}


def _resolve_sensor_roi_channels(roi_cfg, fallback_roi_channels=None):
    """
    Resolve sensor ROI configuration to a set of channel names.

    Supports both the new centralized ROI definitions and legacy inline definitions.

    Parameters
    ----------
    roi_cfg : dict
        ROI configuration from stats.roi in the YAML config.
    fallback_roi_channels : dict, optional
        Legacy roi_channels section from config (for backwards compatibility).

    Returns
    -------
    set
        Set of channel names comprising the ROI.
    """
    roi_channels = set()

    # Load centralized ROI definitions
    roi_definitions = _load_sensor_roi_definitions()

    # Process direct channel lists
    direct_channels = roi_cfg.get('channels') or []
    roi_channels.update([str(ch) for ch in direct_channels])

    # Process channel groups
    channel_groups = roi_cfg.get('channel_groups') or []
    for grp in channel_groups:
        # First, try centralized definitions
        if grp in roi_definitions:
            roi_def = roi_definitions[grp]
            # Handle nested channel_groups in definitions
            if 'channel_groups' in roi_def:
                for nested_grp in roi_def['channel_groups']:
                    if nested_grp in roi_definitions:
                        nested_channels = roi_definitions[nested_grp].get('channels', [])
                        roi_channels.update([str(ch) for ch in nested_channels])
                    else:
                        log.warning(f"Nested ROI group '{nested_grp}' not found in sensor_roi_definitions.yaml")
            # Handle direct channels in definitions
            if 'channels' in roi_def:
                roi_channels.update([str(ch) for ch in roi_def['channels']])
        # Fallback to legacy roi_channels from config
        elif fallback_roi_channels and grp in fallback_roi_channels:
            channels_from_group = fallback_roi_channels.get(grp, [])
            roi_channels.update([str(ch) for ch in channels_from_group])
        else:
            log.warning(f"ROI channel group '{grp}' not found in sensor_roi_definitions.yaml or config.roi_channels")

    return roi_channels


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
    data_ch_names = contrasts[0].ch_names
    conn_cfg = config['stats'].get('connectivity', 'eeg')

    if isinstance(conn_cfg, str):
        adjacency, ch_names = mne.channels.find_ch_adjacency(info_ref, ch_type=conn_cfg)
    elif isinstance(conn_cfg, dict) and conn_cfg.get('method') == 'distance':
        adjacency, ch_names = _distance_adjacency(
            info_ref, threshold=conn_cfg.get('threshold', 0.04)
        )
    else:
        raise ValueError(f"Invalid connectivity configuration: {conn_cfg!r}")

    # Ensure data axis order matches adjacency order (drop non-adjacent channels if present)
    missing = [ch for ch in ch_names if ch not in data_ch_names]
    if missing:
        raise RuntimeError(
            "Adjacency channels missing from evoked data. "
            f"Missing: {missing}"
        )
    picks = mne.pick_channels(data_ch_names, include=ch_names, ordered=True)
    if len(picks) != X.shape[2]:
        log.debug(
            "Cropping sensor data to adjacency picks "
            f"({len(picks)} of {X.shape[2]} channels retained)."
        )
    elif not np.array_equal(picks, np.arange(len(picks))):
        log.debug("Reordering sensor data to match adjacency channel order.")
    X = X[:, :, picks]

    # 2b. Optional ROI restriction
    roi_cfg = (config.get('stats') or {}).get('roi') or {}
    roi_channels = set()
    if isinstance(roi_cfg, dict) and roi_cfg:
        # Use the new centralized ROI resolver
        fallback_roi_channels = config.get('roi_channels', {})
        roi_channels = _resolve_sensor_roi_channels(roi_cfg, fallback_roi_channels)

    if roi_channels:
        keep_mask = [ch in roi_channels for ch in ch_names]
        if not any(keep_mask):
            raise ValueError(
                "ROI restriction requested in sensor config but none of the specified channels "
                f"are present. Requested: {sorted(roi_channels)} | Available: {ch_names}"
            )
        keep_idx = np.flatnonzero(keep_mask)
        keep_ch_names = [ch_names[i] for i in keep_idx]
        log.info(
            f"ROI restriction active for sensor stats: keeping {len(keep_ch_names)} of "
            f"{len(ch_names)} channels ({', '.join(keep_ch_names)})."
        )
        adjacency = csr_matrix(adjacency).tocsr()[keep_idx][:, keep_idx]
        X = X[:, :, keep_idx]
        ch_names = keep_ch_names
        try:
            config['stats']['_roi_channels'] = keep_ch_names
        except Exception:
            pass

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
            subjects_dir = _resolve_fs_subjects_dir(project_root)
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
    """Run a 1D time-cluster permutation test on label time courses per label."""
    import mne

    log.info("Preparing label-averaged time courses for clustering...")

    stats_cfg = config.get('stats') or {}
    lt_cfg = stats_cfg.get('label_timeseries') or {}
    if lt_cfg.get('enabled', True) is False:
        raise ValueError("Label time-series analysis was disabled in the config.")

    parc = lt_cfg.get('parc', 'aparc')
    labels_wanted = lt_cfg.get('labels') or stats_cfg.get('roi', {}).get('labels', [])
    if not labels_wanted:
        raise ValueError("No labels provided for label time-series test (stats.label_timeseries.labels).")
    labels_wanted = [s.lower() for s in labels_wanted]

    project_root = Path(__file__).resolve().parents[2]
    subjects_dir = _resolve_fs_subjects_dir(project_root)
    all_labels = mne.read_labels_from_annot('fsaverage', parc=parc, subjects_dir=subjects_dir, verbose=False)

    def _base_label(name: str) -> str:
        name = name.lower()
        return name[:-3] if name.endswith('-lh') or name.endswith('-rh') else name

    selected_labels = [lab for lab in all_labels if _base_label(lab.name) in labels_wanted]
    if not selected_labels:
        raise ValueError("Selected labels not found on fsaverage. Check parc and label names.")

    label_names = [lab.name for lab in selected_labels]
    label_timecourses = []
    for stc in stcs:
        ltc = mne.extract_label_time_course(
            stc,
            selected_labels,
            src=fsaverage_src,
            mode='mean_flip',
            allow_empty=True,
            verbose=False,
        )
        if ltc.ndim != 2:
            raise ValueError("Label time-course extraction must return a 2D array (n_labels, n_times).")
        label_timecourses.append(ltc)
    X = np.stack(label_timecourses, axis=0)  # (n_subjects, n_labels, n_times)

    times = stcs[0].times
    aw = stats_cfg.get('analysis_window')
    if aw and len(aw) == 2:
        tmin, tmax = float(aw[0]), float(aw[1])
        log.info(f"Label time-series: analysis_window {tmin:.3f}-{tmax:.3f}s")
    else:
        tmin, tmax = float(times[0]), float(times[-1])
    time_mask = (times >= tmin) & (times <= tmax)
    if time_mask.sum() == 0:
        log.warning("Time mask is empty for label TS; using full window.")
        times_c = times
        X_c = X
    else:
        times_c = times[time_mask]
        X_c = X[:, :, time_mask]

    lt_p_threshold = float(lt_cfg.get('p_threshold', stats_cfg.get('p_threshold')))
    lt_tail = int(lt_cfg.get('tail', stats_cfg.get('tail', 0)))
    lt_n_permutations = int(lt_cfg.get('n_permutations', stats_cfg.get('n_permutations')))
    lt_cluster_alpha = float(lt_cfg.get('cluster_alpha', stats_cfg.get('cluster_alpha')))

    try:
        lt_cfg['_resolved_params'] = {
            'p_threshold': lt_p_threshold,
            'tail': lt_tail,
            'n_permutations': lt_n_permutations,
            'cluster_alpha': lt_cluster_alpha,
            'labels': labels_wanted,
            'resolved_label_names': label_names,
        }
    except Exception:
        pass

    n_subjects = X_c.shape[0]
    dof = n_subjects - 1
    if lt_tail == 0:
        thr = t_dist.ppf(1.0 - lt_p_threshold / 2.0, df=dof)
    else:
        thr = t_dist.ppf(1.0 - lt_p_threshold, df=dof)
        thr = -abs(thr) if lt_tail < 0 else abs(thr)

    log.info(
        f"Running 1D time clustering over labels {labels_wanted} with threshold {thr:.3f}, "
        f"n_permutations={lt_n_permutations}, tail={lt_tail}"
    )

    per_label_results = []
    for label_idx, label_name in enumerate(label_names):
        X_label = X_c[:, label_idx, :]
        stats = mne.stats.permutation_cluster_1samp_test(
            X_label,
            threshold=thr,
            tail=lt_tail,
            n_permutations=lt_n_permutations,
            out_type='mask',
            verbose=True,
        )
        per_label_results.append(stats)

    label_mean_ts = X_c.mean(axis=0)

    return per_label_results, times_c, label_mean_ts, label_names

