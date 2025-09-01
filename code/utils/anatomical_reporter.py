import mne
import numpy as np
import pandas as pd
from collections import Counter
import logging
import re
from nilearn import datasets
import nibabel as nib

log = logging.getLogger()

def get_anatomical_labels_for_cluster(cluster_vertices, src, subject, subjects_dir, parc='aparc'):
    """
    Identifies the anatomical labels for a set of source space vertices.
    """
    try:
        labels = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=subjects_dir, verbose=False)
    except FileNotFoundError:
        log.error(f"Annotation file for parcellation '{parc}' not found for subject '{subject}'.")
        return {} # Return empty map if labels can't be loaded

    lh_labels = [label for label in labels if label.hemi == 'lh']
    rh_labels = [label for label in labels if label.hemi == 'rh']

    n_lh_vertices = len(src[0]['vertno'])
    vertex_to_label_map = {}

    for vertex in cluster_vertices:
        if vertex < n_lh_vertices:
            hemi_labels = lh_labels
            hemi_idx = 0
            hemi_vertex_index = vertex
        else:
            hemi_labels = rh_labels
            hemi_idx = 1
            hemi_vertex_index = vertex - n_lh_vertices

        surface_vertex = src[hemi_idx]['vertno'][hemi_vertex_index]

        found = False
        for label in hemi_labels:
            if surface_vertex in label.vertices:
                vertex_to_label_map[vertex] = label.name
                found = True
                break
        if not found:
            vertex_to_label_map[vertex] = "Unknown"
    
    return vertex_to_label_map

def get_mni_coordinates_for_vertex(vertex_index, src, subject, subjects_dir):
    """
    Converts a single source space vertex index to MNI coordinates.
    """
    n_lh_vertices = len(src[0]['vertno'])

    if vertex_index < n_lh_vertices:
        hemi_code = 0
        hemi_vertex_index = vertex_index
    else:
        hemi_code = 1
        hemi_vertex_index = vertex_index - n_lh_vertices
        
    surface_vertex = src[hemi_code]['vertno'][hemi_vertex_index]

    # Convert this surface vertex to MNI coordinates
    mni_coords = mne.vertex_to_mni(
        vertices=[surface_vertex], # Pass as a list to avoid scalar indexing error
        hemis=hemi_code,
        subject=subject,
        subjects_dir=subjects_dir
    )
    return mni_coords[0] # Return the 1D array of coords

def generate_anatomical_report(stats_results, src, subject, subjects_dir, config):
    """
    Generates a detailed anatomical report for significant clusters.
    """
    t_obs, clusters, cluster_p_values, _ = stats_results
    alpha = config['stats']['cluster_alpha']
    parc = config.get('reporting', {}).get('anatomical_atlas', 'aparc.a2009s')

    sig_cluster_indices = np.where(cluster_p_values < alpha)[0]

    if not sig_cluster_indices.size:
        log.info("No significant clusters found to generate anatomical report.")
        return pd.DataFrame()

    report_data = []

    # Sort by p-value to match textual report ordering
    sorted_indices = sig_cluster_indices[np.argsort(cluster_p_values[sig_cluster_indices])]

    # Optional local->global vertex index mapping if ROI was applied upstream
    keep_idx = None
    try:
        ki = config.get('stats', {}).get('_keep_idx', None)
        if ki is not None:
            keep_idx = np.asarray(ki, dtype=int)
    except Exception:
        keep_idx = None

    for i, cluster_idx in enumerate(sorted_indices):
        p_value = cluster_p_values[cluster_idx]
        t_inds, v_inds = clusters[cluster_idx]
        
        # --- Find Peak Activation ---
        cluster_t_vals = t_obs[(t_inds, v_inds)]
        peak_i = int(np.abs(cluster_t_vals).argmax())

        # Remap local ROI indices to global src vertex indices if needed
        v_inds_local = np.asarray(v_inds, dtype=int)
        if keep_idx is not None:
            v_inds_global = keep_idx[v_inds_local]
        else:
            v_inds_global = v_inds_local

        peak_vertex = int(v_inds_global[peak_i])
        
        # --- Convert Peak to MNI ---
        peak_mni_coords = get_mni_coordinates_for_vertex(peak_vertex, src, subject, subjects_dir)
        mni_str = f"({peak_mni_coords[0]:.1f}, {peak_mni_coords[1]:.1f}, {peak_mni_coords[2]:.1f})"

        # --- Get Anatomical Labels ---
        unique_vertices = np.unique(v_inds_global)
        total_vertices_in_cluster = len(unique_vertices)
        v_to_l_map = get_anatomical_labels_for_cluster(unique_vertices, src, subject, subjects_dir, parc)
        
        label_counts = Counter(v_to_l_map.values())
        sorted_labels = label_counts.most_common()

        # Add a row for each anatomical region in the cluster
        for label_name, count in sorted_labels:
            contribution_percent = (count / total_vertices_in_cluster) * 100
            report_data.append({
                'Cluster ID': i + 1,
                'p-value': f"{p_value:.4f}",
                'Anatomical Region': label_name,
                'Vertices in Region': count,
                'Region Contribution (%)': f"{contribution_percent:.1f}%",
                'Peak Activation MNI (mm)': mni_str
            })

    return pd.DataFrame(report_data)


# --- Cortical Cluster Localization summary helpers ---
_BA_CACHE = {
    'atlas': None,
    'img': None,
}


def _ensure_brodmann_loaded():
    if _BA_CACHE['img'] is not None:
        return
    atlas = datasets.fetch_atlas_brodmann_2012()
    _BA_CACHE['atlas'] = atlas
    _BA_CACHE['img'] = nib.load(atlas.maps) if isinstance(atlas.maps, str) else atlas.maps


def _safe_int(v):
    try:
        return int(v)
    except Exception:
        return None


def brodmann_at_mni(mni_xyz, neighborhood: bool = True, radius_vox: int = 1):
    try:
        _ensure_brodmann_loaded()
        img = _BA_CACHE['img']
        atlas = _BA_CACHE['atlas']
        ijk = np.round(nib.affines.apply_affine(np.linalg.inv(img.affine), np.asarray(mni_xyz))).astype(int)
        data = img.get_fdata()
        # Brodmann 2012 atlas is typically 4D (one map per area). Handle both 3D and 4D.
        labels = []
        def label_at(i, j, k):
            if data.ndim == 4:
                if (i < 0 or j < 0 or k < 0 or i >= data.shape[0] or j >= data.shape[1] or k >= data.shape[2]):
                    return None
                vec = data[i, j, k, :]
                if np.allclose(vec, 0):
                    return None
                idx = _safe_int(np.argmax(vec))
                if idx is None or idx >= len(atlas.labels):
                    return None
                return atlas.labels[idx]
            else:
                try:
                    val = int(data[i, j, k])
                except Exception:
                    return None
                if val <= 0 or val >= len(atlas.labels):
                    return None
                return atlas.labels[val]

        i0, j0, k0 = ijk.tolist()
        if neighborhood:
            for di in range(-radius_vox, radius_vox + 1):
                for dj in range(-radius_vox, radius_vox + 1):
                    for dk in range(-radius_vox, radius_vox + 1):
                        lab = label_at(i0 + di, j0 + dj, k0 + dk)
                        if lab is not None:
                            labels.append(lab)
            if labels:
                # modal label
                label_str = Counter(labels).most_common(1)[0][0]
            else:
                label_str = None
        else:
            label_str = label_at(i0, j0, k0)

        if not label_str:
            return "BA N/A", "Unknown"
        nums = re.findall(r"\d+", label_str)
        ba_short = f"BA {'/'.join(nums)}" if nums else "BA N/A"
        return ba_short, label_str
    except Exception:
        return "BA N/A", "Unknown"


def map_label_to_lobe(label_name: str) -> str:
    s = (label_name or "").lower()
    if 'occip' in s or 'calcarine' in s or 'lingual' in s or 'cuneus' in s:
        return 'Occipital'
    if 'pariet' in s or 'postcentral' in s or 'precuneus' in s or 'supramarginal' in s:
        return 'Parietal'
    if 'front' in s or 'precentral' in s or 'subcentral' in s or 'central' in s:
        return 'Frontal'
    if 'temporal' in s or 'fusiform' in s:
        return 'Temporal'
    if 'cingul' in s:
        return 'Cingulate'
    if 'insula' in s:
        return 'Insula'
    return 'Unknown'


def _lobe_from_ba_string(ba_short: str) -> str:
    # crude mapping: group BA into lobes for tie-breaking
    nums = re.findall(r"\d+", ba_short)
    if not nums:
        return 'Unknown'
    try:
        ba = int(nums[0])
    except Exception:
        return 'Unknown'
    if ba in (17, 18, 19):
        return 'Occipital'
    if ba in (5, 7, 39, 40):
        return 'Parietal'
    if ba in (20, 21, 22, 37, 38, 41, 42):
        return 'Temporal'
    if ba in (4, 6, 8, 9, 10, 11, 44, 45, 46, 47):
        return 'Frontal'
    if ba in (23, 24, 31, 32):
        return 'Cingulate'
    return 'Unknown'


def generate_hs_summary_table(stats_results, src, subject, subjects_dir, config, top_k_regions: int = 3) -> pd.DataFrame:
    """
    Builds a Hyde & Spelke-style per-cluster summary.
    Columns: Cluster ID, Primary Lobe, Regions (top), Cluster Size (vertices),
             Brodmann (peak), Peak t, Cluster p (FWER), Peak MNI (mm).
    """
    t_obs, clusters, cluster_p_values, _ = stats_results
    alpha = config['stats']['cluster_alpha']
    parc = config.get('reporting', {}).get('anatomical_atlas', 'aparc')

    sig_cluster_indices = np.where(cluster_p_values < alpha)[0]
    if not sig_cluster_indices.size:
        return pd.DataFrame()

    sorted_indices = sig_cluster_indices[np.argsort(cluster_p_values[sig_cluster_indices])]
    rows = []

    # Optional local->global vertex index mapping if ROI was applied upstream
    keep_idx = None
    try:
        ki = config.get('stats', {}).get('_keep_idx', None)
        if ki is not None:
            keep_idx = np.asarray(ki, dtype=int)
    except Exception:
        keep_idx = None

    for rank, cluster_idx in enumerate(sorted_indices, start=1):
        p_value = float(cluster_p_values[cluster_idx])
        t_inds, v_inds = clusters[cluster_idx]

        cluster_t_vals = t_obs[(t_inds, v_inds)]
        peak_i = int(np.abs(cluster_t_vals).argmax())
        peak_t = float(cluster_t_vals[peak_i])

        # Remap local ROI indices to global src vertex indices if needed
        v_inds_local = np.asarray(v_inds, dtype=int)
        if keep_idx is not None:
            v_inds_global = keep_idx[v_inds_local]
        else:
            v_inds_global = v_inds_local

        peak_vertex = int(v_inds_global[peak_i])

        peak_mni_coords = get_mni_coordinates_for_vertex(peak_vertex, src, subject, subjects_dir)
        mni_str = f"({peak_mni_coords[0]:.1f}, {peak_mni_coords[1]:.1f}, {peak_mni_coords[2]:.1f})"
        hemi = 'LH' if peak_mni_coords[0] < 0 else ('RH' if peak_mni_coords[0] > 0 else 'Unknown')

        unique_vertices = np.unique(v_inds_global)
        cluster_size = int(unique_vertices.size)

        v_to_l_map = get_anatomical_labels_for_cluster(unique_vertices, src, subject, subjects_dir, parc)
        # Majority lobe vote across all vertices
        lobe_counts = Counter()
        for lbl in v_to_l_map.values():
            lobe_counts[map_label_to_lobe(lbl)] += 1
        # Remove unknown from vote
        if 'Unknown' in lobe_counts:
            del lobe_counts['Unknown']
        top_labels = [name for name, _ in Counter(v_to_l_map.values()).most_common(top_k_regions) if name and name != 'Unknown']
        regions_str = ", ".join(top_labels)
        if lobe_counts:
            primary_lobe = lobe_counts.most_common(1)[0][0]
        else:
            primary_lobe = 'Unknown'

        ba_short, _ = brodmann_at_mni(peak_mni_coords, neighborhood=True, radius_vox=1)
        if primary_lobe == 'Unknown':
            lob_from_ba = _lobe_from_ba_string(ba_short)
            if lob_from_ba != 'Unknown':
                primary_lobe = lob_from_ba
            else:
                # final tie-breaker using MNI y (posterior vs anterior)
                primary_lobe = 'Parietal' if peak_mni_coords[1] < 0 else 'Frontal'

        rows.append({
            'Cluster ID': rank,
            'Primary Lobe': primary_lobe,
            'Hemisphere (peak)': hemi,
            'Top regions': regions_str,
            'Cluster Size (vertices)': cluster_size,
            'Brodmann (peak)': ba_short,
            'Peak t': round(peak_t, 3),
            'Cluster p (FWER)': f"{p_value:.4f}",
            'Peak MNI (mm)': mni_str,
        })

    return pd.DataFrame(rows, columns=[
        'Cluster ID', 'Primary Lobe', 'Top regions', 'Cluster Size (vertices)',
        'Brodmann (peak)', 'Peak t', 'Cluster p (FWER)', 'Peak MNI (mm)'
    ])
