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


def generate_hs_summary_table(stats_results, src, subject, subjects_dir, config, top_k_regions: int = 3) -> pd.DataFrame:
    """
    Builds a Hyde & Spelke-style per-cluster summary.
    Columns: Cluster ID, Primary Lobe, Regions (top), Cluster Size (vertices),
             Peak t, Cluster p (FWER), Peak MNI (mm).
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

        if primary_lobe == 'Unknown':
            # final tie-breaker using MNI y (posterior vs anterior)
            primary_lobe = 'Parietal' if peak_mni_coords[1] < 0 else 'Frontal'

        rows.append({
            'Cluster ID': rank,
            'Primary Lobe': primary_lobe,
            'Hemisphere (peak)': hemi,
            'Top regions': regions_str,
            'Cluster Size (vertices)': cluster_size,
            'Peak t': round(peak_t, 3),
            'Cluster p (FWER)': f"{p_value:.4f}",
            'Peak MNI (mm)': mni_str,
        })

    return pd.DataFrame(rows, columns=[
        'Cluster ID', 'Primary Lobe', 'Top regions', 'Cluster Size (vertices)',
        'Peak t', 'Cluster p (FWER)', 'Peak MNI (mm)'
    ])
