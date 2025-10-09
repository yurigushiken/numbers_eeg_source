"""
Enhanced Figure Caption Generator

This module generates scientifically informative figure captions that include:
1. Directional interpretation (what positive/negative t-values mean)
2. Statistical details

Author: Language and Cognitive Neuroscience Lab, Teachers College, Columbia University
"""

import ast
from typing import Dict, Optional, Any


def _clean_condition_label(label: str) -> str:
    """Lightly clean condition labels for readability."""
    if not isinstance(label, str):
        return "Unknown condition"

    cleaned = label.strip()
    replacements = {
        " Changes": "",
        " changes": "",
        " Transitions": "",
        " transitions": "",
        " (Full 1-6 Range)": "",
    }
    for src, tgt in replacements.items():
        cleaned = cleaned.replace(src, tgt)

    return " ".join(cleaned.split())


def _interpret_t_value_direction(
    peak_t: float,
    contrast: Dict[str, Any]
) -> tuple[str, str, str]:
    """
    Interpret the direction of a t-value based on the contrast definition.

    Args:
        peak_t: Peak t-value (positive or negative)
        contrast: Full contrast dictionary from the YAML config

    Returns:
        Tuple of (polarity_phrase, direction_phrase, color_hint)
    """
    contrast = contrast or {}
    contrast_name = contrast.get("name", "")

    cond_a_info = contrast.get("condition_A") or {}
    cond_b_info = contrast.get("condition_B") or {}
    cond_a = _clean_condition_label(cond_a_info.get("name") or cond_a_info.get("condition_set_name") or "Condition A")
    cond_b = _clean_condition_label(cond_b_info.get("name") or cond_b_info.get("condition_set_name") or "Condition B")

    weights = contrast.get("combination_weights") or []
    positive_favors = negative_favors = None

    if isinstance(weights, (list, tuple)) and len(weights) >= 2:
        w_a = weights[0]
        w_b = weights[1]
        if w_a > 0 and w_b < 0:
            positive_favors = cond_a
            negative_favors = cond_b
        elif w_a < 0 and w_b > 0:
            positive_favors = cond_b
            negative_favors = cond_a

    if positive_favors is None or negative_favors is None:
        # Fallback to parsing the contrast name if weights are ambiguous (e.g., >2 conditions)
        parts = contrast_name.split(" vs. ")
        if len(parts) == 2:
            positive_favors = _clean_condition_label(parts[0])
            negative_favors = _clean_condition_label(parts[1])
        else:
            positive_favors = "Condition A"
            negative_favors = "Condition B"

    if peak_t >= 0:
        polarity = "Positive t-values (red)"
        direction = f"stronger activation for {positive_favors} relative to {negative_favors}"
        color = "red"
    else:
        polarity = "Negative t-values (blue)"
        direction = f"stronger activation for {negative_favors} relative to {positive_favors}"
        color = "blue"

    return polarity, direction, color


def generate_sensor_caption(
    cluster_id: int,
    cluster_info: Dict[str, Any],
    contrast: Dict[str, Any],
    time_window_label: Optional[str] = None
) -> str:
    """
    Generate an enhanced caption for sensor-space cluster figures.

    Args:
        cluster_id: Cluster number (1, 2, 3, ...)
        cluster_info: Dictionary containing:
            - p_value: FWER-corrected p-value
            - peak_t: Peak t-value
            - n_channels: Number of channels
            - time_window: Time window string (e.g., "104.0 ms to 176.0 ms")
            - topography: Optional scalp region (e.g., "posterior", "frontal")
            - cluster_mass: Optional cluster mass
        contrast: Contrast dictionary
        time_window_label: Optional component label (e.g., "N1", "P3b")

    Returns:
        Formatted caption string with directional interpretation and statistical details
    """
    p_val = cluster_info.get('p_value', 0.05)
    peak_t = cluster_info.get('peak_t', 0.0)
    n_channels = cluster_info.get('n_channels', 0)
    time_window = cluster_info.get('time_window', '').replace('.0 ms', ' ms')
    topography = cluster_info.get('topography', '')

    # Get directional interpretation
    polarity, direction, color = _interpret_t_value_direction(peak_t, contrast)

    # Determine cluster description
    if cluster_id == 1:
        if topography:
            cluster_desc = f"The grand average difference wave (left) and topographical distribution of t-values (right) for the {topography} sensor cluster"
        else:
            cluster_desc = "The grand average difference wave (left) and topographical distribution of t-values (right) for the sensor cluster"
    else:
        if topography:
            cluster_desc = f"{topography.capitalize()} sensor cluster showing {'opposite' if cluster_id == 2 else 'additional'} directional effects"
        else:
            cluster_desc = f"Sensor cluster #{cluster_id}"

    # Add time window label if provided
    if time_window_label:
        cluster_desc += f" during the {time_window_label} time window"

    # Build the caption
    caption_parts = [
        f"{cluster_desc} (p={p_val:.4f}, peak t={peak_t:.2f}, {n_channels} channels, {time_window}).",
        f"**{polarity} indicate {direction}**."
    ]

    return " ".join(caption_parts)


def generate_source_caption(
    cluster_id: int,
    cluster_info: Dict[str, Any],
    contrast: Dict[str, Any],
    method: str = "dSPM",
    time_window_label: Optional[str] = None,
    analysis_window: Optional[str] = None
) -> str:
    """
    Generate an enhanced caption for source-space cluster figures.

    Args:
        cluster_id: Cluster number (1, 2, 3, ...)
        cluster_info: Dictionary containing:
            - p_value: FWER-corrected p-value
            - peak_t: Peak t-value
            - n_vertices: Number of vertices
            - peak_mni: MNI coordinates string (e.g., "(-22.6, -97.4, 10.8)")
            - primary_region: Primary anatomical region
            - top_regions: Optional list of contributing regions
        contrast: Contrast dictionary
        method: Source estimation method (e.g., "dSPM", "eLORETA")
        time_window_label: Optional component label (e.g., "N1", "P3b")
        analysis_window: Optional analysis time window (e.g., "80-200 ms")

    Returns:
        Formatted caption string with directional interpretation and statistical details
    """
    p_val = cluster_info.get('p_value', 0.05)
    peak_t = cluster_info.get('peak_t', 0.0)
    n_vertices = cluster_info.get('n_vertices', 0)
    peak_mni = cluster_info.get('peak_mni', '')
    primary_region = cluster_info.get('primary_region', '')

    # Clean up region names for readability
    if primary_region:
        region_display = primary_region.replace('lateraloccipital', 'lateral occipital cortex')
        region_display = region_display.replace('inferiorparietal', 'inferior parietal lobule')
        region_display = region_display.replace('superiorparietal', 'superior parietal cortex')
        region_display = region_display.replace('-lh', ' (left hemisphere)')
        region_display = region_display.replace('-rh', ' (right hemisphere)')
    else:
        region_display = "cortical source"

    # Get directional interpretation
    polarity, direction, color = _interpret_t_value_direction(peak_t, contrast)

    # Build opening phrase
    if cluster_id == 1:
        if time_window_label:
            opening = f"Source localization of the {time_window_label} effect to {region_display}"
        else:
            opening = f"Source localization to {region_display}"
    else:
        opening = f"Additional source cluster in {region_display}"

    # Add analysis window if provided
    if analysis_window:
        opening += f" ({analysis_window}"
    else:
        opening += " ("

    # Add statistics
    stats = f"p={p_val:.4f}, peak t={peak_t:.2f}"

    if peak_mni:
        stats += f" at MNI {peak_mni}"

    stats += f", {n_vertices} vertices)"

    # Build caption
    caption = f"{opening} {stats}. **{polarity} indicate {direction}**."

    return caption


def infer_time_window_label(time_window_str) -> Optional[str]:
    """
    Infer ERP component label from time window.

    Args:
        time_window_str: Time window string (e.g., "104.0 ms to 176.0 ms") or list of [start, end] in seconds

    Returns:
        Component label ("N1", "P3b", etc.) or None
    """
    if not time_window_str:
        return None

    def _label_from_ms_range(start_ms: float, end_ms: float) -> Optional[str]:
        mid_ms = (start_ms + end_ms) / 2

        if 80 <= mid_ms <= 200:
            return "N1"
        if 200 <= mid_ms <= 300:
            return "P2/N2"
        if 300 <= mid_ms <= 600:
            return "P3b"
        return None

    def _normalize_to_ms(start_val: Any, end_val: Any, unit: str = "auto") -> Optional[tuple[float, float]]:
        try:
            start = float(start_val)
            end = float(end_val)
        except (TypeError, ValueError):
            return None

        if unit == "seconds" or (unit == "auto" and abs(start) <= 10 and abs(end) <= 10):
            start *= 1000
            end *= 1000
        return start, end

    # Handle list input (e.g., [0.30, 0.496])
    if isinstance(time_window_str, (list, tuple)):
        if len(time_window_str) < 2:
            return None
        ms_range = _normalize_to_ms(time_window_str[0], time_window_str[1])
        if ms_range:
            return _label_from_ms_range(*ms_range)
        return None

    # Handle string input (e.g., "104.0 ms to 176.0 ms")
    import re
    if isinstance(time_window_str, str):
        stripped = time_window_str.strip()

        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                ms_range = _normalize_to_ms(parsed[0], parsed[1])
                if ms_range:
                    return _label_from_ms_range(*ms_range)

        numbers_with_unit = re.findall(r'(\d+\.?\d*)\s*ms', time_window_str)
        if len(numbers_with_unit) >= 2:
            ms_range = _normalize_to_ms(numbers_with_unit[0], numbers_with_unit[1], unit="ms")
            if ms_range:
                return _label_from_ms_range(*ms_range)

        numbers = re.findall(r'(\d+\.?\d*)', time_window_str)
        if len(numbers) >= 2:
            ms_range = _normalize_to_ms(numbers[0], numbers[1])
            if ms_range:
                return _label_from_ms_range(*ms_range)

    return None


def infer_topography_from_channels(channels: list) -> str:
    """
    Infer topographical region from channel list.

    Args:
        channels: List of channel names (e.g., ['E65', 'E70', 'E75', ...])

    Returns:
        Topography label ('posterior', 'frontal', 'central', etc.)
    """
    if not channels:
        return ""

    # GSN HydroCel 128 electrode positions (simplified mapping)
    # Posterior/occipital: E50-E100 (approximately)
    # Frontal: E1-E35 (approximately)
    # Central: E36-E49, E101-E128 (approximately)

    posterior_count = sum(1 for ch in channels if 'E' in ch and 50 <= int(ch[1:]) <= 100)
    frontal_count = sum(1 for ch in channels if 'E' in ch and 1 <= int(ch[1:]) <= 35)

    if posterior_count > frontal_count and posterior_count > len(channels) * 0.4:
        return "posterior"
    elif frontal_count > posterior_count and frontal_count > len(channels) * 0.4:
        return "frontal"
    else:
        return "central"
