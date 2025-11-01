"""
Lightweight configuration helpers that avoid heavy dependencies.

These functions are imported both by pipeline code and unit tests.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import yaml


def load_common_defaults(project_root: str | Path = ".") -> Dict[str, Any]:
    """Load configs/common.yaml if it exists, else return {}.

    This helper is intentionally independent of MNE to keep tests fast/light.
    """
    project_root = Path(project_root)
    common_path = project_root / "configs" / "common.yaml"
    if not common_path.exists():
        return {}
    with open(common_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_common_into_config(
    common: Dict[str, Any],
    analysis_cfg: Dict[str, Any],
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Merge selected defaults from `common` into `analysis_cfg` (non-destructive).

    Rules (minimal option A):
    - If analysis_cfg.epoch_window.baseline is missing, use common.epoch_defaults.baseline
    - Do not overwrite analysis-specific settings if they exist
    - Attach resolved defaults under analysis_cfg['_resolved_defaults'] for traceability
    """
    merged = dict(analysis_cfg)  # shallow copy
    resolved_defaults: Dict[str, Any] = {}

    # Epoch window defaults (full dict plus baseline convenience)
    epoch_window_common = common.get("epoch_window") if common else None
    epoch_window = dict(merged.get("epoch_window") or {})
    if epoch_window_common:
        for key, value in epoch_window_common.items():
            if overwrite or key not in epoch_window:
                epoch_window[key] = value
        merged["epoch_window"] = epoch_window
        resolved_defaults["epoch_window"] = epoch_window_common

    baseline_common = (
        (common.get("epoch_defaults") or {}).get("baseline") if common else None
    )
    if (overwrite or "baseline" not in epoch_window) and baseline_common is not None:
        epoch_window["baseline"] = baseline_common
        merged["epoch_window"] = epoch_window
        resolved_defaults["baseline"] = baseline_common

    # Data defaults (for discovery helpers)
    data_common = common.get("data") or {}
    if data_common:
        merged.setdefault("data", {})
        for key, value in data_common.items():
            if overwrite or key not in merged["data"]:
                merged["data"][key] = value
        resolved_defaults["data_source"] = data_common.get("source")
        resolved_defaults["preprocessing"] = data_common.get("preprocessing")

    # Channel defaults (e.g., non-scalp electrodes)
    channels_common = common.get("channels") or {}
    if channels_common:
        merged.setdefault("channels", {})
        for key, value in channels_common.items():
            if overwrite or key not in merged["channels"]:
                merged["channels"][key] = deepcopy(value)
        resolved_defaults["channels"] = channels_common

    movie_common = common.get("movie") or {}
    if movie_common:
        merged.setdefault("movie", {})
        for key, value in movie_common.items():
            if overwrite or key not in merged["movie"]:
                merged["movie"][key] = deepcopy(value)
        resolved_defaults["movie"] = movie_common

    if resolved_defaults:
        merged.setdefault("_resolved_defaults", {}).update(resolved_defaults)

    return merged


def resolve_data_source_dir(
    project_root: str | Path,
    data_source: Optional[str],
    preprocessing: Optional[str],
) -> Tuple[Path, str]:
    """Return (data_dir, mode='combined') for locating subject files.

    - If data_source is a directory path, return it with mode 'combined'.
    - If data_source in {None, 'new'}, use combined preprocessed data under
      data/data_preprocessed/<preprocessing>.
    - If data_source == 'old', raise to indicate legacy path is unsupported.
    """
    project_root = Path(project_root)
    if data_source and data_source not in {"new", "old"}:
        # Treat as custom relative/absolute path
        custom = Path(data_source)
        if not custom.is_absolute():
            custom = project_root / custom
        return custom, "combined"

    if data_source in (None, "new"):
        pp = preprocessing or "hpf_1.0_lpf_35_baseline-on"
        return project_root / "data" / "data_preprocessed" / pp, "combined"

    # old
    raise ValueError("Legacy 'old' data pipeline is no longer supported.")

