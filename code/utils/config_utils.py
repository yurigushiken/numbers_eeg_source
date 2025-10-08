"""
Lightweight configuration helpers that avoid heavy dependencies.

These functions are imported both by pipeline code and unit tests.
"""
from __future__ import annotations

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


def merge_common_into_config(common: Dict[str, Any], analysis_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge selected defaults from `common` into `analysis_cfg` (non-destructive).

    Rules (minimal option A):
    - If analysis_cfg.epoch_window.baseline is missing, use common.epoch_defaults.baseline
    - Do not overwrite analysis-specific settings if they exist
    - Attach resolved defaults under analysis_cfg['_resolved_defaults'] for traceability
    """
    merged = dict(analysis_cfg)  # shallow copy
    resolved_defaults: Dict[str, Any] = {}

    # Baseline default
    baseline_common = (
        (common.get("epoch_defaults") or {}).get("baseline") if common else None
    )
    epoch_window = dict(merged.get("epoch_window") or {})
    if "baseline" not in epoch_window and baseline_common is not None:
        epoch_window["baseline"] = baseline_common
        merged["epoch_window"] = epoch_window
        resolved_defaults["baseline"] = baseline_common

    # Data defaults (for discovery helpers)
    data_common = common.get("data") or {}
    if data_common:
        resolved_defaults["data_source"] = data_common.get("source")
        resolved_defaults["preprocessing"] = data_common.get("preprocessing")

    if resolved_defaults:
        merged.setdefault("_resolved_defaults", {}).update(resolved_defaults)

    return merged


def resolve_data_source_dir(
    project_root: str | Path,
    data_source: Optional[str],
    preprocessing: Optional[str],
) -> Tuple[Path, str]:
    """Return (data_dir, mode) for locating subject files based on inputs.

    - If data_source is a directory path, return it with mode 'combined'.
    - If data_source == 'new' or None, use combined preprocessed data under
      data/data_preprocessed/<preprocessing>.
    - If data_source == 'old', return legacy roots under data/<accuracy>/ (mode 'split').
    The caller still needs to append accuracy folder when mode == 'split'.
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
    return project_root / "data", "split"

