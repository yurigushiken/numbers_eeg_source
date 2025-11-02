from __future__ import annotations

from copy import deepcopy
from typing import Sequence


def _normalize(text: str | None) -> str:
    return (text or "").strip().lower()


def _find_condition(config: dict, identifier: str) -> dict | None:
    candidates: list[dict] = []
    contrast_cfg = config.get("contrast") or {}
    for key in ("condition_A", "condition_B"):
        cond = contrast_cfg.get(key)
        if cond:
            candidates.append(cond)

    extra = config.get("conditions")
    if isinstance(extra, dict):
        candidates.extend(extra.values())
    elif isinstance(extra, Sequence):
        candidates.extend(item for item in extra if isinstance(item, dict))

    target = _normalize(identifier)
    for cond in candidates:
        if _normalize(cond.get("name")) == target:
            return cond
        if _normalize(cond.get("condition_set_name")) == target:
            return cond
    return None


def _get_condition(config: dict, override: str | None, default_key: str) -> dict:
    if override:
        condition = _find_condition(config, override)
        if condition is None:
            raise ValueError(f"Could not locate condition matching '{override}' in config")
        return deepcopy(condition)

    condition = (config.get("contrast") or {}).get(default_key)
    if not condition:
        raise ValueError(f"Config missing contrast.{default_key}")
    return deepcopy(condition)


_CONDITION_FIELDS = {
    "name",
    "condition_set_name",
    "conditions",
    "accuracy",
    "baseline",
    "data_subset",
    "metadata",
    "metadata_filters",
}


def _extract_condition_dict(source: dict, *, fallback_name: str | None = None) -> dict:
    if not isinstance(source, dict):
        raise TypeError("Condition specification must be a dictionary")

    condition = {}
    for field in _CONDITION_FIELDS:
        if field in source:
            condition[field] = deepcopy(source[field])

    if fallback_name and "name" not in condition:
        condition["name"] = fallback_name

    if "conditions" not in condition and "condition_set_name" not in condition:
        raise ValueError("Condition entry must specify 'conditions' or 'condition_set_name'")

    return condition


def _format_accuracy_display(values: Sequence[str]) -> str | None:
    formatted = [str(val).upper() for val in values if val]
    if not formatted:
        return None
    if all(token == formatted[0] for token in formatted):
        return formatted[0]
    return "/".join(formatted)


def _build_panel_spec(config: dict, accuracy_override: str | None) -> dict | None:
    panel_cfg = config.get("panel") or {}
    tracks_cfg = panel_cfg.get("tracks")
    if not tracks_cfg:
        return None

    if not isinstance(tracks_cfg, Sequence):
        raise ValueError("panel.tracks must be a sequence of track definitions")

    layout_cfg = panel_cfg.get("layout") or []
    if isinstance(layout_cfg, Sequence) and len(layout_cfg) == 2:
        rows = max(int(layout_cfg[0]), 1)
        cols = max(int(layout_cfg[1]), 1)
    else:
        rows = len(tracks_cfg)
        cols = 1

    tracks: list[dict] = []
    for idx, entry in enumerate(tracks_cfg):
        if not isinstance(entry, dict):
            raise ValueError("Each panel track must be a dictionary")

        label = entry.get("label") or entry.get("name") or f"Track {idx + 1}"

        components: list[dict] = []
        components_cfg = entry.get("components")
        if isinstance(components_cfg, Sequence) and components_cfg:
            for comp_index, comp in enumerate(components_cfg):
                if not isinstance(comp, dict):
                    raise ValueError("panel.components entries must be dictionaries")
                weight = float(comp.get("weight", 1.0))
                cond_source = comp.get("condition") or comp
                condition_cfg = _extract_condition_dict(cond_source, fallback_name=f"{label} component {comp_index + 1}")
                accuracy_value = comp.get("accuracy") or condition_cfg.get("accuracy") or entry.get("accuracy") or accuracy_override or "all"
                accuracy_value = str(accuracy_value).lower()
                condition_cfg["accuracy"] = accuracy_value
                components.append({
                    "condition": condition_cfg,
                    "weight": weight,
                    "accuracy": accuracy_value,
                })
        elif "weights" in entry and isinstance(entry.get("conditions"), Sequence):
            cond_sets = entry["conditions"]
            weights = entry.get("weights")
            if not isinstance(weights, Sequence) or len(cond_sets) != len(weights):
                raise ValueError("panel track 'weights' must match length of 'conditions'")
            for cond_set, weight in zip(cond_sets, weights, strict=False):
                if isinstance(cond_set, (str, bytes)):
                    cond_values = [cond_set]
                else:
                    cond_values = list(cond_set)
                base = dict(entry)
                base.pop("weights", None)
                base["conditions"] = cond_values
                condition_cfg = _extract_condition_dict(base, fallback_name=label)
                accuracy_value = condition_cfg.get("accuracy") or accuracy_override or "all"
                accuracy_value = str(accuracy_value).lower()
                condition_cfg["accuracy"] = accuracy_value
                components.append({
                    "condition": condition_cfg,
                    "weight": float(weight),
                    "accuracy": accuracy_value,
                })
        else:
            cond_source = entry.get("condition") or entry
            condition_cfg = _extract_condition_dict(cond_source, fallback_name=label)
            accuracy_value = cond_source.get("accuracy") or condition_cfg.get("accuracy") or entry.get("accuracy") or accuracy_override or "all"
            accuracy_value = str(accuracy_value).lower()
            condition_cfg["accuracy"] = accuracy_value
            weight = float(entry.get("weight", 1.0))
            components.append({
                "condition": condition_cfg,
                "weight": weight,
                "accuracy": accuracy_value,
            })

        accuracy_display = _format_accuracy_display([comp["accuracy"] for comp in components])
        tracks.append({
            "label": label,
            "components": components,
            "accuracy_display": accuracy_display,
        })

    if rows * cols < len(tracks):
        raise ValueError("panel.layout is too small for the number of tracks")

    title = panel_cfg.get("title") or config.get("analysis_name")

    return {
        "layout": (rows, cols),
        "tracks": tracks,
        "title": title,
    }


def _clean_label(text: str) -> str:
    if "(" in text and ")" in text:
        return text.split("(")[0].strip()
    return text.strip()


def _build_contrast_spec(config: dict, args) -> dict:
    condition_a_cfg = _get_condition(config, getattr(args, "condition_a", None), "condition_A")
    condition_b_cfg = _get_condition(config, getattr(args, "condition_b", None), "condition_B")

    label_a = _clean_label(condition_a_cfg.get("name") or condition_a_cfg.get("condition_set_name") or "Condition A")
    label_b = _clean_label(condition_b_cfg.get("name") or condition_b_cfg.get("condition_set_name") or "Condition B")

    acc_a = str(condition_a_cfg.get("accuracy") or getattr(args, "accuracy", None) or "all").lower()
    acc_b = str(condition_b_cfg.get("accuracy") or getattr(args, "accuracy", None) or "all").lower()
    condition_a_cfg["accuracy"] = acc_a
    condition_b_cfg["accuracy"] = acc_b

    comp_a = {
        "condition": condition_a_cfg,
        "weight": 1.0,
        "accuracy": acc_a,
    }
    comp_b = {
        "condition": condition_b_cfg,
        "weight": 1.0,
        "accuracy": acc_b,
    }

    tracks = [
        {
            "label": label_a,
            "components": [comp_a],
            "accuracy_display": _format_accuracy_display([acc_a]),
        },
        {
            "label": label_b,
            "components": [comp_b],
            "accuracy_display": _format_accuracy_display([acc_b]),
        },
        {
            "label": f"{label_a} - {label_b}",
            "components": [
                {"condition": deepcopy(condition_a_cfg), "weight": 1.0, "accuracy": acc_a},
                {"condition": deepcopy(condition_b_cfg), "weight": -1.0, "accuracy": acc_b},
            ],
            "accuracy_display": _format_accuracy_display([acc_a, acc_b]),
        },
    ]

    title = f"{label_a} ({str(acc_a).upper()}) vs {label_b} ({str(acc_b).upper()})"

    return {
        "layout": (2, 2),
        "tracks": tracks,
        "title": title,
    }


def _build_movie_spec(config: dict, args) -> dict:
    panel_spec = _build_panel_spec(config, getattr(args, "accuracy", None))
    if panel_spec is not None:
        return panel_spec
    return _build_contrast_spec(config, args)


def build_movie_spec(config: dict, args) -> dict:
    """Public entry-point used by the CLI to build a movie spec from config."""
    return _build_movie_spec(config, args)


