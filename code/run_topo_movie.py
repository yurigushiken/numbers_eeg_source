"""Generate sensor-space topomap videos for two conditions and their difference."""

from __future__ import annotations

import argparse
import logging
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

# Ensure headless rendering regardless of environment
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import imageio.v2 as imageio

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from code.utils import data_loader  # noqa: E402


log = logging.getLogger("topo_movie")

COLOR_SCALE = (-4.0, 4.0)
DEFAULT_FPS = 10.0

DEFAULT_NON_SCALP_CHANNELS = [
    "E1", "E8", "E14", "E17", "E21", "E25", "E32", "E38", "E43",
    "E44", "E48", "E49", "E56", "E63", "E68", "E73", "E81", "E88",
    "E94", "E99", "E107", "E113", "E114", "E119", "E120", "E121",
    "E125", "E126", "E127", "E128",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export sensor-space topomap videos")
    parser.add_argument("--config", required=True, help="Path to the sensor YAML config")
    parser.add_argument(
        "--condition-a",
        dest="condition_a",
        help="Override for condition A (match 'name' or 'condition_set_name')",
    )
    parser.add_argument(
        "--condition-b",
        dest="condition_b",
        help="Override for condition B (match 'name' or 'condition_set_name')",
    )
    parser.add_argument(
        "--accuracy",
        choices=["all", "acc1", "acc0"],
        help="Optional accuracy filter override",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help=(
            "Number of samples to skip between frames. "
            "With 250 Hz data, 1 => 4 ms per frame (default)."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Frames per second for the exported videos",
    )
    parser.add_argument(
        "--output-dir",
        default="topographic_movies/topos",
        help="Base directory for video exports (created if missing)",
    )
    parser.add_argument(
        "--skip-panel",
        action="store_true",
        help="Skip rendering the 3-row combined panel video",
    )
    parser.add_argument(
        "--include-belt",
        action="store_true",
        help="Keep below-belt electrodes instead of removing them from topomaps",
    )
    return parser.parse_args()


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


def _condition_cache_key(condition_cfg: dict, accuracy: str) -> tuple:
    items: list[tuple[str, object]] = []
    for key, value in condition_cfg.items():
        if key == "name":
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items.append((key, tuple(value)))
        elif isinstance(value, dict):
            items.append((key, tuple(sorted(value.items()))))
        else:
            items.append((key, value))
    return (accuracy, tuple(sorted(items)))


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


def _build_contrast_spec(config: dict, args: argparse.Namespace) -> dict:
    condition_a_cfg = _get_condition(config, args.condition_a, "condition_A")
    condition_b_cfg = _get_condition(config, args.condition_b, "condition_B")

    label_a = _clean_label(condition_a_cfg.get("name") or condition_a_cfg.get("condition_set_name") or "Condition A")
    label_b = _clean_label(condition_b_cfg.get("name") or condition_b_cfg.get("condition_set_name") or "Condition B")

    acc_a = str(condition_a_cfg.get("accuracy") or args.accuracy or "all").lower()
    acc_b = str(condition_b_cfg.get("accuracy") or args.accuracy or "all").lower()
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
                {
                    "condition": deepcopy(condition_a_cfg),
                    "weight": 1.0,
                    "accuracy": acc_a,
                },
                {
                    "condition": deepcopy(condition_b_cfg),
                    "weight": -1.0,
                    "accuracy": acc_b,
                },
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


def _build_movie_spec(config: dict, args: argparse.Namespace) -> dict:
    panel_spec = _build_panel_spec(config, args.accuracy)
    if panel_spec is not None:
        return panel_spec
    return _build_contrast_spec(config, args)


def _parse_frame_window(config: dict) -> tuple[float | None, float | None]:
    movie_cfg = config.get("movie") or {}
    window = movie_cfg.get("frame_window")

    tmin: float | None = None
    tmax: float | None = None

    if isinstance(window, (list, tuple)) and len(window) == 2:
        try:
            tmin = float(window[0]) if window[0] is not None else None
        except Exception:
            tmin = None
        try:
            tmax = float(window[1]) if window[1] is not None else None
        except Exception:
            tmax = None

    if tmin is None or tmax is None:
        epoch_cfg = config.get("epoch_window") or {}
        if tmin is None:
            try:
                raw_tmin = epoch_cfg.get("tmin")
                if raw_tmin is not None:
                    tmin = float(raw_tmin)
            except Exception:
                tmin = None
        if tmax is None:
            try:
                raw_tmax = epoch_cfg.get("tmax")
                if raw_tmax is not None:
                    tmax = float(raw_tmax)
            except Exception:
                tmax = None

    if tmin is not None and tmax is not None and tmin > tmax:
        log.warning(
            "Frame window tmin (%.3f) exceeds tmax (%.3f); swapping values.",
            tmin,
            tmax,
        )
        tmin, tmax = tmax, tmin

    return tmin, tmax


def _compute_per_subject_evoked(
    subject_dirs: Iterable[Path],
    condition_cfg: dict,
    baseline: Sequence[float] | None,
    accuracy: str,
) -> tuple[list[mne.Evoked], int]:
    per_subject: list[mne.Evoked] = []
    total_epochs = 0

    for subject_dir in subject_dirs:
        evokeds, epochs_list = data_loader.get_evoked_for_condition(
            subject_dir,
            condition_cfg,
            baseline=baseline,
            accuracy=accuracy,
        )
        if not evokeds:
            continue
        if len(evokeds) > 1:
            subject_evoked = mne.grand_average(evokeds)
        else:
            subject_evoked = evokeds[0].copy()

        per_subject.append(subject_evoked)

        subject_epoch_count = 0
        for epochs in epochs_list:
            try:
                subject_epoch_count += len(epochs)
            except TypeError:
                continue
        if subject_epoch_count == 0:
            try:
                subject_epoch_count = int(subject_evoked.nave or 0)
            except Exception:
                subject_epoch_count = 0
        total_epochs += subject_epoch_count

    if not per_subject:
        identifier = condition_cfg.get("name") or \
            condition_cfg.get("condition_set_name") or \
            ",".join(condition_cfg.get("conditions", []))
        raise RuntimeError(f"No evoked data found for condition '{identifier}'")
    return per_subject, total_epochs


def _identify_non_scalp_channels(info: mne.Info, candidates: Sequence[str]) -> list[str]:
    names = set(info.get("ch_names", []))
    return [ch for ch in candidates if ch in names]


def _render_frame(
    evokeds: Sequence[mne.Evoked],
    time_index: int,
    scales: Sequence[tuple[float, float]],
    labels: Sequence[str],
    frame_path: Path,
    title: str | None,
    layout: tuple[int, int],
) -> Path:
    rows, cols = layout
    if rows <= 0 or cols <= 0:
        raise ValueError("Panel layout must have positive row and column counts")

    figsize = (4.0 * cols, 3.4 * rows)
    # Use GridSpec to allow the bottom row to span both columns in contrast mode
    if rows == 2 and cols == 2 and len(evokeds) == 3:
        fig = plt.figure(figsize=figsize, dpi=180)
        gs = fig.add_gridspec(2, 2)
        ax_top_left = fig.add_subplot(gs[0, 0])
        ax_top_right = fig.add_subplot(gs[0, 1])
        ax_bottom_center = fig.add_subplot(gs[1, :])  # span both columns
        flat_axes = [ax_top_left, ax_top_right, ax_bottom_center]
    else:
        fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=180, squeeze=False)
        flat_axes = list(axes.flat)
    time_ms = evokeds[0].times[time_index] * 1000.0

    for idx, ax in enumerate(flat_axes):
        if idx >= len(evokeds):
            ax.axis("off")
            continue

        evoked = evokeds[idx]
        scale = scales[idx]
        label = labels[idx]

        data = evoked.data[:, time_index] * 1e6
        im, _ = mne.viz.plot_topomap(
            data,
            evoked.info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            vlim=(scale[0], scale[1]),
        )
        title_lines = [label, f"@ {time_ms:.0f} ms"]
        ax.set_title("\n".join(title_lines), fontsize=11)

        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.18)
        cbar.set_ticks(np.arange(int(COLOR_SCALE[0]), int(COLOR_SCALE[1]) + 1, 1))
        cbar.set_label("Amplitude (µV)")

    if title:
        text = fig.suptitle(title, fontsize=13, y=0.99, ha="center")
        try:
            text.set_wrap(True)
        except AttributeError:
            pass

    fig.subplots_adjust(top=0.9, bottom=0.08, hspace=0.6, wspace=0.35)
    fig.savefig(frame_path, dpi=180)
    plt.close(fig)
    return frame_path


def main() -> None:
    args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = data_loader.load_config(config_path, project_root=PROJECT_ROOT)
    if config.get("domain") != "sensor":
        raise ValueError("This exporter currently supports sensor-space configs only.")

    config_slug = config_path.stem
    slug = config_slug

    movie_spec = _build_movie_spec(config, args)

    movie_defaults = (config.get("movie") or {})
    fps = args.fps
    if fps is None:
        try:
            fps = float(movie_defaults.get("fps", DEFAULT_FPS))
        except Exception:
            fps = DEFAULT_FPS

    if args.skip_panel:
        log.info("Skipping panel rendering (--skip-panel).")
        return

    baseline = None
    epoch_cfg = config.get("epoch_window") or {}
    if epoch_cfg:
        baseline = epoch_cfg.get("baseline")

    data_cfg = config.get("data") or {}
    data_source_override = data_cfg.get("source")
    preprocessing_override = data_cfg.get("preprocessing")
    custom_data_source = None
    if data_source_override:
        custom_data_source = str(data_source_override)
    elif preprocessing_override:
        custom_data_source = str(Path("data") / "data_preprocessed" / str(preprocessing_override))

    subject_dirs = data_loader.get_subject_dirs(
        "all",
        project_root=PROJECT_ROOT,
        data_source=custom_data_source,
    )
    if not subject_dirs:
        raise RuntimeError("Could not locate any subject directories; did you preprocess data?")

    condition_cache: dict[tuple, tuple[mne.Evoked, int]] = {}
    track_evokeds: list[mne.Evoked] = []
    track_display_labels: list[str] = []

    for track in movie_spec["tracks"]:
        components = track["components"]
        component_evokeds: list[mne.Evoked] = []
        component_counts: list[int] = []
        for component in components:
            condition_cfg = component["condition"]
            accuracy_flag = component["accuracy"]
            cache_key = _condition_cache_key(condition_cfg, accuracy_flag)
            if cache_key not in condition_cache:
                per_subject, epoch_count = _compute_per_subject_evoked(
                    subject_dirs,
                    condition_cfg,
                    baseline,
                    accuracy_flag,
                )
                aggregated = mne.grand_average(list(per_subject))
                condition_cache[cache_key] = (aggregated, epoch_count)
            cached_evoked, cached_epochs = condition_cache[cache_key]
            component_evokeds.append(cached_evoked.copy())
            component_counts.append(cached_epochs)

        if len(component_evokeds) == 1:
            track_evoked = component_evokeds[0]
        else:
            weights = [component["weight"] for component in components]
            track_evoked = mne.combine_evoked(component_evokeds, weights=weights)

        track_evokeds.append(track_evoked)

        accuracy_display = track.get("accuracy_display")
        if accuracy_display:
            accuracy_display = str(accuracy_display)
            if "/" in accuracy_display:
                accuracy_display_fmt = " vs ".join(accuracy_display.split("/"))
            else:
                accuracy_display_fmt = accuracy_display
        else:
            accuracy_display_fmt = None

        if len(component_counts) == 1:
            count_display = f"n={component_counts[0]}"
        else:
            count_display = "n=" + " vs ".join(str(c) for c in component_counts)

        if accuracy_display_fmt and count_display:
            display = f"{track['label']} ({accuracy_display_fmt}, {count_display})"
        elif accuracy_display_fmt:
            display = f"{track['label']} ({accuracy_display_fmt})"
        else:
            display = f"{track['label']} ({count_display})"

        track_display_labels.append(display)

    if not track_evokeds:
        raise RuntimeError("No tracks available to render; check configuration")

    channels_cfg = config.get("channels") or {}
    configured_non_scalp = channels_cfg.get("non_scalp")
    if isinstance(configured_non_scalp, (str, bytes)):
        configured_non_scalp = [configured_non_scalp]
    elif not isinstance(configured_non_scalp, Sequence):
        configured_non_scalp = None
    non_scalp_candidates = list(configured_non_scalp or DEFAULT_NON_SCALP_CHANNELS)

    dropped_channels: list[str] = []
    if not args.include_belt:
        dropped_channels = _identify_non_scalp_channels(track_evokeds[0].info, non_scalp_candidates)
        if dropped_channels:
            log.info(
                "Dropping %d non-scalp electrodes: %s",
                len(dropped_channels),
                ", ".join(dropped_channels),
            )
            for idx, evoked in enumerate(track_evokeds):
                track_evokeds[idx] = evoked.copy().drop_channels(dropped_channels, on_missing="ignore")

    times = track_evokeds[0].times
    for evoked in track_evokeds[1:]:
        if not np.allclose(times, evoked.times):
            raise RuntimeError("Time axes are not aligned across evokeds; cannot build synced movie.")

    scales = [COLOR_SCALE for _ in track_evokeds]

    step = max(args.frame_step or 1, 1)
    frame_tmin, frame_tmax = _parse_frame_window(config)

    start_idx = 0
    end_idx = len(times) - 1

    if frame_tmin is not None:
        idx = int(np.searchsorted(times, frame_tmin, side="left"))
        start_idx = min(max(idx, 0), len(times) - 1)

    if frame_tmax is not None:
        idx = int(np.searchsorted(times, frame_tmax, side="right") - 1)
        end_idx = min(max(idx, 0), len(times) - 1)

    if start_idx > end_idx:
        log.warning(
            "Frame window %.3f–%.3f outside data range %.3f–%.3f; using full window.",
            frame_tmin if frame_tmin is not None else float(times[0]),
            frame_tmax if frame_tmax is not None else float(times[-1]),
            float(times[0]),
            float(times[-1]),
        )
        start_idx = 0
        end_idx = len(times) - 1

    frame_indices = list(range(start_idx, end_idx + 1, step))
    if not frame_indices:
        frame_indices = [end_idx]
    elif frame_indices[-1] != end_idx:
        frame_indices.append(end_idx)

    output_root = Path(args.output_dir)
    frame_root = output_root / "frames" / slug
    frame_root.mkdir(parents=True, exist_ok=True)

    prefix = f"{slug}_panel"
    video_path = output_root / f"{prefix}.mp4"

    frames: list[Path] = []
    for order, idx in enumerate(frame_indices):
        frame_path = frame_root / f"{prefix}_{order:04d}.png"
        _render_frame(
            track_evokeds,
            idx,
            scales,
            track_display_labels,
            frame_path,
            movie_spec.get("title"),
            movie_spec.get("layout", (len(track_evokeds), 1)),
        )
        frames.append(frame_path)

    if frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(video_path, format="mp4", fps=fps, macro_block_size=1) as writer:
            for frame_path in frames:
                frame_img = imageio.imread(frame_path)
                if frame_img.shape[0] % 2:
                    frame_img = np.pad(frame_img, ((0, 1), (0, 0), (0, 0)), mode="edge")
                if frame_img.shape[1] % 2:
                    frame_img = np.pad(frame_img, ((0, 0), (0, 1), (0, 0)), mode="edge")
                writer.append_data(frame_img)
        log.info("Wrote %s (%d frames)", video_path, len(frames))


if __name__ == "__main__":
    main()

