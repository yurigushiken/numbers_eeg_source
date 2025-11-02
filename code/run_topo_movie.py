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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import mne
import numpy as np
import imageio.v2 as imageio

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from code.utils import data_loader  # noqa: E402
from code.topo_movie.specs import build_movie_spec  # noqa: E402
from code.topo_movie.data import _condition_cache_key, _compute_per_subject_evoked  # noqa: E402
from code.topo_movie.plot import _load_thumbnail_image, _clamp, _render_frame, build_track_thumbnails  # noqa: E402


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


 


 


def _resolve_centerpiece_image(
    config: dict,
    *,
    thumbnail_root_path: Path | None,
    thumbnail_cache: dict[Path, np.ndarray | None],
    project_root: Path,
) -> tuple[np.ndarray | None, dict[str, object]]:
    overlays_cfg = config.get("overlays") or {}
    options = overlays_cfg.get("static_center_logo") or {}
    if not options:
        return None, {}

    analysis_name = str(config.get("analysis_name") or "")
    analysis_suffix = ""
    if analysis_name:
        parts = analysis_name.split("_")
        analysis_suffix = parts[-1] if parts else analysis_name
    numerosity_first_digit = None
    for ch in analysis_suffix:
        if ch.isdigit():
            numerosity_first_digit = ch
            break

    context = {
        "analysis_name": analysis_name,
        "analysis_suffix": analysis_suffix,
        "numerosity_first_digit": numerosity_first_digit or "",
    }

    filename_value = options.get("filename")
    path_value = options.get("path")
    candidate_path: Path | None = None

    if filename_value:
        try:
            formatted = str(filename_value).format(**context)
        except KeyError:
            formatted = str(filename_value)
        if thumbnail_root_path is None:
            log.warning(
                "Centerpiece filename '%s' specified but thumbnail root unavailable; skipping.",
                filename_value,
            )
            return None, options
        candidate_path = thumbnail_root_path / formatted

    if candidate_path is None and path_value:
        try:
            formatted = str(path_value).format(**context)
        except KeyError:
            formatted = str(path_value)
        candidate_path = Path(formatted)

    if candidate_path is None:
        return None, options

    if not candidate_path.is_absolute():
        root_override = options.get("root")
        base_dir: Path | None = None
        if root_override:
            root_path = Path(str(root_override))
            if not root_path.is_absolute():
                base_dir = project_root / root_path
            else:
                base_dir = root_path
        elif thumbnail_root_path is not None and not path_value:
            base_dir = thumbnail_root_path
        else:
            base_dir = project_root
        candidate_path = base_dir / candidate_path

    image = _load_thumbnail_image(candidate_path, cache=thumbnail_cache)
    if image is None:
        log.warning("Centerpiece image not found: %s", candidate_path)
    return image, options


 


def _identify_non_scalp_channels(info: mne.Info, candidates: Sequence[str]) -> list[str]:
    names = set(info.get("ch_names", []))
    return [ch for ch in candidates if ch in names]


 


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

    movie_spec = build_movie_spec(config, args)

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

    thumbnail_cfg = (movie_defaults.get("thumbnail") or {})
    thumbnail_root = thumbnail_cfg.get("root")
    if thumbnail_root:
        thumbnail_root_path = Path(thumbnail_root)
        if not thumbnail_root_path.is_absolute():
            thumbnail_root_path = PROJECT_ROOT / thumbnail_root_path
    else:
        thumbnail_root_path = None
    thumbnail_metadata_key = thumbnail_cfg.get("metadata_key")
    thumbnail_width = float(thumbnail_cfg.get("width", 0.32))
    thumbnail_height = float(thumbnail_cfg.get("height", thumbnail_width))
    thumbnail_width_clamped = max(min(thumbnail_width, 0.95), 0.05)
    thumbnail_height_clamped = max(min(thumbnail_height, 0.95), 0.05)
    thumbnail_loc = thumbnail_cfg.get("loc", "lower left")
    thumbnail_borderpad = float(thumbnail_cfg.get("borderpad", 0.6))
    thumbnail_position = thumbnail_cfg.get("position")
    thumbnail_position_units = str(thumbnail_cfg.get("position_units", "axes")).lower()
    mirror_padding = float(thumbnail_cfg.get("mirror_padding", 0.0))
    caption_enabled = bool(thumbnail_cfg.get("caption_enabled", False))
    caption_color = thumbnail_cfg.get("caption_color")
    try:
        caption_size = float(thumbnail_cfg.get("caption_size", 8.0))
    except Exception:
        caption_size = 8.0
    caption_italic = bool(thumbnail_cfg.get("caption_italic", False))
    caption_prefix_prime = thumbnail_cfg.get("caption_prefix_prime") or ""
    caption_prefix_oddball = thumbnail_cfg.get("caption_prefix_oddball") or ""
    caption_offset = thumbnail_cfg.get("caption_offset") or [0.0, -0.08]

    thumbnail_cache: dict[Path, np.ndarray | None] = {}
    centerpiece_image, centerpiece_cfg = _resolve_centerpiece_image(
        config,
        thumbnail_root_path=thumbnail_root_path,
        thumbnail_cache=thumbnail_cache,
        project_root=PROJECT_ROOT,
    )
    if centerpiece_cfg:
        centerpiece_options = {
            "width": float(centerpiece_cfg.get("width", 0.3)),
            "height": float(centerpiece_cfg.get("height", centerpiece_cfg.get("width", 0.3))),
            "position": centerpiece_cfg.get("position") or [0.35, 0.35],
            "position_units": str(centerpiece_cfg.get("position_units", "figure")).lower(),
            "border_color": centerpiece_cfg.get("border_color"),
            "border_width": float(centerpiece_cfg.get("border_width", 1.0)),
            "caption": centerpiece_cfg.get("caption"),
            "caption_offset": centerpiece_cfg.get("caption_offset") or [0.0, -0.05],
            "caption_size": float(centerpiece_cfg.get("caption_size", 8)),
            "caption_color": centerpiece_cfg.get("caption_color") or "black",
        }
    else:
        centerpiece_options = {}

    condition_cache: dict[tuple, tuple[mne.Evoked, int]] = {}
    track_evokeds: list[mne.Evoked] = []
    track_display_labels: list[str] = []
    track_thumbnails: list[dict[str, object]] = []
    missing_thumbnail_keys: set[str] = set()

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

        thumb_entries, missing = build_track_thumbnails(
            components,
            thumbnail_root_path=thumbnail_root_path,
            thumbnail_metadata_key=thumbnail_metadata_key,
            thumbnail_width=thumbnail_width,
            thumbnail_height=thumbnail_height,
            thumbnail_width_clamped=thumbnail_width_clamped,
            thumbnail_height_clamped=thumbnail_height_clamped,
            thumbnail_position=thumbnail_position,
            thumbnail_position_units=thumbnail_position_units,
            thumbnail_loc=thumbnail_loc,
            thumbnail_borderpad=thumbnail_borderpad,
            thumbnail_cache=thumbnail_cache,
            caption_enabled=caption_enabled,
            caption_color=caption_color,
            caption_size=caption_size,
            caption_italic=caption_italic,
            caption_prefix_prime=caption_prefix_prime,
            caption_prefix_oddball=caption_prefix_oddball,
            caption_offset=caption_offset,
            mirror_padding=mirror_padding,
        )
        track_thumbnails.append(thumb_entries)
        missing_thumbnail_keys.update(missing)

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

    if missing_thumbnail_keys:
        missing_list = sorted(missing_thumbnail_keys)
        preview = ", ".join(missing_list[:5])
        if len(missing_list) > 5:
            preview += f", +{len(missing_list) - 5} more"
        log.warning(
            "Unable to attach %d thumbnail(s); missing files: %s",
            len(missing_list),
            preview,
        )

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
            track_thumbnails,
            centerpiece_image,
            centerpiece_options,
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

