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
        default=10.0,
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


def _compute_per_subject_evoked(
    subject_dirs: Iterable[Path],
    condition_cfg: dict,
    baseline: Sequence[float] | None,
    accuracy_override: str | None,
) -> list[mne.Evoked]:
    per_subject: list[mne.Evoked] = []
    acc_flag = condition_cfg.get("accuracy") or accuracy_override or "all"

    for subject_dir in subject_dirs:
        evokeds, _ = data_loader.get_evoked_for_condition(
            subject_dir,
            condition_cfg,
            baseline=baseline,
            accuracy=acc_flag,
        )
        if not evokeds:
            continue
        if len(evokeds) > 1:
            per_subject.append(mne.grand_average(evokeds))
        else:
            per_subject.append(evokeds[0].copy())

    if not per_subject:
        raise RuntimeError(
            f"No evoked data found for condition set '{condition_cfg.get('condition_set_name')}'"
        )
    return per_subject


def _compute_scale(evoked: mne.Evoked, mask: np.ndarray) -> tuple[float, float]:
    data = evoked.get_data() * 1e6
    if mask.shape[0] != data.shape[1]:
        mask = np.ones(data.shape[1], dtype=bool)
    slice_data = data[:, mask]
    if slice_data.size == 0:
        slice_data = data
    vmax = np.nanpercentile(np.abs(slice_data), 99.0)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.nanmax(np.abs(slice_data))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return (-float(vmax), float(vmax))


BELT_CHANNELS = [
    "E1", "E8", "E14", "E17", "E21", "E25", "E32", "E38", "E43",
    "E44", "E48", "E49", "E56", "E63", "E68", "E73", "E81", "E88",
    "E94", "E99", "E107", "E113", "E114", "E119", "E120", "E121",
    "E125", "E126", "E127", "E128",
]


def _identify_belt_channels(info: mne.Info) -> list[str]:
    names = set(info.get("ch_names", []))
    return [ch for ch in BELT_CHANNELS if ch in names]


def _render_frame(
    evokeds: Sequence[mne.Evoked],
    time_index: int,
    scales: Sequence[tuple[float, float]],
    labels: Sequence[str],
    frame_path: Path,
) -> Path:
    n_rows = len(evokeds)
    width = 5.4 if n_rows > 1 else 4.0
    fig, axes = plt.subplots(n_rows, 1, figsize=(width, 3.4 * n_rows), dpi=180)
    if n_rows == 1:
        axes = [axes]
    time_ms = evokeds[0].times[time_index] * 1000.0

    for ax, evoked, scale, label in zip(axes, evokeds, scales, labels, strict=False):
        data = evoked.data[:, time_index] * 1e6
        im, _ = mne.viz.plot_topomap(
            data,
            evoked.info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            vlim=(scale[0], scale[1]),
        )
        ax.set_title(f"{label} @ {time_ms:.0f} ms", fontsize=11)
        frac = 0.06 if n_rows > 1 else 0.046
        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=frac, pad=0.12)
        cbar.set_label("Amplitude (µV)")

    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.32)
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
    analysis_name = config.get("analysis_name") or config_slug
    slug = config_slug

    condition_a_cfg = _get_condition(config, args.condition_a, "condition_A")
    condition_b_cfg = _get_condition(config, args.condition_b, "condition_B")

    label_a = condition_a_cfg.get("name") or condition_a_cfg.get("condition_set_name") or "Condition A"
    label_b = condition_b_cfg.get("name") or condition_b_cfg.get("condition_set_name") or "Condition B"
    diff_label = f"{label_a} − {label_b}"

    baseline = None
    epoch_cfg = config.get("epoch_window") or {}
    if epoch_cfg:
        baseline = epoch_cfg.get("baseline")

    subject_dirs = data_loader.get_subject_dirs("all", project_root=PROJECT_ROOT)
    if not subject_dirs:
        raise RuntimeError("Could not locate any subject directories; did you preprocess data?")

    per_subject_a = _compute_per_subject_evoked(subject_dirs, condition_a_cfg, baseline, args.accuracy)
    per_subject_b = _compute_per_subject_evoked(subject_dirs, condition_b_cfg, baseline, args.accuracy)

    grand_a = mne.grand_average(list(per_subject_a))
    grand_b = mne.grand_average(list(per_subject_b))

    dropped_channels: list[str] = []
    if not args.include_belt:
        dropped_channels = _identify_belt_channels(grand_a.info)
        if dropped_channels:
            log.info(
                "Dropping %d below-belt electrodes: %s",
                len(dropped_channels),
                ", ".join(dropped_channels),
            )
            grand_a = grand_a.copy().drop_channels(dropped_channels, on_missing="ignore")
            grand_b = grand_b.copy().drop_channels(dropped_channels, on_missing="ignore")

    grand_diff = mne.combine_evoked([grand_a, grand_b], weights=[1.0, -1.0])

    times = grand_a.times
    if not np.allclose(times, grand_b.times) or not np.allclose(times, grand_diff.times):
        raise RuntimeError("Time axes are not aligned across evokeds; cannot build synced movie.")

    tmin = epoch_cfg.get("tmin")
    tmax = epoch_cfg.get("tmax")
    if tmin is None or tmax is None:
        mask = np.ones_like(times, dtype=bool)
    else:
        mask = (times >= float(tmin)) & (times <= float(tmax))
    scale_a = _compute_scale(grand_a, mask)
    scale_b = _compute_scale(grand_b, mask)
    scale_diff = _compute_scale(grand_diff, mask)

    step = max(args.frame_step or 1, 1)
    frame_indices = list(range(0, len(times), step)) or [0]
    if frame_indices[-1] != len(times) - 1:
        frame_indices.append(len(times) - 1)
    output_root = Path(args.output_dir)
    frame_root = output_root / "frames" / slug
    frame_root.mkdir(parents=True, exist_ok=True)

    tracks: list[tuple] = []
    if not args.skip_panel:
        tracks.append(
            ([grand_a, grand_b, grand_diff], [scale_a, scale_b, scale_diff], [label_a, label_b, diff_label], output_root / f"{slug}_panel.mp4", f"{slug}_panel")
        )

    if tracks:
        assert len(tracks) == 1
        evs, scales, labels, video_path, prefix = tracks[0]
        frames: list[Path] = []
        for order, idx in enumerate(frame_indices):
            frame_path = frame_root / f"{prefix}_{order:04d}.png"
            _render_frame(evs, idx, scales, labels, frame_path)
            frames.append(frame_path)

        if frames:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            with imageio.get_writer(video_path, format="mp4", fps=args.fps, macro_block_size=1) as writer:
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

