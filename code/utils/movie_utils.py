"""Utilities for generating source and sensor movies from pipeline outputs."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import mne
import numpy as np

from .plotting import _check_fsaverage_path


LOG = logging.getLogger(__name__)


def generate_movies(
    *,
    config: dict,
    output_dir: Path | str,
    analysis_name: str,
    derivatives_root: Path | str | None = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    """Generate source and (optionally) sensor movies based on configuration.

    Parameters
    ----------
    config : dict
        Source analysis configuration dictionary.
    output_dir : Path | str
        Directory that already contains the grand-average STC output.
    analysis_name : str
        Name of the analysis (used to locate saved artifacts).
    derivatives_root : Path | str | None
        Root derivatives directory. If omitted, it is inferred from ``output_dir``.
    logger : logging.Logger | None
        Optional logger to use for status and warning messages.

    Returns
    -------
    dict
        Mapping of artifact labels (``source``, ``topo``, ``stitched``) to the
        generated file paths. The mapping is empty when movie generation is
        disabled or fails.
    """

    log = logger or LOG
    output_dir = Path(output_dir)

    movie_cfg = (config.get("visualization") or {}).get("movie") or {}
    if not movie_cfg or not bool(movie_cfg.get("enabled", False)):
        log.debug("Visualization movie generation disabled in config; skipping.")
        return {}

    fps = int(movie_cfg.get("fps", 30))
    codec = movie_cfg.get("codec", "libx264")
    stats_cfg = (config.get("stats") or {})
    movie_window = movie_cfg.get("time_window") or stats_cfg.get("analysis_window")
    n_frames = int(movie_cfg.get("n_frames", 30))
    include_topo = bool(movie_cfg.get("include_topo", False))
    stitch_movies = bool(movie_cfg.get("stitch", False))

    stc_path = output_dir / f"{analysis_name}_grand_average-stc.h5"
    try:
        stc = mne.read_source_estimate(str(stc_path))
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Unable to read grand-average STC from %s: %s", stc_path, exc)
        return {}

    stc_tmin = float(stc.times[0])
    stc_tmax = float(stc.times[-1])
    requested_window = _normalize_window(movie_window, (stc_tmin, stc_tmax))
    movie_tmin, movie_tmax = _clamp_window(
        requested_window,
        (stc_tmin, stc_tmax),
        float(stc.tstep),
    )

    if movie_tmin != requested_window[0] or movie_tmax != requested_window[1]:
        log.info(
            "Requested movie window %s clamped to available STC window [%.3f, %.3f].",
            _format_window(requested_window),
            movie_tmin,
            movie_tmax,
        )

    try:
        stc = stc.copy().crop(tmin=movie_tmin, tmax=movie_tmax)
        movie_tmin = float(stc.times[0])
        movie_tmax = float(stc.times[-1])
    except Exception as exc:  # pragma: no cover - defensive
        log.warning(
            "Failed to crop STC to %.3f-%.3f s; proceeding with full range (%s).",
            movie_tmin,
            movie_tmax,
            exc,
        )
        stc = stc.copy()

    if stc.data.size == 0:
        log.warning("Grand-average STC contains no samples after cropping; skipping movies.")
        return {}

    stc_binned, frame_times = _downsample_stc(stc, n_frames)
    if frame_times.size == 0:
        log.warning("No time samples available for movie export; skipping.")
        return {}

    source_movie_path = output_dir / "movie_source.mp4"
    _render_source_movie(stc_binned, frame_times, source_movie_path, fps=fps, codec=codec)

    results: Dict[str, str] = {"source": str(source_movie_path)}

    topo_movie_path: Optional[Path] = None
    if include_topo:
        sensor_evoked = _resolve_sensor_evoked(config, output_dir, derivatives_root, logger=log)
        if sensor_evoked is None:
            log.warning(
                "Sensor grand average not found for report group %s; skipping topomap movie.",
                config.get("report_group"),
            )
        else:
            topo_movie_path = output_dir / "movie_topo.mp4"

            evoked_window = (float(sensor_evoked.times[0]), float(sensor_evoked.times[-1]))
            if sensor_evoked.times.size > 1:
                evoked_tstep = float(sensor_evoked.times[1] - sensor_evoked.times[0])
            else:
                evoked_tstep = float(1.0 / sensor_evoked.info.get("sfreq", 1.0))

            topo_tmin, topo_tmax = _clamp_window(
                (movie_tmin, movie_tmax),
                evoked_window,
                evoked_tstep,
            )

            if topo_tmin != movie_tmin or topo_tmax != movie_tmax:
                log.info(
                    "Sensor movie window clamped to available range [%.3f, %.3f].",
                    topo_tmin,
                    topo_tmax,
                )

            topo_frame_times = frame_times[(frame_times >= topo_tmin) & (frame_times <= topo_tmax)]
            if topo_frame_times.size == 0:
                topo_frame_times = np.array([topo_tmax], dtype=float)

            aligned_times = _align_topo_times(sensor_evoked, topo_frame_times)
            _render_topo_movie(sensor_evoked, aligned_times, topo_movie_path, fps=fps, codec=codec)
            results["topo"] = str(topo_movie_path)

    if stitch_movies and topo_movie_path is not None and topo_movie_path.exists():
        stitched_path = output_dir / "movie_source_topo.mp4"
        stitched = _stitch_movies(source_movie_path, topo_movie_path, stitched_path, codec=codec)
        if stitched is not None:
            results["stitched"] = str(stitched)

    return results


def _downsample_stc(stc: mne.SourceEstimate, n_frames: int) -> tuple[mne.SourceEstimate, np.ndarray]:
    """Reduce an STC to at most ``n_frames`` time points via window averaging."""

    stc_copy = stc.copy()
    n_times = stc_copy.data.shape[1]
    if n_frames <= 0 or n_times <= n_frames:
        return stc_copy, stc_copy.times.copy()

    edges = np.linspace(0, n_times, num=n_frames + 1, dtype=int)
    new_data = np.zeros((stc_copy.data.shape[0], n_frames), dtype=stc_copy.data.dtype)
    new_times = np.zeros(n_frames, dtype=float)
    times = stc_copy.times

    for idx in range(n_frames):
        start = edges[idx]
        stop = edges[idx + 1]
        if start == stop:
            stop = min(start + 1, n_times)
        segment = stc_copy.data[:, start:stop]
        if segment.size == 0:
            segment = stc_copy.data[:, start:start + 1]
            stop = min(start + 1, n_times)
        new_data[:, idx] = segment.mean(axis=1)
        new_times[idx] = times[start:stop].mean()

    tstep = new_times[1] - new_times[0] if n_frames > 1 else stc_copy.tstep
    reduced = mne.SourceEstimate(
        new_data,
        vertices=stc_copy.vertices,
        tmin=float(new_times[0]),
        tstep=float(tstep),
        subject=stc_copy.subject,
    )
    return reduced, new_times


def _normalize_window(window: Optional[Iterable[float]], fallback: Tuple[float, float]) -> Tuple[float, float]:
    """Return a sanitized (tmin, tmax) tuple from YAML or fallback values."""

    if window is None:
        return fallback

    try:
        values = [float(window[0]), float(window[1])]
    except Exception:
        return fallback

    if values[0] == values[1]:
        values[1] += 1e-6

    if values[0] > values[1]:
        values = [values[1], values[0]]

    return float(values[0]), float(values[1])


def _clamp_window(
    requested: Tuple[float, float],
    available: Tuple[float, float],
    tstep: float,
) -> Tuple[float, float]:
    """Clamp the requested time window to the available range."""

    req_start, req_end = requested
    avail_start, avail_end = available

    start = max(req_start, avail_start)
    end = min(req_end, avail_end)

    if end <= start:
        end = avail_end
        start = max(avail_start, avail_end - max(tstep, 1e-6))
        if end <= start:
            start = avail_start

    start = max(avail_start, min(start, avail_end))
    end = max(start + 1e-6, min(end, avail_end))

    return float(start), float(end)


def _format_window(window: Tuple[float, float]) -> str:
    return f"[{window[0]:.3f}, {window[1]:.3f}]"


def _render_source_movie(
    stc: mne.SourceEstimate,
    frame_times: np.ndarray,
    output_path: Path,
    *,
    fps: int,
    codec: str,
) -> None:
    """Render a PyVista-based movie for the STC across ``frame_times``."""

    log = LOG
    subjects_dir, _ = _check_fsaverage_path()

    brain = stc.plot(
        subject=stc.subject or "fsaverage",
        subjects_dir=subjects_dir,
        hemi="both",
        surface="white",
        cortex="high_contrast",
        smoothing_steps=10,
        time_viewer=False,
        background="white",
        foreground="black",
        colorbar=True,
        show_traces=False,
    )

    try:
        actual_tmin = float(stc.times[0])
        actual_tmax = float(stc.times[-1])
        clipped_times = np.clip(frame_times, actual_tmin, actual_tmax)
        if clipped_times.size == 0:
            clipped_times = np.array([actual_tmax], dtype=float)

        movie_kwargs = dict(
            filename=str(output_path),
            tmin=float(clipped_times[0]),
            tmax=float(clipped_times[-1]),
            framerate=fps,
            time_dilation=1.0,
            interpolation="linear",
        )
        if codec:
            movie_kwargs["codec"] = codec
        try:
            brain.save_movie(**movie_kwargs)
        except TypeError:
            movie_kwargs.pop("codec", None)
            brain.save_movie(**movie_kwargs)
    except Exception as exc:  # pragma: no cover - plotting backend variances
        log.warning("Failed to render source movie to %s: %s", output_path, exc)
        output_path.unlink(missing_ok=True)
    finally:
        brain.close()


def _render_topo_movie(
    evoked: mne.Evoked,
    frame_times: np.ndarray,
    output_path: Path,
    *,
    fps: int,
    codec: str,
) -> None:
    """Render a topomap animation using the supplied ``Evoked`` object."""

    import matplotlib.pyplot as plt

    log = LOG

    try:
        anim = evoked.animate_topomap(
            times=frame_times,
            frame_rate=fps,
            ch_type="eeg",
            blit=False,
            time_unit="s",
            show=False,
        )
        fig = getattr(anim, "_fig", None)
        if fig is None and isinstance(anim, tuple) and len(anim) == 2:
            fig, anim = anim
        save_kwargs = dict(fps=fps)
        if codec:
            save_kwargs["codec"] = codec
        try:
            anim.save(str(output_path), **save_kwargs)
        except TypeError:
            save_kwargs.pop("codec", None)
            anim.save(str(output_path), **save_kwargs)
        finally:
            if fig is not None:
                plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting backend variances
        log.warning("Failed to render topomap movie to %s: %s", output_path, exc)
        output_path.unlink(missing_ok=True)


def _stitch_movies(
    source_path: Path,
    topo_path: Path,
    output_path: Path,
    *,
    codec: str,
) -> Optional[Path]:
    """Combine the source and topomap movies side-by-side using ffmpeg."""

    log = LOG
    if shutil.which("ffmpeg") is None:
        log.info("ffmpeg not available on PATH; skipping stitched movie export.")
        return None

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-i",
        str(topo_path),
        "-filter_complex",
        "hstack=inputs=2",
        "-c:v",
        codec,
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as exc:  # pragma: no cover - environment-dependent
        log.warning("Failed to stitch source/topo movies with ffmpeg: %s", exc)
        output_path.unlink(missing_ok=True)
        return None

    return output_path


def _resolve_sensor_evoked(
    config: dict,
    output_dir: Path,
    derivatives_root: Path | str | None,
    *,
    logger: Optional[logging.Logger] = None,
) -> Optional[mne.Evoked]:
    """Locate and load the matching sensor grand-average for topo movies."""

    log = logger or LOG
    report_group = config.get("report_group")
    if not report_group:
        log.warning("report_group missing in config; cannot resolve sensor movie input.")
        return None

    if derivatives_root is None:
        # output_dir := <derivatives>/source/<timestamp-analysis>
        try:
            derivatives_root = output_dir.parent.parent
        except IndexError:  # pragma: no cover - defensive
            derivatives_root = output_dir

    sensor_root = Path(derivatives_root) / "sensor"
    if not sensor_root.exists():
        return None

    pattern = f"*-sensor_{report_group}"
    candidates = sorted(sensor_root.glob(pattern), reverse=True)
    for candidate in candidates:
        for suffix in ("-evoked.fif", "-ave.fif", "_evoked.fif", "_ave.fif"):
            evoked_path = candidate / f"sensor_{report_group}_grand_average{suffix}"
            if not evoked_path.exists():
                continue
            try:
                evoked = mne.read_evokeds(str(evoked_path), condition=0, verbose=False)
                return evoked
            except Exception as exc:
                log.warning("Failed to load evoked data from %s: %s", evoked_path, exc)
                continue

    return None


def _align_topo_times(evoked: mne.Evoked, frame_times: Iterable[float]) -> np.ndarray:
    """Map STC frame times onto the closest available Evoked samples."""

    evoked_times = evoked.times
    frame_times = np.asarray(list(frame_times), dtype=float)
    if frame_times.size == 0:
        return frame_times

    indices = np.searchsorted(evoked_times, frame_times, side="left")
    indices = np.clip(indices, 0, len(evoked_times) - 1)

    for idx, target in enumerate(frame_times):
        current = indices[idx]
        prev = max(current - 1, 0)
        if abs(evoked_times[current] - target) >= abs(evoked_times[prev] - target):
            indices[idx] = prev

    aligned = evoked_times[indices]
    return aligned

