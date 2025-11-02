from __future__ import annotations

import logging
from typing import Sequence

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import mne
import numpy as np
import imageio.v2 as imageio

log = logging.getLogger("topo_movie")


def _load_thumbnail_image(image_path, *, cache):
    if image_path in cache:
        return cache[image_path]
    try:
        data = imageio.imread(image_path)
    except FileNotFoundError:
        log.warning("Thumbnail image not found: %s", image_path)
        cache[image_path] = None
        return None
    except Exception as exc:
        log.warning("Failed to load thumbnail image %s: %s", image_path, exc)
        cache[image_path] = None
        return None
    cache[image_path] = data
    return data


def _clamp(value: float, minimum: float, maximum: float) -> float:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def _mirror_position(
    base_position,
    *,
    width: float,
    height: float,
    units: str,
    padding: float = 0.0,
):
    """Mirror [x, y] across the horizontal center in the same coord system.
    axes units: allow values outside [0,1]; figure units: clamp to canvas.
    """
    x0 = float(base_position[0]); y0 = float(base_position[1])
    units_lc = str(units).lower()
    mirrored_x = 1.0 - x0 - float(width) + float(padding)
    mirrored_y = y0
    if units_lc == "figure":
        mirrored_x = _clamp(mirrored_x, 0.0, 1.0 - float(width))
        mirrored_y = _clamp(mirrored_y, 0.0, 1.0 - float(height))
    return [mirrored_x, mirrored_y]


def _render_frame(
    evokeds: Sequence[mne.Evoked],
    time_index: int,
    scales: Sequence[tuple[float, float]],
    labels: Sequence[str],
    thumbnails: Sequence[dict[str, object]],
    centerpiece,  # np.ndarray | None
    centerpiece_options: dict[str, object],
    frame_path,
    title: str | None,
    layout: tuple[int, int],
):
    rows, cols = layout
    if rows <= 0 or cols <= 0:
        raise ValueError("Panel layout must have positive row and column counts")

    figsize = (4.0 * cols, 3.4 * rows)
    if rows == 2 and cols == 2 and len(evokeds) == 3:
        fig = plt.figure(figsize=figsize, dpi=180)
        gs = fig.add_gridspec(2, 2)
        ax_top_left = fig.add_subplot(gs[0, 0])
        ax_top_right = fig.add_subplot(gs[0, 1])
        ax_bottom_center = fig.add_subplot(gs[1, :])
        flat_axes = [ax_top_left, ax_top_right, ax_bottom_center]
    else:
        fig = plt.figure(figsize=figsize, dpi=180)
        gs = fig.add_gridspec(rows, cols)
        flat_axes: list[plt.Axes] = []
        total_tracks = len(evokeds)
        full_rows = min(total_tracks // cols, rows)
        track_counter = 0
        for r in range(full_rows):
            for c in range(cols):
                if track_counter >= total_tracks:
                    break
                ax = fig.add_subplot(gs[r, c])
                flat_axes.append(ax)
                track_counter += 1
        if track_counter < total_tracks and full_rows < rows:
            remainder = total_tracks - track_counter
            sub_spec = gs[full_rows, :]
            sub_gs = sub_spec.subgridspec(1, remainder)
            for c in range(remainder):
                ax = fig.add_subplot(sub_gs[0, c])
                flat_axes.append(ax)
                track_counter += 1

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
        cbar.set_ticks(np.arange(int(scales[idx][0]), int(scales[idx][1]) + 1, 1))
        cbar.set_label("Amplitude (µV)")

        thumb_entry = thumbnails[idx] if idx < len(thumbnails) else None
        if isinstance(thumb_entry, dict):
            for thumb in thumb_entry.get("images", []):
                img = thumb.get("image")
                if img is None:
                    continue
                width = float(thumb.get("width", 0.32))
                height = float(thumb.get("height", width))
                width = max(min(width, 0.95), 0.05)
                height = max(min(height, 0.95), 0.05)
                position = thumb.get("position")
                units = str(thumb.get("position_units", "axes")).lower()
                if isinstance(position, Sequence) and len(position) == 2:
                    try:
                        x0 = float(position[0]); y0 = float(position[1])
                    except Exception:
                        x0, y0 = 0.05, 0.05
                    if units == "figure":
                        inset = ax.figure.add_axes([x0, y0, width, height], zorder=ax.get_zorder() + 1)
                    else:
                        inset = ax.inset_axes([x0, y0, width, height], transform=ax.transAxes)
                        inset.set_clip_on(False)
                else:
                    loc = str(thumb.get("loc", "lower left"))
                    borderpad = float(thumb.get("borderpad", 0.6))
                    inset = inset_axes(ax, width=f"{width * 100:.0f}%", height=f"{height * 100:.0f}%", loc=loc, borderpad=borderpad)
                    inset.set_clip_on(False)
                inset.axis("off")
                inset.set_facecolor("white")
                if img.ndim == 2:
                    inset.imshow(img, cmap="gray", vmin=np.min(img), vmax=np.max(img))
                else:
                    inset.imshow(img)

                cap = thumb.get("caption")
                if cap:
                    cap_offset = thumb.get("caption_offset") or [0.0, -0.08]
                    try:
                        dx = float(cap_offset[0]); dy = float(cap_offset[1])
                    except Exception:
                        dx, dy = 0.0, -0.08
                    font_kwargs = {
                        "fontsize": float(thumb.get("caption_size", 8.0)),
                        "color": thumb.get("caption_color") or "black",
                    }
                    if bool(thumb.get("caption_italic", False)):
                        font_kwargs["fontstyle"] = "italic"
                    inset.text(0.5 + dx, dy, str(cap), ha="center", va="top", transform=inset.transAxes, **font_kwargs)

    if title:
        text = fig.suptitle(title, fontsize=13, y=0.99, ha="center")
        try:
            text.set_wrap(True)
        except AttributeError:
            pass

    if centerpiece is not None:
        c_width = float(centerpiece_options.get("width", 0.3))
        c_height = float(centerpiece_options.get("height", c_width))
        position = centerpiece_options.get("position") or [0.35, 0.35]
        units = str(centerpiece_options.get("position_units", "figure")).lower()
        try:
            x0 = float(position[0]); y0 = float(position[1])
        except Exception:
            x0, y0 = 0.35, 0.35
        c_width = max(min(c_width, 0.95), 0.05)
        c_height = max(min(c_height, 0.95), 0.05)
        if units == "axes" and len(thumbnails) > 0:
            ref_ax = fig.axes[0] if fig.axes else None
            if ref_ax is not None:
                inset = ref_ax.inset_axes([x0, y0, c_width, c_height], transform=ref_ax.transAxes)
                inset.set_clip_on(False)
            else:
                inset = fig.add_axes([x0, y0, c_width, c_height], zorder=10)
        else:
            inset = fig.add_axes([x0, y0, c_width, c_height], zorder=10)
        inset.set_xticks([]); inset.set_yticks([])
        inset.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        inset.set_facecolor("white")
        border_color = centerpiece_options.get("border_color")
        border_width = float(centerpiece_options.get("border_width", 1.0))
        patch = inset.patch
        if border_color:
            patch.set_edgecolor(border_color); patch.set_linewidth(border_width)
        else:
            patch.set_edgecolor("none")
        if centerpiece.ndim == 2:
            inset.imshow(centerpiece, cmap="gray", vmin=np.min(centerpiece), vmax=np.max(centerpiece))
        else:
            inset.imshow(centerpiece)
        caption = centerpiece_options.get("caption")
        if caption:
            offset = centerpiece_options.get("caption_offset") or [0.0, -0.05]
            try:
                dx = float(offset[0]); dy = float(offset[1])
            except Exception:
                dx, dy = 0.0, -0.05
            inset.text(
                0.5 + dx, dy, str(caption), ha="center", va="top", transform=inset.transAxes,
                fontsize=float(centerpiece_options.get("caption_size", 8)),
                color=centerpiece_options.get("caption_color") or "black",
            )

    fig.subplots_adjust(top=0.9, bottom=0.08, hspace=0.6, wspace=0.35)
    fig.savefig(frame_path, dpi=180)
    plt.close(fig)
    return frame_path


def build_track_thumbnails(
    components,
    *,
    thumbnail_root_path,
    thumbnail_metadata_key,
    thumbnail_width,
    thumbnail_height,
    thumbnail_width_clamped,
    thumbnail_height_clamped,
    thumbnail_position,
    thumbnail_position_units,
    thumbnail_loc,
    thumbnail_borderpad,
    thumbnail_cache,
    caption_enabled: bool = False,
    caption_color: str | None = None,
    caption_size: float | None = None,
    caption_italic: bool = False,
    caption_prefix_prime: str | None = None,
    caption_prefix_oddball: str | None = None,
    caption_offset: list[float] | tuple[float, float] | None = None,
    mirror_padding: float = 0.0,
):
    """Replicate current thumbnail assembly for a single track (no behavior change)."""
    thumb_entries: dict[str, object] = {"images": []}
    missing_thumbnail_keys: set[str] = set()
    if thumbnail_root_path is not None and thumbnail_metadata_key:
        prime_info: dict[str, object] | None = None
        for component in components:
            metadata = (component.get("condition") or {}).get("metadata")
            if not isinstance(metadata, dict):
                continue
            primary_candidate = metadata.get(thumbnail_metadata_key)
            if primary_candidate:
                from pathlib import Path
                candidate_path = Path(primary_candidate)
                if not candidate_path.is_absolute():
                    candidate_path = thumbnail_root_path / candidate_path
                image = _load_thumbnail_image(candidate_path, cache=thumbnail_cache)
                if image is not None:
                    base_position = thumbnail_position or [0.05, 0.05]
                    if not isinstance(base_position, (list, tuple)) or len(base_position) != 2:
                        base_position = [0.05, 0.05]
                    used_width = thumbnail_width_clamped
                    used_height = thumbnail_height_clamped
                    base_name = candidate_path.name
                    caption_text = None
                    if caption_enabled:
                        prefix = caption_prefix_prime or ""
                        caption_text = f"{prefix}{base_name}"
                    prime_info = {
                        "image": image,
                        "position": [float(base_position[0]), float(base_position[1])],
                        "position_units": thumbnail_position_units,
                        "width": used_width,
                        "height": used_height,
                        "caption": caption_text,
                        "caption_size": caption_size,
                        "caption_color": caption_color,
                        "caption_italic": caption_italic,
                        "caption_offset": caption_offset,
                    }
                    thumb_entries["images"].append(prime_info)
                else:
                    missing_thumbnail_keys.add(str(candidate_path))

            oddball_candidate = metadata.get("oddball")
            if oddball_candidate:
                from pathlib import Path
                oddball_path = Path(oddball_candidate)
                if not oddball_path.is_absolute():
                    oddball_path = thumbnail_root_path / oddball_path
                oddball_img = _load_thumbnail_image(oddball_path, cache=thumbnail_cache)
                if oddball_img is not None:
                    custom_position = metadata.get("oddball_position")
                    if isinstance(custom_position, (list, tuple)) and len(custom_position) == 2:
                        try:
                            custom_x = float(custom_position[0]); custom_y = float(custom_position[1])
                            valid_custom = True
                        except Exception:
                            valid_custom = False
                        else:
                            valid_custom = True
                    else:
                        valid_custom = False

                    if valid_custom:
                        custom_units = str(metadata.get("oddball_position_units") or thumbnail_position_units).lower()
                        custom_width = float(metadata.get("oddball_width", thumbnail_width))
                        custom_height = float(metadata.get("oddball_height", thumbnail_height))
                        base_name = oddball_path.name
                        caption_text = None
                        if caption_enabled:
                            prefix = caption_prefix_oddball or ""
                            caption_text = f"{prefix}{base_name}"
                        thumb_entries["images"].append({
                            "image": oddball_img,
                            "position": [custom_x, custom_y],
                            "position_units": custom_units,
                            "width": max(min(custom_width, 0.95), 0.05),
                            "height": max(min(custom_height, 0.95), 0.05),
                            "caption": caption_text,
                            "caption_size": caption_size,
                            "caption_color": caption_color,
                            "caption_italic": caption_italic,
                            "caption_offset": caption_offset,
                        })
                        continue

                    if prime_info is not None:
                        units = str(prime_info["position_units"]).lower()
                        used_width = float(prime_info["width"])
                        used_height = float(prime_info["height"])
                        base_pos = prime_info["position"]
                        oddball_position = _mirror_position(
                            base_pos, width=used_width, height=used_height, units=units, padding=mirror_padding
                        )
                        oddball_units = units
                    else:
                        used_width = thumbnail_width_clamped
                        used_height = thumbnail_height_clamped
                        oddball_position = [0.95 - used_width, 0.05]
                        oddball_units = thumbnail_position_units

                    base_name = oddball_path.name
                    caption_text = None
                    if caption_enabled:
                        prefix = caption_prefix_oddball or ""
                        caption_text = f"{prefix}{base_name}"
                    thumb_entries["images"].append({
                        "image": oddball_img,
                        "position": oddball_position,
                        "position_units": oddball_units,
                        "width": used_width,
                        "height": used_height,
                        "caption": caption_text,
                        "caption_size": caption_size,
                        "caption_color": caption_color,
                        "caption_italic": caption_italic,
                        "caption_offset": caption_offset,
                    })
                else:
                    missing_thumbnail_keys.add(str(oddball_path))

    return thumb_entries, missing_thumbnail_keys


