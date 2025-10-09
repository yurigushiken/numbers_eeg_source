"""Utility script to compare preprocessing variants for sensor/source pipelines.

This script calculates simple signal-to-noise metrics for the All Direction
contrast across different preprocessed datasets. The goal is to provide an
evidence-based way to decide which preprocessing parameters yield stronger
N1/P3b effects (important for dSPM and eLORETA localization).

Run with:
    python temp_tests/evaluate_preprocessing_variants.py --config configs/all_direction_effect/sensor_all_direction_effect.yaml --accuracy acc1

Outputs a CSV and Markdown summary under temp_tests/ with per-dataset metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.utils import data_loader  # type: ignore  # noqa: E402
from code.utils.electrodes import ELECTRODE_GROUPS  # type: ignore  # noqa: E402
DATA_PREPROCESSED_DIR = PROJECT_ROOT / "data" / "data_preprocessed"


@dataclass
class RoiMetric:
    peak_uv: float
    snr: float


def _get_roi_indices(ch_names: List[str], roi: Iterable[str]) -> List[int]:
    """Return the indices for ROI channels present in the data."""

    return [ch_names.index(ch) for ch in roi if ch in ch_names]


def _compute_roi_metrics(evoked, roi_indices: List[int], time_mask: np.ndarray, baseline_mask: np.ndarray, polarity: str) -> RoiMetric:
    if not roi_indices:
        return RoiMetric(np.nan, np.nan)

    data = evoked.data[roi_indices, :]
    baseline_std = float(np.std(data[:, baseline_mask]))
    if polarity == "negative":
        peak = float(np.mean(np.min(data[:, time_mask], axis=1)))
    else:
        peak = float(np.mean(np.max(data[:, time_mask], axis=1)))

    if baseline_std == 0:
        snr = np.nan
    else:
        snr = abs(peak) / baseline_std

    # Convert to microvolts for readability
    return RoiMetric(peak_uv=peak * 1e6, snr=snr)


def evaluate_preprocessing_variants(
    config_path: Path,
    accuracy: str = "acc1",
    datasets: Iterable[str] | None = None,
    max_subjects: int | None = None,
) -> pd.DataFrame:
    """Evaluate ROI metrics for each preprocessing folder.

    Parameters
    ----------
    config_path : Path
        Sensor-space YAML defining the contrast to evaluate.
    accuracy : str
        Accuracy filter ("all" or "acc1").
    datasets : iterable of str, optional
        Specific preprocessing folders to analyse. If None, all folders
        under data/data_preprocessed are used.
    max_subjects : int, optional
        Limit the number of subjects per dataset for quicker diagnostics.

    Returns
    -------
    DataFrame with aggregated metrics per dataset.
    """

    config = data_loader.load_config(str(config_path), project_root=PROJECT_ROOT)

    epoch_cfg = config.get("epoch_window", {})
    baseline = tuple(epoch_cfg.get("baseline", (-0.2, 0.0)))

    def _roi_channels(key: str) -> List[str]:
        entry = ELECTRODE_GROUPS.get(key, {})
        if isinstance(entry, dict):
            channels: List[str] = []
            for region in entry.values():
                channels.extend(region.get("electrodes", []))
            return channels
        if isinstance(entry, (list, tuple)):
            return list(entry)
        return []

    times_windows = {
        "N1": (0.08, 0.20, "negative", _roi_channels("N1")),
        "P3b": (0.30, 0.50, "positive", _roi_channels("P3b")),
    }

    if datasets is None:
        datasets = sorted(
            d.name for d in DATA_PREPROCESSED_DIR.iterdir() if d.is_dir()
        )

    results = []
    for dataset in datasets:
        custom_path = Path("data") / "data_preprocessed" / dataset
        try:
            subject_dirs = data_loader.get_subject_dirs(
                accuracy,
                project_root=PROJECT_ROOT,
                data_source=str(custom_path),
            )
        except Exception as exc:
            results.append(
                {
                    "dataset": dataset,
                    "error": str(exc),
                }
            )
            continue

        if not subject_dirs:
            results.append(
                {
                    "dataset": dataset,
                    "n_subjects": 0,
                    "mean_trials_A": np.nan,
                    "mean_trials_B": np.nan,
                    "mean_N1_peak_uv": np.nan,
                    "mean_P3b_peak_uv": np.nan,
                    "mean_N1_snr": np.nan,
                    "mean_P3b_snr": np.nan,
                }
            )
            continue

        condA = config["contrast"]["condition_A"]
        condB = config["contrast"]["condition_B"]

        subject_rows = []
        for subject_dir in subject_dirs[: max_subjects or len(subject_dirs)]:
            contrast_evoked, _ = data_loader.create_subject_contrast(
                subject_dir, config, accuracy=accuracy
            )
            if contrast_evoked is None:
                continue

            ch_names = contrast_evoked.ch_names
            times = contrast_evoked.times
            baseline_mask = (times >= baseline[0]) & (times < baseline[1])

            roi_metrics = {}
            for name, (tmin, tmax, polarity, roi_channels) in times_windows.items():
                time_mask = (times >= tmin) & (times <= tmax)
                indices = _get_roi_indices(ch_names, roi_channels)
                roi_metrics[name] = _compute_roi_metrics(
                    contrast_evoked, indices, time_mask, baseline_mask, polarity
                )

            # Trial counts per condition
            n_trials_A = 0
            n_trials_B = 0
            try:
                evoked_A, epochs_A = data_loader.get_evoked_for_condition(
                    subject_dir,
                    condA,
                    baseline=baseline,
                    accuracy=accuracy,
                )
                if epochs_A:
                    n_trials_A = sum(len(ep) for ep in epochs_A)
            except Exception:
                pass

            try:
                evoked_B, epochs_B = data_loader.get_evoked_for_condition(
                    subject_dir,
                    condB,
                    baseline=baseline,
                    accuracy=accuracy,
                )
                if epochs_B:
                    n_trials_B = sum(len(ep) for ep in epochs_B)
            except Exception:
                pass

            subject_rows.append(
                {
                    "subject": subject_dir.name,
                    "n_trials_A": n_trials_A,
                    "n_trials_B": n_trials_B,
                    "N1_peak_uv": roi_metrics["N1"].peak_uv,
                    "P3b_peak_uv": roi_metrics["P3b"].peak_uv,
                    "N1_snr": roi_metrics["N1"].snr,
                    "P3b_snr": roi_metrics["P3b"].snr,
                }
            )

        if not subject_rows:
            results.append(
                {
                    "dataset": dataset,
                    "n_subjects": 0,
                    "mean_trials_A": np.nan,
                    "mean_trials_B": np.nan,
                    "mean_N1_peak_uv": np.nan,
                    "mean_P3b_peak_uv": np.nan,
                    "mean_N1_snr": np.nan,
                    "mean_P3b_snr": np.nan,
                }
            )
            continue

        df_sub = pd.DataFrame(subject_rows)
        results.append(
            {
                "dataset": dataset,
                "n_subjects": int(df_sub.shape[0]),
                "mean_trials_A": float(df_sub["n_trials_A"].mean()),
                "mean_trials_B": float(df_sub["n_trials_B"].mean()),
                "mean_N1_peak_uv": float(df_sub["N1_peak_uv"].mean()),
                "mean_P3b_peak_uv": float(df_sub["P3b_peak_uv"].mean()),
                "mean_N1_snr": float(df_sub["N1_snr"].mean()),
                "mean_P3b_snr": float(df_sub["P3b_snr"].mean()),
            }
        )

    df = pd.DataFrame(results)
    df["rank_score"] = (
        df[["mean_N1_snr", "mean_P3b_snr"]]
        .fillna(0)
        .mean(axis=1, skipna=True)
    )
    df = df.sort_values("rank_score", ascending=False, na_position="last")
    return df


def _write_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "preprocessing_metrics.csv"
    md_path = output_dir / "preprocessing_metrics.md"

    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate preprocessing variants.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/all_direction_effect/sensor_all_direction_effect.yaml"),
        help="Sensor-space YAML to evaluate.",
    )
    parser.add_argument(
        "--accuracy",
        type=str,
        default="acc1",
        choices=["all", "acc1"],
        help="Accuracy filter to apply.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional subset of preprocessing folders to evaluate.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Limit number of subjects per dataset for faster diagnostics.",
    )

    args = parser.parse_args()

    df = evaluate_preprocessing_variants(
        config_path=args.config,
        accuracy=args.accuracy,
        datasets=args.datasets,
        max_subjects=args.max_subjects,
    )
    _write_outputs(df, PROJECT_ROOT / "temp_tests")

    print("Saved results to temp_tests/preprocessing_metrics.csv and .md")
    print(df.head())


if __name__ == "__main__":
    main()
