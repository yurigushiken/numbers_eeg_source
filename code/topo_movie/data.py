from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import mne

from code.utils import data_loader


def _freeze_value(value: object) -> object:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze_value(v)) for k, v in value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_value(v) for v in value)
    return value


def _condition_cache_key(condition_cfg: dict, accuracy: str) -> tuple:
    items: list[tuple[str, object]] = []
    for key, value in condition_cfg.items():
        if key == "name":
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items.append((key, tuple(_freeze_value(v) for v in value)))
        elif isinstance(value, dict):
            items.append((key, tuple(sorted((k, _freeze_value(v)) for k, v in value.items()))))
        else:
            items.append((key, value))
    return (accuracy, tuple(sorted(items)))


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


