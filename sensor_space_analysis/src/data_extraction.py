"""
Data extraction utilities for brain–behavior correlation analysis.

Functions extract trial-level ERP amplitudes from a pre-specified sensor
cluster (left anterior–temporal, "Cluster 2") and assemble a per-trial
DataFrame with accuracy and reaction time from metadata.

All amplitudes are reported in microvolts (µV).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, List

import numpy as np
import pandas as pd
import mne

from code.utils import data_loader


# Cluster #2 as provided by consultant
CLUSTER2_CHANNELS: List[str] = [
    'E32', 'E33', 'E34', 'E38', 'E39', 'E43', 'E44', 'E48', 'E49', 'E56', 'E57', 'E127', 'E128'
]

CARDINALITY_CODES: set[str] = {"11", "22", "33", "44", "55", "66"}


def load_subject_epochs(subject_id: str, data_dir: Path | str) -> mne.Epochs:
    """Load preprocessed epochs for one subject from combined file.

    Parameters
    ----------
    subject_id : str
        Subject ID, e.g., "02" for sub-02.
    data_dir : Path | str
        Root directory containing combined epoch files
        (`sub-XX_preprocessed-epo.fif`).

    Returns
    -------
    mne.Epochs
        Epochs object with metadata.
    """
    data_dir = Path(data_dir)
    fif_path = data_dir / f"sub-{subject_id}_preprocessed-epo.fif"
    epochs = mne.read_epochs(str(fif_path), preload=True, verbose=False)
    assert epochs.metadata is not None, "Epochs missing metadata"
    assert 'Condition' in epochs.metadata.columns, "Metadata missing 'Condition'"
    assert 'Target.ACC' in epochs.metadata.columns, "Metadata missing 'Target.ACC'"
    assert 'Target.RT' in epochs.metadata.columns, "Metadata missing 'Target.RT'"
    return epochs


def identify_no_change_trials(epochs: mne.Epochs) -> np.ndarray:
    """Return a boolean mask (length = n_epochs) for cardinality trials.

    Cardinalities are Condition in {11, 22, 33, 44, 55, 66}.
    """
    cond = epochs.metadata['Condition'].astype(str)
    return cond.isin(CARDINALITY_CODES).to_numpy()


def _pick_cluster2(info: mne.Info) -> np.ndarray:
    """Return channel indices for Cluster #2 that exist in `info`.

    Uses channel name intersection to be robust to missing channels.
    """
    present = [ch for ch in CLUSTER2_CHANNELS if ch in info['ch_names']]
    return mne.pick_channels(info['ch_names'], include=present)


def _time_mask(times: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    """Inclusive time mask for a given window (tmin, tmax)."""
    tmin, tmax = float(window[0]), float(window[1])
    return (times >= tmin) & (times <= tmax)


def extract_cluster2_data(epochs: mne.Epochs, time_window: Tuple[float, float] = (0.148, 0.364)) -> np.ndarray:
    """Extract per-trial mean amplitude (µV) over Cluster #2 within time window.

    Steps:
    - Pick Cluster #2 channels
    - Compute spatial mean across those channels
    - Compute temporal mean within the specified window

    Returns
    -------
    np.ndarray, shape (n_trials,)
        Mean amplitude in microvolts per trial.
    """
    X = epochs.get_data()  # (n_epochs, n_channels, n_times), units = Volts
    picks = _pick_cluster2(epochs.info)
    if picks.size == 0:
        raise RuntimeError("None of the Cluster #2 channels were found in this recording.")
    Xc = X[:, picks, :]
    # Spatial average
    Xc_mean = Xc.mean(axis=1)
    # Time mask
    tmask = _time_mask(epochs.times, time_window)
    if not tmask.any():
        raise ValueError("Time window yielded no samples.")
    # Temporal average
    vals_v = Xc_mean[:, tmask].mean(axis=1)
    # Convert to µV
    return vals_v * 1e6


def _interpret_accuracy_rt(acc_val, rt_val) -> tuple[int, float]:
    """Map metadata accuracy/RT to clean values for analysis.

    Rules (post cardinality filtering):
    - accuracy: 1 if >= 0.5 else 0; None/NaN => 0
    - RT: keep numeric; NaN => 0.0; negative => 0.0
    """
    try:
        acc_float = float(acc_val)
        acc = 1 if acc_float >= 0.5 else 0
    except Exception:
        acc = 0
    try:
        rt_float = float(rt_val)
        if not np.isfinite(rt_float) or rt_float < 0:
            rt_float = 0.0
    except Exception:
        rt_float = 0.0
    return acc, rt_float


def build_trial_dataframe(subject_id: str, epochs: mne.Epochs, cluster2_amps: np.ndarray) -> pd.DataFrame:
    """Construct per-trial DataFrame with subject, condition, accuracy, RT, amplitude.

    No cardinality filtering is performed here; caller decides filtering policy.
    """
    md = epochs.metadata.copy()
    cond_str = md['Condition'].astype(str).reset_index(drop=True)
    acc_vals = md['Target.ACC'].reset_index(drop=True)
    rt_vals = md['Target.RT'].reset_index(drop=True)
    mapped = [
        _interpret_accuracy_rt(a, r)
        for a, r in zip(acc_vals.tolist(), rt_vals.tolist())
    ]
    acc = pd.Series([m[0] for m in mapped], dtype='int64')
    rt = pd.Series([m[1] for m in mapped], dtype='float64')

    df = pd.DataFrame(
        {
            'subject_id': str(subject_id),
            'condition': cond_str.astype(str),
            'accuracy': acc,
            'rt': rt,
            'left_temp_amp': pd.Series(cluster2_amps, dtype='float64'),
        }
    )
    return df


def _list_subject_ids_from_combined(data_dir: Path | str) -> List[str]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob('sub-*_preprocessed-epo.fif'))
    out = []
    for f in files:
        name = f.stem  # sub-XX_preprocessed-epo
        if name.startswith('sub-'):
            sid = name.split('_')[0].replace('sub-', '')
            out.append(sid)
    return out


def aggregate_all_subjects(data_dir: Path | str, output_csv: Path | None) -> pd.DataFrame:
    """Aggregate all subjects into a trial-level DataFrame and optionally save CSV.

    Steps per subject:
    - Load combined epochs with metadata
    - Remove no-change (cardinality) trials
    - Compute Cluster #2 mean amplitude per trial
    - Build per-trial DataFrame and append
    """
    data_dir = Path(data_dir)
    # Prefer central helper to discover subjects, but fall back to scan
    try:
        subj_dirs = data_loader.get_subject_dirs('all', project_root=Path.cwd(), data_source=str(data_dir))
        subject_ids = []
        for p in subj_dirs:
            # p ends with data_dir/sub-XX
            pp = Path(p)
            sid = pp.name.replace('sub-', '')
            if sid:
                subject_ids.append(sid)
        if not subject_ids:
            subject_ids = _list_subject_ids_from_combined(data_dir)
    except Exception:
        subject_ids = _list_subject_ids_from_combined(data_dir)

    frames: List[pd.DataFrame] = []
    for sid in subject_ids:
        try:
            epochs = load_subject_epochs(sid, data_dir)
        except Exception:
            continue
        # Filter out no-change before RT analysis
        card_mask = identify_no_change_trials(epochs)
        if card_mask.any():
            keep_idx = np.flatnonzero(~card_mask)
            epochs = epochs[keep_idx]
        # Compute cluster amplitudes
        amps = extract_cluster2_data(epochs, time_window=(0.148, 0.364))
        df_subj = build_trial_dataframe(sid, epochs, amps)
        frames.append(df_subj)

    if not frames:
        raise RuntimeError("No subject data could be aggregated.")

    df_all = pd.concat(frames, axis=0, ignore_index=True)
    # Basic sanity clamps
    df_all['rt'] = df_all['rt'].clip(lower=0, upper=3000)
    df_all['left_temp_amp'] = df_all['left_temp_amp'].clip(lower=-1000, upper=1000)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(output_csv, index=False)
    return df_all

