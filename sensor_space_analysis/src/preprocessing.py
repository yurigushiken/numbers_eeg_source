"""Preprocessing helpers for trial-level EEG–behavior dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd


def identify_rt_outliers(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask flagging extreme RTs for reporting.

    Conservative rule: mark RT < 100 ms or RT > 3000 ms as outliers.
    Does not remove rows.
    """
    rt = df['rt'].astype(float)
    mask = (rt < 100.0) | (rt > 3000.0)
    return mask.fillna(False)


def subject_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-subject summary metrics.

    - n_trials_acc0 / n_trials_acc1
    - mean_rt_acc0 / mean_rt_acc1
    - mean_left_temp_acc0 / mean_left_temp_acc1
    - accuracy_rate
    """
    # Ensure types
    data = df.copy()
    data['accuracy'] = data['accuracy'].astype(int)
    data['rt'] = data['rt'].astype(float)
    data['left_temp_amp'] = data['left_temp_amp'].astype(float)

    grp = data.groupby(['subject_id', 'accuracy'])
    counts = grp.size().unstack(fill_value=0)
    rt_mean = grp['rt'].mean().unstack()
    amp_mean = grp['left_temp_amp'].mean().unstack()

    out = pd.DataFrame(index=counts.index).reset_index()
    out['n_trials_acc0'] = counts.get(0, 0).values
    out['n_trials_acc1'] = counts.get(1, 0).values
    out['mean_rt_acc0'] = rt_mean.get(0).fillna(0.0).values
    out['mean_rt_acc1'] = rt_mean.get(1).fillna(0.0).values
    out['mean_left_temp_acc0'] = amp_mean.get(0).fillna(0.0).values
    out['mean_left_temp_acc1'] = amp_mean.get(1).fillna(0.0).values
    total = out['n_trials_acc0'] + out['n_trials_acc1']
    out['accuracy_rate'] = out['n_trials_acc1'] / total.replace(0, np.nan)
    out['accuracy_rate'] = out['accuracy_rate'].fillna(0.0)
    return out.rename(columns={'subject_id': 'subject_id'})


def validate_trial_counts(df: pd.DataFrame, min_trials: int = 10) -> bool:
    """Validate that each subject has at least `min_trials` per accuracy condition."""
    counts = df.groupby(['subject_id', 'accuracy']).size().unstack(fill_value=0)
    ok0 = (counts.get(0, 0) >= min_trials).all()
    ok1 = (counts.get(1, 0) >= min_trials).all()
    return bool(ok0 and ok1)


def center_within_subject(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a copy of df with within-subject centered columns (suffix `_cws`)."""
    out = df.copy()
    for col in columns:
        mu = out.groupby('subject_id')[col].transform('mean')
        out[f'{col}_cws'] = out[col].astype(float) - mu.astype(float)
    return out

