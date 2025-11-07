"""Between-subject analysis of left temporal activity vs behavior."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def _landing_digit(cond_str: str) -> int | None:
    try:
        if len(cond_str) >= 2 and cond_str.isdigit():
            return int(cond_str[-1])
    except Exception:
        return None
    return None


def compute_subject_summary(df_trials: pd.DataFrame) -> pd.DataFrame:
    """Compute subject-level summary variables.

    Expects trial-level df with columns: subject_id, condition (str), accuracy (0/1), rt, left_temp_amp.
    """
    d = df_trials.copy()
    d['accuracy'] = d['accuracy'].astype(int)
    d['rt'] = d['rt'].astype(float)
    d['left_temp_amp'] = d['left_temp_amp'].astype(float)

    # Derive landing digit (oddball)
    d['landing'] = d['condition'].astype(str).str[-1].astype(int)
    d['is_small'] = d['landing'].between(1, 3)
    d['is_large'] = d['landing'].between(4, 6)

    # Means by accuracy within subject
    by = d.groupby(['subject_id', 'accuracy'])
    mean_amp = by['left_temp_amp'].mean().unstack()

    # Basic aggregates by subject
    by_subj = d.groupby('subject_id')
    acc_rate = by_subj['accuracy'].mean()
    rt_correct = d[d['accuracy'] == 1].groupby('subject_id')['rt']
    mean_rt_correct = rt_correct.mean()
    sd_rt_correct = rt_correct.std(ddof=1)

    # Error rate by size
    def _err_rate(mask):
        sub = d[mask]
        return sub.groupby('subject_id')['accuracy'].apply(lambda x: 1.0 - x.mean())

    err_small = _err_rate(d['is_small'])
    err_large = _err_rate(d['is_large'])
    err_diff = (err_large - err_small)

    # Counts
    counts = by['left_temp_amp'].size().unstack(fill_value=0)

    out = pd.DataFrame(index=by_subj.size().index)
    out['subject_id'] = out.index
    out['mean_left_temp_Acc0'] = mean_amp.get(0).fillna(np.nan)
    out['mean_left_temp_Acc1'] = mean_amp.get(1).fillna(np.nan)
    out['left_temp_diff'] = out['mean_left_temp_Acc0'] - out['mean_left_temp_Acc1']
    out['accuracy_rate'] = acc_rate
    out['mean_RT_correct'] = mean_rt_correct
    out['sd_RT_correct'] = sd_rt_correct
    out['error_rate_small'] = err_small
    out['error_rate_large'] = err_large
    out['error_rate_diff'] = err_diff
    out['n_trials_Acc0'] = counts.get(0, 0)
    out['n_trials_Acc1'] = counts.get(1, 0)
    out.reset_index(drop=True, inplace=True)
    return out


def _pearsonr_ci(x, y, alpha=0.05) -> Tuple[float, float, float, float]:
    r, p = stats.pearsonr(x, y)
    n = len(x)
    if n < 4 or not np.isfinite(r):
        return r, p, np.nan, np.nan
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return r, p, lo, hi


def compute_correlations(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute the 4 hypothesis-driven correlations with Bonferroni correction."""
    rows = []
    # 1: left_temp_diff vs accuracy_rate
    pairs = [
        ("left_temp_diff", "accuracy_rate", "H1: diff vs accuracy (neg)"),
        ("left_temp_diff", "error_rate_diff", "H2: diff vs error large-small (neg)"),
        ("left_temp_diff", "mean_RT_correct", "H3: diff vs RTcorrect (neg)"),
        ("mean_left_temp_Acc0", "sd_RT_correct", "H4: Acc0 amp vs RT SD (neg)"),
    ]
    for xk, yk, label in pairs:
        x = summary[xk].astype(float)
        y = summary[yk].astype(float)
        mask = x.notna() & y.notna()
        r, p, lo, hi = _pearsonr_ci(x[mask], y[mask])
        rows.append({
            'x': xk, 'y': yk, 'label': label, 'r': r, 'p': p, 'ci_low': lo, 'ci_high': hi, 'n': int(mask.sum())
        })
    res = pd.DataFrame(rows)
    # Bonferroni
    res['p_bonf'] = np.minimum(res['p'] * 4.0, 1.0)
    return res


def _effect_size_label(r: float) -> str:
    a = abs(r)
    if a < 0.3:
        return 'small'
    if a < 0.5:
        return 'medium'
    return 'large'


def generate_between_subjects_figures(summary: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_context('talk')

    # 01 correlation matrix (2 x 4 panel)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=150)
    xvars = ['left_temp_diff', 'mean_left_temp_Acc0']
    yvars = ['accuracy_rate', 'error_rate_diff', 'mean_RT_correct', 'sd_RT_correct']
    for i, xk in enumerate(xvars):
        for j, yk in enumerate(yvars):
            ax = axes[i, j]
            sub = summary[[xk, yk]].dropna()
            sns.regplot(data=sub, x=xk, y=yk, ax=ax, scatter_kws={'s': 30, 'alpha': 0.8}, line_kws={'color': 'k'})
            r, p = stats.pearsonr(sub[xk], sub[yk])
            ax.set_title(f"r={r:.2f}, p={p:.3f}")
            ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_dir / '01_correlation_matrix.png')
    plt.close(fig)

    # 02 subject profiles heatmap (z-scored)
    cols = ['mean_left_temp_Acc0', 'mean_left_temp_Acc1', 'left_temp_diff', 'accuracy_rate',
            'error_rate_small', 'error_rate_large', 'error_rate_diff', 'mean_RT_correct', 'sd_RT_correct']
    prof = summary[cols].copy()
    prof_z = (prof - prof.mean()) / prof.std(ddof=0)
    order = np.argsort(summary['left_temp_diff'].values)
    prof_z = prof_z.iloc[order]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    sns.heatmap(prof_z, cmap='coolwarm', center=0.0, ax=ax)
    ax.set_title('Subject profiles (z-scored), sorted by left_temp_diff')
    fig.tight_layout()
    fig.savefig(figures_dir / '02_subject_profiles.png')
    plt.close(fig)

    # 03 group comparison (top/bottom 8 by left_temp_diff)
    ordered = summary.sort_values('left_temp_diff')
    high = ordered.head(8)  # most negative (high verbal)
    low = ordered.tail(8)   # least negative
    measures = ['accuracy_rate', 'error_rate_diff', 'mean_RT_correct']
    means = []
    ses = []
    labels = []
    grp = []
    for m in measures:
        means.extend([high[m].mean(), low[m].mean()])
        ses.extend([high[m].sem(ddof=1), low[m].sem(ddof=1)])
        labels.extend([m, m])
        grp.extend(['High verbal (most negative 8)', 'Low verbal (least negative 8)'])
    bar = pd.DataFrame({'measure': labels, 'group': grp, 'mean': means, 'se': ses})
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    sns.barplot(data=bar, x='measure', y='mean', hue='group', ax=ax, errorbar=None)
    # add error bars
    for i, row in bar.iterrows():
        ax.errorbar(i // 2 + (0 if row['group'].startswith('High') else 0.35), row['mean'], yerr=row['se'], fmt='none', color='k')
    ax.set_title('Group comparison: High verbal vs Low verbal (N=8 each)')
    ax.set_xlabel('Measure')
    ax.set_ylabel('Mean (±SE)')
    fig.tight_layout()
    fig.savefig(figures_dir / '03_group_comparison.png')
    plt.close(fig)

    # 04 key scatter: accuracy_rate vs left_temp_diff
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    sns.regplot(data=summary, x='left_temp_diff', y='accuracy_rate', ax=ax)
    ax.set_xlabel('Acc0 - Acc1 left temporal amplitude (µV)')
    ax.set_ylabel('Accuracy rate')
    ax.set_title('Accuracy vs left temporal difference (Acc0 - Acc1)')
    fig.tight_layout()
    fig.savefig(figures_dir / '04_accuracy_by_left_temp.png')
    plt.close(fig)


def save_between_subjects_outputs(summary: pd.DataFrame, corrs: pd.DataFrame, group_comp: pd.DataFrame | None, out_dir: Path) -> None:
    (out_dir / 'data').mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / 'data' / 'subject_summary.csv', index=False)
    corrs.to_csv(out_dir / 'data' / 'correlations.csv', index=False)
    if group_comp is not None:
        group_comp.to_csv(out_dir / 'data' / 'group_comparison.csv', index=False)


def compute_group_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    ordered = summary.sort_values('left_temp_diff')
    high = ordered.head(8)
    low = ordered.tail(8)
    rows = []
    for m in ['accuracy_rate', 'error_rate_diff', 'mean_RT_correct']:
        t, p = stats.ttest_ind(high[m].dropna(), low[m].dropna(), equal_var=False)
        rows.append({'measure': m, 'mean_high': high[m].mean(), 'mean_low': low[m].mean(), 't': t, 'p': p, 'n_high': len(high), 'n_low': len(low)})
    return pd.DataFrame(rows)

