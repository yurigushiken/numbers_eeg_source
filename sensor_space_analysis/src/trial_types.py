"""Trial-type classification and analysis for left temporal activity."""
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


def classify_trials_by_rt(df_trials: pd.DataFrame) -> pd.DataFrame:
    """Classify trials within each subject based on RT quartiles for correct trials.

    Adds columns: trial_type (error / fast_correct / medium_fast_correct / medium_slow_correct / slow_correct)
    """
    d = df_trials.copy()
    d['trial_type'] = ''
    # Mark errors first
    d.loc[d['accuracy'] == 0, 'trial_type'] = 'error'

    # Process correct trials within each subject
    def _label_correct(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        corr = sub[sub['accuracy'] == 1]
        if len(corr) >= 4:
            q25, q50, q75 = np.percentile(corr['rt'], [25, 50, 75])
            bins = [(-np.inf, q25), (q25, q50), (q50, q75), (q75, np.inf)]
            labels = ['fast_correct', 'medium_fast_correct', 'medium_slow_correct', 'slow_correct']
            for (lo, hi), lab in zip(bins, labels):
                idx = (sub['accuracy'] == 1) & (sub['rt'] > lo) & (sub['rt'] <= hi)
                sub.loc[idx, 'trial_type'] = lab
        else:
            sub.loc[sub['accuracy'] == 1, 'trial_type'] = 'fast_correct'
        return sub

    d = d.groupby('subject_id', group_keys=False).apply(_label_correct)
    return d


def summarize_by_trial_type(trials: pd.DataFrame) -> pd.DataFrame:
    by = trials.groupby('trial_type')
    out = by['left_temp_amp'].agg(['mean', 'sem', 'count']).reset_index()
    out.rename(columns={'sem': 'se', 'count': 'n_trials'}, inplace=True)
    return out


def success_rate_by_amplitude(trials: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    # Bin by left_temp_amp
    binned, edges = pd.qcut(trials['left_temp_amp'], q=bins, retbins=True, duplicates='drop')
    grp = trials.groupby(binned)['accuracy']
    prop = grp.mean()
    n = grp.size()
    # Wilson score interval for binomial CI
    z = 1.96
    phat = prop.values
    nn = n.values.astype(float)
    denom = 1 + z**2 / nn
    center = (phat + z**2 / (2*nn)) / denom
    half = z * np.sqrt((phat*(1-phat) + z**2/(4*nn)) / nn) / denom
    lo = center - half
    hi = center + half
    centers = [interval.mid for interval in prop.index.categories]
    return pd.DataFrame({'bin_center': centers, 'prop_correct': prop.values, 'n': n.values, 'ci_low': lo, 'ci_high': hi})


def anova_trial_type(trials: pd.DataFrame) -> pd.DataFrame:
    groups = [g['left_temp_amp'].values for _, g in trials.groupby('trial_type')]
    F, p = stats.f_oneway(*groups)
    # eta-squared
    all_vals = trials['left_temp_amp'].values
    grand = np.mean(all_vals)
    ss_between = sum([len(g)*(np.mean(g)-grand)**2 for g in groups])
    ss_total = np.sum((all_vals - grand)**2)
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    return pd.DataFrame({'F': [F], 'p': [p], 'eta2': [eta2], 'k_levels': [len(groups)], 'n_total': [len(all_vals)]})


def generate_trial_type_figures(trials: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_context('talk')

    # 01 amplitude by trial type (violin)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    order = ['error', 'slow_correct', 'medium_slow_correct', 'medium_fast_correct', 'fast_correct']
    sns.violinplot(data=trials, x='trial_type', y='left_temp_amp', order=order, ax=ax, inner='box')
    ax.set_xlabel('Trial type')
    ax.set_ylabel('Left temporal amplitude (µV)')
    fig.tight_layout()
    fig.savefig(figures_dir / '01_amplitude_by_trial_type.png')
    plt.close(fig)

    # 02 amplitude vs RT scatter with KDE contours for correct
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    sns.scatterplot(data=trials, x='left_temp_amp', y='rt', hue='trial_type', alpha=0.6, s=12, ax=ax)
    corr = trials[trials['accuracy'] == 1]
    try:
        sns.kdeplot(data=corr, x='left_temp_amp', y='rt', levels=5, color='k', linewidths=1.0, ax=ax)
    except Exception:
        pass
    ax.set_xlabel('Left temporal amplitude (µV)')
    ax.set_ylabel('Reaction Time (ms)')
    fig.tight_layout()
    fig.savefig(figures_dir / '02_amplitude_vs_RT_scatter.png')
    plt.close(fig)

    # 03 success rate by amplitude (deciles)
    sr = success_rate_by_amplitude(trials, bins=10)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(sr['bin_center'], sr['prop_correct'], marker='o')
    ax.fill_between(sr['bin_center'], sr['ci_low'], sr['ci_high'], alpha=0.2)
    ax.set_xlabel('Left temporal amplitude (bin centers)')
    ax.set_ylabel('Proportion correct (95% CI)')
    fig.tight_layout()
    fig.savefig(figures_dir / '03_success_rate_by_amplitude.png')
    plt.close(fig)

    # 04 subject patterns small multiples
    subjects = sorted(trials['subject_id'].unique())
    n = len(subjects)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 2.5), dpi=150, sharex=True, sharey=True)
    axes = axes.ravel()
    for i, sid in enumerate(subjects):
        sub = trials[trials['subject_id'] == sid]
        ax = axes[i]
        sns.scatterplot(data=sub, x='left_temp_amp', y='rt', hue='accuracy', palette='Set1', s=10, ax=ax, legend=False)
        ax.set_title(f"S{sub} {sid}", fontsize=9)
        ax.grid(True, alpha=0.2)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    fig.tight_layout()
    fig.savefig(figures_dir / '04_subject_patterns.png')
    plt.close(fig)

    # 05 mixture model (optional)
    try:
        from sklearn.mixture import GaussianMixture
        corr = trials[trials['accuracy'] == 1]
        X = corr[['left_temp_amp', 'rt']].dropna().values
        if len(X) >= 10:
            gmm = GaussianMixture(n_components=3, random_state=42)
            labels = gmm.fit_predict(X)
            corr2 = corr.dropna(subset=['left_temp_amp', 'rt']).copy()
            corr2['cluster'] = labels
            fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
            sns.scatterplot(data=corr2, x='left_temp_amp', y='rt', hue='cluster', ax=ax)
            fig.tight_layout()
            fig.savefig(figures_dir / '05_mixture_model.png')
            plt.close(fig)
    except Exception:
        pass


def save_trial_type_outputs(trials: pd.DataFrame, summary: pd.DataFrame, anova_df: pd.DataFrame, out_dir: Path) -> None:
    (out_dir / 'data').mkdir(parents=True, exist_ok=True)
    trials.to_csv(out_dir / 'data' / 'trial_classification.csv', index=False)
    summary.to_csv(out_dir / 'data' / 'trial_type_summary.csv', index=False)
    anova_df.to_csv(out_dir / 'data' / 'anova_results.csv', index=False)

