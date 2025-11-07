"""Visualization utilities for brain–behavior correlation analysis."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

# Headless rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def create_scatter_plot(df: pd.DataFrame, output_path: Path | str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4), dpi=150)
    sns.scatterplot(data=df, x='left_temp_amp', y='rt', hue='accuracy', palette='Set1', alpha=0.6, s=12)
    plt.xlabel('Left temporal amplitude (µV)')
    plt.ylabel('Reaction Time (ms)')
    plt.title('Trial-level RT vs Left Temporal Amplitude')
    plt.legend(title='Accuracy', loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def create_spaghetti_plot(df: pd.DataFrame, output_path: Path | str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5), dpi=150)
    for sid, sub in df.groupby('subject_id'):
        if len(sub) < 2:
            continue
        x = sub['left_temp_amp_cws'].to_numpy()
        y = sub['rt'].to_numpy()
        # Fit simple line per subject
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        xs = np.linspace(x.min(), x.max(), 10)
        ys = coef[0] * xs + coef[1]
        plt.plot(xs, ys, alpha=0.5)
    plt.xlabel('Left temporal amplitude (cws, µV)')
    plt.ylabel('Reaction Time (ms)')
    plt.title('Subject-specific slopes (spaghetti)')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def create_interaction_plot(preds: pd.DataFrame, output_path: Path | str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4), dpi=150)
    if 'accuracy' in preds.columns:
        sns.lineplot(data=preds, x='left_temp_amp_cws', y='rt_pred', hue='accuracy', palette='Set1')
    else:
        sns.lineplot(data=preds, x='left_temp_amp_cws', y='rt_pred', color='#1f77b4')
    plt.xlabel('Left temporal amplitude (cws, µV)')
    plt.ylabel('Predicted RT (ms)')
    plt.title('Interaction: Left temporal × Accuracy')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def create_forest_plot(rx: pd.DataFrame, output_path: Path | str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = rx.copy().sort_values('left_temp_amp_cws')
    plt.figure(figsize=(6, max(4, 0.25 * len(df))), dpi=150)
    y_pos = np.arange(len(df))
    plt.hlines(y=y_pos, xmin=df['left_temp_amp_cws'], xmax=df['left_temp_amp_cws'], colors='tab:blue')
    plt.plot(df['left_temp_amp_cws'], y_pos, 'o', color='tab:blue')
    plt.yticks(y_pos, df['subject_id'])
    plt.axvline(0, color='k', lw=0.8, ls='--')
    plt.xlabel('Subject slope (left_temp_amp_cws → RT)')
    plt.title('Subject-specific slopes (forest)')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def create_subject_heatmap(summ: pd.DataFrame, output_path: Path | str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Select a few columns
    cols = ['n_trials_acc0', 'n_trials_acc1', 'mean_rt_acc0', 'mean_rt_acc1', 'accuracy_rate']
    data = summ.set_index('subject_id')[cols]
    plt.figure(figsize=(6, max(4, 0.25 * len(data))), dpi=150)
    sns.heatmap(data, cmap='viridis', annot=False)
    plt.title('Subject-level summary')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def create_distribution_plots(df: pd.DataFrame, output_path: Path | str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    sns.violinplot(data=df, x='accuracy', y='left_temp_amp', ax=axes[0], inner='box')
    axes[0].set_title('Amplitude by Accuracy')
    axes[0].set_xlabel('Accuracy (0/1)')
    axes[0].set_ylabel('Left temporal amplitude (µV)')
    sns.violinplot(data=df, x='accuracy', y='rt', ax=axes[1], inner='box')
    axes[1].set_title('RT by Accuracy')
    axes[1].set_xlabel('Accuracy (0/1)')
    axes[1].set_ylabel('Reaction Time (ms)')
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
