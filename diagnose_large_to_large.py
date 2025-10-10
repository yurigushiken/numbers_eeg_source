#!/usr/bin/env python
"""
Diagnostic script to investigate why large_to_large_direction analysis finds no significant clusters.

This will:
1. Check trial counts for each condition
2. Examine effect sizes in the ERPs
3. Compare with successful small_to_small analysis
"""

import numpy as np
import mne
from pathlib import Path
import yaml
from code.utils import data_loader

def analyze_contrast(config_path, accuracy='acc1'):
    """Analyze a contrast and report statistics."""
    print("=" * 80)
    print(f"ANALYZING: {config_path}")
    print("=" * 80)

    # Load config
    config = data_loader.load_config(config_path)
    analysis_name = config['analysis_name']

    # Get subjects
    subject_dirs = data_loader.get_subject_dirs(accuracy, data_source='new')
    print(f"\nNumber of subjects: {len(subject_dirs)}")

    # Analyze each subject
    trial_counts_A = []
    trial_counts_B = []
    contrast_amplitudes = []

    epoch_cfg = config.get('epoch_window', {})
    baseline = epoch_cfg.get('baseline')

    for i, subject_dir in enumerate(subject_dirs):
        try:
            # Get condition A trials
            evoked_A, epochs_A = data_loader.get_evoked_for_condition(
                subject_dir,
                config['contrast']['condition_A'],
                baseline=baseline,
                accuracy=accuracy,
            )

            # Get condition B trials
            evoked_B, epochs_B = data_loader.get_evoked_for_condition(
                subject_dir,
                config['contrast']['condition_B'],
                baseline=baseline,
                accuracy=accuracy,
            )

            if evoked_A and evoked_B:
                # Count trials
                n_A = sum([len(e) for e in epochs_A])
                n_B = sum([len(e) for e in epochs_B])
                trial_counts_A.append(n_A)
                trial_counts_B.append(n_B)

                # Get grand average and measure effect size
                ga_A = mne.grand_average(evoked_A)
                ga_B = mne.grand_average(evoked_B)

                # Compute contrast at posterior channels (similar to N1 bilateral)
                posterior_channels = ['E66', 'E65', 'E59', 'E60', 'E67', 'E71', 'E70',
                                     'E84', 'E76', 'E77', 'E85', 'E91', 'E90', 'E83']

                picks_A = mne.pick_channels(ga_A.info['ch_names'], include=posterior_channels,
                                            ordered=False)
                picks_B = mne.pick_channels(ga_B.info['ch_names'], include=posterior_channels,
                                            ordered=False)

                if len(picks_A) > 0 and len(picks_B) > 0:
                    # Get data in N1 window (80-200ms)
                    n1_times = (ga_A.times >= 0.08) & (ga_A.times <= 0.20)
                    data_A = ga_A.get_data(picks=picks_A)[:, n1_times].mean()
                    data_B = ga_B.get_data(picks=picks_B)[:, n1_times].mean()
                    contrast_amplitudes.append(data_A - data_B)

        except Exception as e:
            print(f"  Subject {i+1} ({subject_dir.name}): ERROR - {e}")
            continue

    # Report statistics
    print(f"\n{'='*80}")
    print(f"TRIAL COUNTS")
    print(f"{'='*80}")
    print(f"\nCondition A ({config['contrast']['condition_A']['name']}):")
    print(f"  Mean ± SD: {np.mean(trial_counts_A):.1f} ± {np.std(trial_counts_A):.1f}")
    print(f"  Range: {np.min(trial_counts_A)} - {np.max(trial_counts_A)}")
    print(f"  Per subject: {trial_counts_A}")

    print(f"\nCondition B ({config['contrast']['condition_B']['name']}):")
    print(f"  Mean ± SD: {np.mean(trial_counts_B):.1f} ± {np.std(trial_counts_B):.1f}")
    print(f"  Range: {np.min(trial_counts_B)} - {np.max(trial_counts_B)}")
    print(f"  Per subject: {trial_counts_B}")

    print(f"\n{'='*80}")
    print(f"EFFECT SIZE (Posterior ROI, N1 window 80-200ms)")
    print(f"{'='*80}")
    print(f"\nContrast amplitudes (A - B) in µV:")
    contrast_uv = np.array(contrast_amplitudes) * 1e6
    print(f"  Mean ± SD: {np.mean(contrast_uv):.2f} ± {np.std(contrast_uv):.2f} µV")
    print(f"  Range: {np.min(contrast_uv):.2f} - {np.max(contrast_uv):.2f} µV")

    # Compute Cohen's d
    if len(contrast_uv) > 0:
        cohens_d = np.mean(contrast_uv) / np.std(contrast_uv)
        print(f"  Cohen's d: {cohens_d:.3f}")

        # T-test
        from scipy import stats
        t_stat, p_val = stats.ttest_1samp(contrast_uv, 0)
        print(f"  t({len(contrast_uv)-1}) = {t_stat:.3f}, p = {p_val:.4f}")

        if p_val < 0.05:
            print(f"  *** SIGNIFICANT effect at p < 0.05 ***")
        else:
            print(f"  NOT significant at p < 0.05")

    print("\n")
    return {
        'trial_counts_A': trial_counts_A,
        'trial_counts_B': trial_counts_B,
        'contrast_amplitudes': contrast_uv,
        'cohens_d': cohens_d if len(contrast_uv) > 0 else None,
        'p_value': p_val if len(contrast_uv) > 0 else None
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DIAGNOSTIC ANALYSIS: LARGE vs SMALL TO LARGE/SMALL")
    print("="*80 + "\n")

    # Analyze large_to_large
    large_results = analyze_contrast(
        "configs/large_to_large_direction/sensor_large_to_large_direction.yaml"
    )

    # Analyze small_to_small for comparison
    small_results = analyze_contrast(
        "configs/small_to_small_direction/sensor_small_to_small_direction.yaml"
    )

    # Summary comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    print(f"\nLarge-to-Large:")
    print(f"  Mean trials/condition: {np.mean(large_results['trial_counts_A'] + large_results['trial_counts_B']):.1f}")
    print(f"  Effect size (Cohen's d): {large_results['cohens_d']:.3f}")
    print(f"  p-value: {large_results['p_value']:.4f}")

    print(f"\nSmall-to-Small:")
    print(f"  Mean trials/condition: {np.mean(small_results['trial_counts_A'] + small_results['trial_counts_B']):.1f}")
    print(f"  Effect size (Cohen's d): {small_results['cohens_d']:.3f}")
    print(f"  p-value: {small_results['p_value']:.4f}")

    print(f"\n{'='*80}")
    print("DIAGNOSIS:")
    print("="*80)

    if large_results['cohens_d'] and abs(large_results['cohens_d']) < 0.3:
        print("\n⚠️  SMALL EFFECT SIZE for large-to-large direction effect")
        print("   Effect size (|d| < 0.3) may be too small to detect with cluster-based")
        print("   permutation testing, even with liberal thresholds.")

    if np.mean(large_results['trial_counts_A'] + large_results['trial_counts_B']) < 50:
        print("\n⚠️  LOW TRIAL COUNTS for large-to-large conditions")
        print("   Fewer trials = lower SNR = harder to detect effects")

    if large_results['p_value'] and large_results['p_value'] > 0.05:
        print("\n⚠️  NO SIGNIFICANT EFFECT at sensor level")
        print("   The effect may genuinely be absent or very weak in this contrast")

    print("\n")
