# Instructions for Extended Brain-Behavior Analyses
**Date:** 2025-11-04
**Project:** Numerical Cognition EEG - Left Temporal Cluster 2
**Previous Work:** Initial brain-behavior analysis complete (`sensor_space_analysis/`)

---

## Background Context

You've completed the initial brain-behavior correlation analysis examining whether left temporal activity (Cluster 2, 148-364 ms) predicts reaction time and accuracy.

**Key Discovery:** All error trials (Acc0) have RT=0 because they represent **missed detections** (participant didn't press spacebar), not slow responses. This means:
- ✅ We CAN test within-correct-trial relationships (RT varies)
- ✅ We CAN test between-condition differences (Acc0 vs Acc1)
- ❌ We CANNOT test within-error-trial relationships (no RT variance)

**What We Found:**
- Within correct trials: More negative left temporal amplitude → longer RT (-2.3 ms/µV, p=0.025)
- Between conditions: Errors show 1.61 µV more negative amplitude than correct trials (d=-1.22, p=0.012)
- Individual differences: Large variability in brain-behavior coupling across subjects

**Now we need to extend these findings with two complementary analyses.**

---

## Analysis #1: Between-Subject Analysis
**Question:** Do subjects who show stronger left temporal negativity in error trials have different behavioral profiles?

### Rationale
Since we can't test within-error RT variance (all errors have RT=0), we instead test whether **individual differences** in error-related left temporal activity predict **overall performance patterns**. This addresses the original hypothesis at the between-subject level.

### Hypothesis
Subjects who show **stronger left temporal negativity in errors** are those who **over-rely on verbal strategies**, leading to:
- Lower overall accuracy (more missed detections)
- Higher error rates specifically on large numbers (4-5-6) where verbal counting is harder
- Longer RTs on correct trials (consistent verbal processing even when successful)

### Data Structure
Create a **subject-level summary** with these variables:

```python
subject_summary = {
    'subject_id': '02',  # Subject identifier

    # Left temporal amplitude (from existing cluster 2 electrodes: 148-364 ms)
    'mean_left_temp_Acc0': -3.45,  # Average left temporal amplitude in error trials
    'mean_left_temp_Acc1': -1.84,  # Average left temporal amplitude in correct trials
    'left_temp_diff': -1.61,       # Acc0 - Acc1 (negative = more negative in errors)

    # Behavioral measures
    'accuracy_rate': 0.82,          # Proportion of correct trials (total)
    'mean_RT_correct': 485.3,       # Average RT on correct trials (ms)
    'sd_RT_correct': 152.7,         # SD of RT on correct trials (RT variability)

    # Numerosity-specific error rates
    'error_rate_small': 0.15,       # Proportion errors on trials with 1-2-3
    'error_rate_large': 0.23,       # Proportion errors on trials with 4-5-6
    'error_rate_diff': 0.08,        # Large - Small (positive = more errors on large)

    # Trial counts (for weighting if needed)
    'n_trials_Acc0': 39,
    'n_trials_Acc1': 178,
}
```

### Analysis Steps

#### Step 1: Extract Subject-Level Data
```python
# For each subject, compute:
# 1. Mean left temporal amplitude in Acc0 trials (from cluster 2 channels, 148-364 ms window)
# 2. Mean left temporal amplitude in Acc1 trials
# 3. Difference score (Acc0 - Acc1)
# 4. Overall accuracy rate
# 5. Mean RT on correct trials
# 6. SD of RT on correct trials
# 7. Error rate on small-number trials (primes/targets in 1-2-3 range)
# 8. Error rate on large-number trials (primes/targets in 4-5-6 range)
```

**Data Source:**
- Use the same EEG epochs from the original analysis
- Extract left temporal amplitude from Cluster 2 channels: E32, E33, E34, E38, E39, E43, E44, E48, E49, E56, E57, E127, E128
- Time window: 148-364 ms
- Behavioral data comes from epoch metadata (Target.ACC, Target.RT, Prime/Target numerosity)

#### Step 2: Compute Correlations
Test these specific hypotheses across 24 subjects:

```python
# Hypothesis 1: Stronger error-related left temporal negativity → Lower accuracy
correlation_1 = scipy.stats.pearsonr(left_temp_diff, accuracy_rate)
# Prediction: Negative correlation (more negative diff = lower accuracy)

# Hypothesis 2: Stronger error-related left temporal negativity → More errors on large numbers
correlation_2 = scipy.stats.pearsonr(left_temp_diff, error_rate_diff)
# Prediction: Negative correlation (more negative diff = bigger large-vs-small error gap)

# Hypothesis 3: Stronger error-related left temporal negativity → Slower correct RTs
correlation_3 = scipy.stats.pearsonr(left_temp_diff, mean_RT_correct)
# Prediction: Negative correlation (more negative diff = longer RTs)

# Hypothesis 4: Left temporal in errors predicts RT variability
correlation_4 = scipy.stats.pearsonr(mean_left_temp_Acc0, sd_RT_correct)
# Prediction: Negative correlation (more negative = more variable strategy use)
```

**Note on direction:** `left_temp_diff = Acc0 - Acc1`. Since errors are more negative:
- Negative diff values = enhanced negativity in errors (as expected)
- More negative diff = stronger verbal engagement in errors
- Correlations should be negative if hypothesis is correct

#### Step 3: Visualizations
Create these plots:

1. **Scatter plot matrix** (4x4 grid):
   - X-axis options: `left_temp_diff`, `mean_left_temp_Acc0`
   - Y-axis options: `accuracy_rate`, `error_rate_diff`, `mean_RT_correct`, `sd_RT_correct`
   - Add regression lines with 95% CI
   - Annotate with r and p values

2. **Individual profiles heatmap**:
   - Rows: 24 subjects (sorted by `left_temp_diff`)
   - Columns: All summary variables (z-scored for visualization)
   - Color scale: Blue (low) to Red (high)
   - Highlights subjects who show strong verbal-strategy profile

3. **Categorical comparison**:
   - Split subjects into "High verbal" (most negative 8 subjects) vs "Low verbal" (least negative 8)
   - Compare behavioral measures with independent t-tests
   - Bar plots with error bars

### Statistical Reporting
Report for each correlation:
- Pearson r with 95% CI
- p-value (two-tailed)
- Effect size interpretation (small: |r|<0.3, medium: 0.3-0.5, large: >0.5)
- Scatter plot with regression line

**Multiple comparisons correction:**
- Use Bonferroni correction (4 main tests → alpha = 0.05/4 = 0.0125)
- Report both uncorrected and corrected p-values

### Expected Output Files
```
sensor_space_analysis/outputs/between_subjects/
├── data/
│   ├── subject_summary.csv                    # Subject-level summary data
│   ├── correlations.csv                       # Correlation results
│   └── group_comparison.csv                   # High vs low verbal groups
├── figures/
│   ├── 01_correlation_matrix.png              # 4x4 scatter matrix
│   ├── 02_subject_profiles.png                # Heatmap of subject profiles
│   ├── 03_group_comparison.png                # High vs low verbal comparison
│   └── 04_accuracy_by_left_temp.png          # Key relationship visualization
└── reports/
    └── between_subjects_report.html           # Comprehensive HTML report
```

---

## Analysis #2: Trial-Type Analysis
**Question:** Can we distinguish "verbal success" (slow-correct) from "verbal failure" (errors) and "visual success" (fast-correct) based on left temporal activity?

### Rationale
The current analysis shows:
- Errors have enhanced left temporal negativity
- Within correct trials, more negativity → longer RT

This suggests three potential trial types:
1. **Fast visual processing** (low negativity, fast RT, correct)
2. **Verbal mediation** (high negativity, slow RT, correct)
3. **Failed verbal** (high negativity, RT=0, error)

If this model is correct, we should see that:
- **Slow-correct trials** resemble **errors** in left temporal amplitude
- **Fast-correct trials** show **minimal** left temporal engagement
- There's a **continuum** from fast-visual → slow-verbal → failed-verbal

### Data Structure
Create a **trial-type classification**:

```python
trial_classification = {
    'trial_id': 12345,
    'subject_id': '02',
    'accuracy': 1,              # 0=error, 1=correct
    'RT': 523.5,                # RT in ms (0 for errors)
    'left_temp_amp': -2.34,     # Left temporal amplitude (µV)

    # Classification based on RT percentiles WITHIN CORRECT TRIALS ONLY
    'trial_type': 'slow_correct',  # Options: 'error', 'fast_correct', 'medium_correct', 'slow_correct'
    'RT_percentile': 85.3,          # RT percentile within correct trials (NA for errors)
    'RT_category': 'P75+',          # 'P0-25', 'P25-50', 'P50-75', 'P75+'
}
```

### Analysis Steps

#### Step 1: Classify Trials
```python
# For each subject separately:
for subject in subjects:
    # 1. Extract all correct trials for this subject
    correct_trials = trials[trials.accuracy == 1]

    # 2. Compute RT percentiles (within this subject's correct trials)
    RT_25th = np.percentile(correct_trials.RT, 25)
    RT_50th = np.percentile(correct_trials.RT, 50)  # Median
    RT_75th = np.percentile(correct_trials.RT, 75)

    # 3. Classify correct trials into quartiles
    correct_trials['trial_type'] = np.select(
        [
            correct_trials.RT <= RT_25th,
            (correct_trials.RT > RT_25th) & (correct_trials.RT <= RT_50th),
            (correct_trials.RT > RT_50th) & (correct_trials.RT <= RT_75th),
            correct_trials.RT > RT_75th
        ],
        ['fast_correct', 'medium_fast_correct', 'medium_slow_correct', 'slow_correct']
    )

    # 4. Add error trials as separate category
    error_trials['trial_type'] = 'error'

    # 5. Combine all trials
    all_trials = pd.concat([correct_trials, error_trials])
```

**Why within-subject classification?** Subjects have different baseline RT distributions. We want to identify slow-vs-fast **relative to each person's typical speed**.

#### Step 2: Compare Left Temporal Amplitude Across Trial Types
```python
# For each trial type, compute:
# - Mean left temporal amplitude
# - SE of mean
# - 95% CI

# Primary comparison:
trial_type_comparison = {
    'error':              {'mean': -3.45, 'SE': 0.23, 'n_trials': 936},
    'slow_correct':       {'mean': -2.87, 'SE': 0.18, 'n_trials': 1062},  # P75+ RT
    'medium_slow_correct':{'mean': -2.21, 'SE': 0.17, 'n_trials': 1062},  # P50-75
    'medium_fast_correct':{'mean': -1.68, 'SE': 0.16, 'n_trials': 1062},  # P25-50
    'fast_correct':       {'mean': -1.32, 'SE': 0.17, 'n_trials': 1062},  # P0-25
}

# Statistical test: One-way ANOVA followed by post-hoc comparisons
# Key contrasts:
# 1. Error vs Slow-correct (are they similar?)
# 2. Slow-correct vs Fast-correct (continuum?)
# 3. Error vs Fast-correct (opposite ends?)
```

#### Step 3: Test "Verbal Route as Risky Strategy" Hypothesis
**Prediction:** If the verbal route is risky but sometimes effective:
- Among **high left temporal negativity trials**, some succeed (slow-correct) and some fail (error)
- Among **low left temporal negativity trials**, most succeed (fast-correct) with few failures

**Analysis:**
```python
# Define "high verbal engagement" as left_temp_amp < -2.0 µV (more negative)
# Define "low verbal engagement" as left_temp_amp > -1.0 µV

high_verbal_trials = trials[trials.left_temp_amp < -2.0]
low_verbal_trials = trials[trials.left_temp_amp > -1.0]

# Compute success rates
high_verbal_success_rate = high_verbal_trials.accuracy.mean()
low_verbal_success_rate = low_verbal_trials.accuracy.mean()

# Test: Is low-verbal more reliable?
# Prediction: low_verbal_success_rate > high_verbal_success_rate
```

#### Step 4: Mixture Modeling (Optional Advanced)
Use **Gaussian Mixture Models** to find natural clusters in (left_temp_amp, RT) space:

```python
from sklearn.mixture import GaussianMixture

# Prepare data (correct trials only, since errors have RT=0)
X = correct_trials[['left_temp_amp', 'RT']].values

# Fit 2-3 component mixture model
gmm = GaussianMixture(n_components=3, random_state=42)
correct_trials['cluster'] = gmm.fit_predict(X)

# Interpret clusters:
# - Cluster 0: Fast visual (low amp, fast RT)?
# - Cluster 1: Verbal mediation (high amp, slow RT)?
# - Cluster 2: Mixed/intermediate?

# Then compare errors to each cluster
```

### Visualizations

1. **Left temporal amplitude by trial type** (violin plot):
   - X-axis: Trial type (Error, Slow-correct, Medium-slow, Medium-fast, Fast-correct)
   - Y-axis: Left temporal amplitude (µV)
   - Show distributions + means with SE bars
   - Annotate with post-hoc comparison results

2. **2D scatter plot** (left temporal vs RT):
   - X-axis: Left temporal amplitude
   - Y-axis: RT (jittered at 0 for errors)
   - Color by trial type
   - Add density contours for correct trials
   - Show error trials as distinct cluster at RT=0

3. **Success rate by left temporal amplitude** (binned):
   - Bin left temporal amplitude into deciles
   - Compute proportion correct in each bin
   - Plot proportion correct vs amplitude (with 95% CI)
   - Shows: Is there an amplitude range where verbal route succeeds?

4. **Individual subject patterns** (small multiples):
   - 24 panels (one per subject)
   - Each panel: left temporal vs RT scatter
   - Color by accuracy
   - Shows: Do all subjects show this pattern, or just some?

### Statistical Reporting

1. **One-way ANOVA:** Trial type (5 levels) predicting left temporal amplitude
   - Report F, df, p, η²

2. **Post-hoc comparisons** (with Bonferroni correction):
   - Error vs Slow-correct
   - Error vs Fast-correct
   - Slow-correct vs Fast-correct
   - Report mean difference, 95% CI, p-value

3. **Success rate comparison:**
   - High verbal vs Low verbal engagement
   - Chi-square test or logistic regression
   - Report odds ratio with 95% CI

### Expected Output Files
```
sensor_space_analysis/outputs/trial_types/
├── data/
│   ├── trial_classification.csv               # All trials with type labels
│   ├── trial_type_summary.csv                 # Mean amplitude by type
│   ├── success_rate_by_amplitude.csv          # Binned success rates
│   └── anova_results.csv                      # Statistical test results
├── figures/
│   ├── 01_amplitude_by_trial_type.png         # Violin plot
│   ├── 02_amplitude_vs_RT_scatter.png         # 2D scatter with trial types
│   ├── 03_success_rate_by_amplitude.png       # Binned success curve
│   ├── 04_subject_patterns.png                # Small multiples (24 subjects)
│   └── 05_mixture_model.png                   # GMM clusters (optional)
└── reports/
    └── trial_types_report.html                # Comprehensive HTML report
```

---

## Implementation Notes

### Code Organization
```
sensor_space_analysis/
├── src/
│   ├── between_subjects.py       # NEW: Analysis #1
│   ├── trial_types.py            # NEW: Analysis #2
│   ├── data_extraction.py        # Reuse existing
│   ├── preprocessing.py          # Reuse existing
│   └── visualization.py          # Extend with new plots
├── run_between_subjects.py       # NEW: Entry point for Analysis #1
├── run_trial_types.py            # NEW: Entry point for Analysis #2
└── outputs/
    ├── between_subjects/         # NEW
    └── trial_types/              # NEW
```

### Reusable Code
You can reuse from existing analysis:
- `data_extraction.aggregate_all_subjects()` - Get trial-level data
- Subject loop structure
- Left temporal electrode definitions (Cluster 2: E32, E33, E34, E38, E39, E43, E44, E48, E49, E56, E57, E127, E128)
- Time window: 148-364 ms

### What's Different
- **Analysis #1:** Subject-level aggregation (not trial-level)
- **Analysis #2:** Trial-type classification based on RT quartiles within-subject

### Data Validation Checks
1. Verify all subjects have both Acc0 and Acc1 trials
2. Check that RT=0 only occurs in Acc0 trials
3. Ensure left temporal amplitude has reasonable range (-10 to +10 µV)
4. Confirm trial counts match expected (after filtering cardinality trials)

---

## Deliverables

### For Analysis #1 (Between-Subject):
1. ✅ Python script: `run_between_subjects.py`
2. ✅ Subject-level summary CSV
3. ✅ Correlation results CSV
4. ✅ All figures (4 figures minimum)
5. ✅ HTML report with interpretation

### For Analysis #2 (Trial-Type):
1. ✅ Python script: `run_trial_types.py`
2. ✅ Trial classification CSV
3. ✅ Statistical test results CSV
4. ✅ All figures (5 figures minimum)
5. ✅ HTML report with interpretation

### Testing
For each analysis:
- Run on all 24 subjects
- Verify outputs are generated
- Check that figures load correctly in HTML report
- Confirm statistical results are reasonable (effect sizes, p-values)

---

## Questions to Address in Reports

### Analysis #1 Report Should Answer:
1. Do subjects with stronger error-related left temporal negativity have lower accuracy?
2. Do they show worse performance specifically on large numbers (4-5-6)?
3. Do they have longer RTs on correct trials (consistent verbal strategy use)?
4. Can we identify "high verbal" vs "low verbal" subject groups?

### Analysis #2 Report Should Answer:
1. Do slow-correct trials show similar left temporal negativity to errors?
2. Is there a continuum from fast-correct → slow-correct → error in terms of left temporal engagement?
3. Is the verbal route "risky" (sometimes succeeds, sometimes fails)?
4. Do all subjects show this pattern, or only some?

---

## Timeline Suggestion
- **Analysis #1:** ~4-6 hours (simpler: subject-level correlations)
- **Analysis #2:** ~6-8 hours (more complex: trial classification + mixture modeling)

**Start with Analysis #1** - it's more straightforward and will give immediate insights into individual differences. Then proceed to Analysis #2 for the deeper mechanistic understanding.

---

## Notes
- Use the same Cluster 2 electrode definitions from original analysis
- Maintain same preprocessing (filtering, baseline, etc.)
- Generate HTML reports in same style as original analysis
- Include comprehensive interpretation, not just statistics

**Good luck! These analyses will provide crucial insights into how verbal strategies contribute to both success and failure in numerical cognition.** 🧠📊
