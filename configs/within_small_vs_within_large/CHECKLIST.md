# Within-Small vs Within-Large: Pre-Flight Checklist

## ✅ Setup Complete

- [x] Created new config directory: `configs/within_small_vs_within_large/`
- [x] Added condition sets to `code/condition_sets.yaml`:
  - `WITHIN_SMALL_CHANGES` (1-2-3 range)
  - `WITHIN_LARGE_CHANGES` (4-5-6 range)
- [x] Created sensor analysis config: `sensor_within_small_vs_within_large.yaml`
- [x] Created documentation:
  - `ANALYSIS_RATIONALE.md` (scientific background)
  - `README.md` (how to run)
  - `SUMMARY.md` (executive summary)
  - `CHECKLIST.md` (this file)
- [x] Created trial count verification script: `temp/check_within_small_large_trials.py`

---

## 🎯 Ready to Run

### Option 1: Quick Check Trial Counts First (Recommended)

```powershell
conda activate numbers_eeg_source
python temp\check_within_small_large_trials.py
```

**Expected output**: ~50-100 trials per subject per condition

---

### Option 2: Run Full Analysis

```powershell
conda activate numbers_eeg_source
python -m code.run_sensor_analysis_pipeline --config configs\within_small_vs_within_large\sensor_within_small_vs_within_large.yaml --accuracy all
```

**Runtime**: ~10 minutes
**Output location**: `derivatives/sensor/<timestamp>-sensor_within_small_vs_within_large/`

---

## 📊 What to Look For in Results

### Primary Interest: Cluster 4 Region

**Location**: Right posterior-parietal
**Electrodes**: E55, E62, E77-E80, E85-E87, E90-E93, E95-E98, E100-E101, E103, E109-E110
**Time**: 252-364 ms

**Predicted Effect**: Within-Large > Within-Small (positive difference)

**Check**:
1. Is there a significant cluster in this region/time?
2. What's the direction of the effect?
3. What's the effect size (Cohen's d)?
4. Compare to Acc0 vs Acc1 Cluster 4 (d=0.99)

---

### Secondary Interests

**Left Temporal (Cluster 2 Region)**:
- May show opposite pattern if verbal counting easier for small numbers
- Electrodes: E32, E33, E34, E38, E39, E43, E44, E48, E49, E56, E57, E127, E128
- Time: 148-364 ms

**Bilateral Posterior (P1/N1)**:
- May show early perceptual differences
- Time: 100-200 ms

**P3b (Cluster 1 Region)**:
- May show decision confidence differences
- Time: 368-496 ms

---

## 🔍 Interpretation Guide

### Result A: Significant Right Parietal (Cluster 4) Effect

**If Within-Large > Within-Small** ✅:
- **Conclusion**: Cluster 4 is an ANS proxy!
- **Implication**: Large numbers recruit right parietal magnitude system more than small
- **Next step**: Extract Cluster 4 amps for brain-behavior analysis

**If Within-Small > Within-Large** ⚠️:
- **Conclusion**: Cluster 4 may reflect precise processing, not estimation
- **Implication**: Challenges ANS interpretation
- **Next step**: Rethink what Cluster 4 represents

---

### Result B: No Cluster 4 Effect

**If no significant cluster in right parietal**:
- **Conclusion**: Cluster 4 from Acc0 vs Acc1 is error-specific, not magnitude-specific
- **Implication**: Right parietal activation relates to performance monitoring, not ANS
- **Next step**: Focus on other clusters; test alternative hypotheses

---

### Result C: Multiple Significant Clusters

**If effects across multiple regions**:
- **Conclusion**: Rich spatiotemporal dissociation between PI and ANS
- **Implication**: Can map different stages of processing
- **Next step**: Detailed time-course analysis; source localization

---

## 📈 Success Metrics

### Minimum Success:
- Analysis runs without errors
- At least one significant cluster emerges
- Results are interpretable (support OR refute hypothesis)

### Ideal Success:
- Cluster 4 shows predicted effect (Large > Small)
- Effect size similar to Acc0 vs Acc1 (d ~ 1.0)
- Clean topography (right parietal focus)
- Timing consistent with ANS engagement (250-350 ms)

### Exceptional Success:
- Multiple dissociable clusters mapping onto PI vs ANS systems
- Left hemisphere (Cluster 2) shows opposite pattern
- Can link to behavioral predictions (Weber's law, distance effects)

---

## 🚨 Potential Issues & Solutions

### Issue 1: Insufficient Trials
**Symptom**: Warning about low trial counts
**Check**: Run `temp\check_within_small_large_trials.py`
**Solution**: Should still be OK - large/small transitions are common

### Issue 2: No Significant Clusters
**Symptom**: Report says "No significant clusters found"
**Possible causes**:
- Effect is real but small (underpowered)
- Hypothesis is wrong (no magnitude-range effect)
- Statistical threshold too strict

**Solutions**:
- Try lowering p_threshold (e.g., 0.05 instead of 0.01)
- Run ROI-restricted analysis (Cluster 4 only)
- Check effect sizes even if not significant

### Issue 3: Confusing Pattern of Results
**Symptom**: Clusters don't match predictions
**Approach**:
- Don't force interpretation!
- Document what you see
- Consider alternative explanations
- May reveal something new about the data

---

## 📝 After Running: Documentation

### Record in Lab Notes:
1. Date and time of analysis
2. Any warnings or errors
3. Number of significant clusters found
4. Peak locations and times
5. Effect sizes and p-values
6. Initial interpretation

### Share Results:
1. Screenshot of report text file
2. Cluster topomaps (especially if Cluster 4 appears)
3. ERP plots
4. Brief interpretation (2-3 sentences)

### Next Steps Decision:
- [ ] Results support ANS hypothesis → Proceed with brain-behavior integration
- [ ] Results challenge ANS hypothesis → Explore alternative interpretations
- [ ] Results ambiguous → Run targeted ROI analysis
- [ ] Results reveal unexpected finding → Design follow-up analysis

---

## 🎓 Learning Objectives

By running this analysis, you will:
1. Test a specific, theoretically-motivated hypothesis
2. Link sensor-space findings to cognitive theory
3. Evaluate alternative interpretations of existing clusters
4. Practice scientific reasoning (predictions → data → interpretation)

---

## ✨ Final Pre-Flight Check

Before running, confirm:
- [ ] Conda environment is activated
- [ ] You're in the project root directory (`D:\numbers_eeg_source`)
- [ ] You have time to wait ~10 minutes for completion
- [ ] You're ready to interpret results objectively (support OR refute hypothesis)

---

## 🚀 Ready? Let's Go!

```powershell
conda activate numbers_eeg_source
python -m code.run_sensor_analysis_pipeline --config configs\within_small_vs_within_large\sensor_within_small_vs_within_large.yaml --accuracy all
```

**Good luck! This is an exciting test of a fundamental hypothesis in numerical cognition!** 🧠🔢
