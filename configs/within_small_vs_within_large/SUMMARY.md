# Within-Small vs Within-Large: Executive Summary

## What We Created

✅ **New condition sets** in `code/condition_sets.yaml`:
- `WITHIN_SMALL_CHANGES`: All transitions within 1-2-3 range (12, 21, 13, 31, 23, 32)
- `WITHIN_LARGE_CHANGES`: All transitions within 4-5-6 range (45, 54, 46, 64, 56, 65)

✅ **New sensor analysis config**: `configs/within_small_vs_within_large/sensor_within_small_vs_within_large.yaml`

✅ **Supporting documentation**:
- `ANALYSIS_RATIONALE.md` - Full scientific background and predictions
- `README.md` - Quick start guide and interpretation help
- `temp/check_within_small_large_trials.py` - Trial count verification script

---

## Why This Analysis Matters

From your **Acc0 vs Acc1** analysis, you discovered **Cluster 4**:
- Right posterior-parietal region
- 252-364 ms time window
- Errors show **more positive** activation (+0.91 µV)
- Large effect size (Cohen's d = 0.99)

**Key Question**: What does Cluster 4 represent?

### Hypothesis: ANS Proxy

Right parietal cortex is the classic location for the **Approximate Number System (ANS)**, which processes larger numerosities (4+) via magnitude estimation.

**If Cluster 4 reflects ANS**, then:
- Large numerosities (4-6) should activate it **more** than small numerosities (1-3)
- This is because large numbers **require** approximate estimation
- Small numbers can use precise **Parallel Individuation** (PI) instead

---

## The Test

**Contrast**: Within-Small (1-2-3 changes) vs Within-Large (4-5-6 changes)

**Design**:
- Both conditions have **6 transition types** (balanced)
- Both **exclude cardinality** (no-change) trials
- Both balanced for **direction** (3 increasing, 3 decreasing)

**Critical Difference**: Numerosity range (subitizing vs estimation)

---

## Expected Results

### Scenario 1: Cluster 4 Shows Large > Small ✅
**Interpretation**: Cluster 4 is an ANS proxy!
- Confirms dual-system hypothesis
- Right parietal recruitment scales with need for magnitude estimation
- Small numbers bypass this system via PI

**Implications**:
- Strong support for PI vs ANS distinction
- Neural evidence for classical behavioral theories
- Cluster 4 can be used as ANS marker in future analyses

### Scenario 2: No Cluster 4 Difference ⚠️
**Interpretation**: Cluster 4 is NOT primarily magnitude-driven
- May reflect general task difficulty
- May be error-specific rather than numerosity-specific
- Need alternative explanation

**Implications**:
- Challenges simple ANS interpretation
- Suggests more complex role for right parietal
- May need to rethink Cluster 4 from Acc0 vs Acc1

### Scenario 3: Multiple Clusters with Different Patterns 🔍
**Interpretation**: Dissociable systems for small vs large
- Early clusters: Perceptual differences
- Mid-latency clusters: PI vs ANS engagement
- Late clusters: Decision confidence differences

**Implications**:
- Rich spatiotemporal signature of dual systems
- Can map timing of system transitions
- Individual clusters may predict different behaviors

---

## Connection to Brain-Behavior Analysis

This analysis **sets up** the planned Cluster 2 brain-behavior work:

### Integrated Model
```
Trial Properties:
  - Numerosity Range (Small/Large) → Cluster 4 activation
  - Accuracy (Correct/Error) → Clusters 2 & 4 activation
  - Direction (Inc/Dec) → Multiple clusters

Brain Measures:
  - Cluster 2 (Left temporal, 148-364 ms) → Verbal/semantic processing
  - Cluster 4 (Right parietal, 252-364 ms) → Magnitude estimation

Behavior:
  - Reaction Time
  - Accuracy
```

**Ultimate Question**: How do these brain systems **interact** to produce behavior?

---

## Next Steps

### Immediate (Do This Now!)

1. **Run the analysis**:
   ```powershell
   python -m code.run_sensor_analysis_pipeline --config configs\within_small_vs_within_large\sensor_within_small_vs_within_large.yaml --accuracy all
   ```

2. **Examine results**:
   - Check for Cluster 4 region effects (right parietal, 252-364 ms)
   - Look at topographic distribution
   - Compare effect sizes to Acc0 vs Acc1

3. **Interpret findings**:
   - Does it support ANS hypothesis?
   - Are there unexpected clusters?
   - How does it relate to Cluster 2 (left temporal)?

### Follow-Up (After Initial Results)

1. **Extract Cluster 4 data for brain-behavior**:
   - Add "numerosity range" as predictor
   - Test: Does Cluster 4 predict RT more for large numbers?

2. **Run targeted ROI analysis** (if whole-brain is noisy):
   - Restrict to Cluster 4 electrodes
   - Increases sensitivity for specific hypothesis

3. **Explore distance effects**:
   - Within-small: Does 13 differ from 12?
   - Within-large: Does 46 differ from 45?
   - Tests Weber's law predictions

---

## Scientific Impact

This analysis addresses **fundamental questions**:

1. **Are there distinct neural systems for small vs large numbers?**
   - Classical question in numerical cognition
   - Direct EEG test with high temporal resolution

2. **What is the role of right parietal cortex?**
   - ANS implementation (predicted)
   - General magnitude processing
   - Task difficulty/attention (alternative)

3. **When do PI and ANS engage during processing?**
   - The 252-364 ms window is **mid-latency**
   - After perception, before decision
   - Critical transition point

---

## Timeline

- **Analysis runtime**: ~10 minutes (sensor only)
- **Interpretation**: ~30 minutes (examine clusters, compare to predictions)
- **Integration with Cluster 2 work**: After brain-behavior analysis completes
- **Manuscript implications**: Major finding if Cluster 4 shows predicted effect!

---

## Bottom Line

You already have an exciting finding with Cluster 4 from the **Acc0 vs Acc1** analysis. Now we're testing **what it means**:

- **If it's ANS**: Huge implications for dual-system theories
- **If it's not**: Still important - tells us what Cluster 4 actually does
- **Either way**: Advances understanding of numerical cognition

**This is a high-priority, theoretically-motivated analysis that builds directly on your existing discoveries!**

Ready to run it? The config is all set up. Just execute the command and see what emerges!

```powershell
conda activate numbers_eeg_source
python -m code.run_sensor_analysis_pipeline --config configs\within_small_vs_within_large\sensor_within_small_vs_within_large.yaml --accuracy all
```

🚀 **Let's see if Cluster 4 is the ANS!** 🚀
