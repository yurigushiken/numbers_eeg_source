# Within-Small vs Within-Large Analysis: Scientific Rationale

## Overview

This analysis directly tests the **dual-system hypothesis** of numerical cognition by contrasting brain responses to numerical changes within the subitizing range (1-2-3) versus the estimation range (4-5-6).

---

## Theoretical Background

### Two Systems of Numerical Processing

**1. Parallel Individuation (PI) / Object File System**
- Operates on **small numerosities (1-3)**
- Precise, capacity-limited enumeration
- Does not rely on magnitude estimation
- Linked to visual short-term memory and object tracking

**2. Approximate Number System (ANS)**
- Operates on **larger numerosities (4+)**
- Approximate, ratio-dependent magnitude estimation
- Weber's law applies (discrimination depends on ratio, not absolute difference)
- Linked to parietal magnitude representation

---

## Hypothesis: Cluster 4 as ANS Proxy

From the **Acc0 vs Acc1** analysis, we discovered **Cluster 4**:

```
Right Posterior-Parietal Region
- Time window: 252-364 ms (mid-latency)
- 22 electrodes: E55, E62, E77-E80, E85-E87, E90-E93, E95-E98, E100-E101, E103, E109-E110
- Effect: Errors show MORE POSITIVE activation (+0.91 µV, Cohen's d = 0.99)
```

**Key Observation**: Right parietal cortex is classically associated with the **ANS** and **magnitude processing**.

### Primary Hypothesis

**If Cluster 4 reflects ANS-based magnitude estimation**, then:

✅ **Within-large transitions (4-5-6)** should show **stronger Cluster 4 activation** than within-small transitions (1-2-3)

This is because:
- Large numerosities (4-6) **require** ANS-based estimation
- Small numerosities (1-3) can use precise PI without ANS engagement
- Cluster 4 activity should scale with **reliance on magnitude estimation**

---

## Contrast Design

### Condition A: Within-Small Changes (1-2-3 range)
**Transitions**: 12, 21, 13, 31, 23, 32

**Characteristics**:
- All within subitizing range
- Can be discriminated via **precise object individuation**
- Minimal reliance on magnitude estimation
- Expected to recruit bilateral posterior regions (object tracking)

### Condition B: Within-Large Changes (4-5-6 range)
**Transitions**: 45, 54, 46, 64, 56, 65

**Characteristics**:
- All beyond subitizing capacity
- **Require** approximate magnitude estimation
- Should strongly engage ANS (right parietal)
- Weber's law applies (e.g., 4→6 easier than 5→6)

### The Contrast: Small - Large

**Predicted Pattern**:
- **Cluster 4 (right parietal, 252-364 ms)**: Large > Small (stronger for ANS-dependent numerosities)
- **Left temporal (Cluster 2, 148-364 ms)**: May show opposite pattern if verbal counting is easier for small numbers
- **Bilateral posterior (Cluster 1, P3b)**: May show Small > Large if PI provides more confident decisions

---

## Key Predictions

### Prediction 1: Cluster 4 Activation
**Within-large** transitions should show **more positive** Cluster 4 amplitude than within-small transitions.

**Reasoning**:
- Large numerosities force reliance on right parietal magnitude representation
- Small numerosities can bypass this system

### Prediction 2: Topographic Pattern
The spatial distribution should show:
- **Right parietal focus** for within-large (ANS signature)
- **Bilateral posterior** for within-small (PI signature)

### Prediction 3: Temporal Dynamics
- **Early (100-200 ms, N1)**: Both conditions should show similar sensory processing
- **Mid-latency (250-364 ms, Cluster 4 window)**: Divergence begins, large > small in right parietal
- **Late (368-496 ms, P3b)**: Decision/confidence differences may emerge

### Prediction 4: Individual Differences
Subjects with stronger Cluster 4 responses to large numerosities should show:
- Poorer accuracy on large-number discrimination
- Larger Weber fractions (poorer ANS acuity)
- Greater reliance on approximation strategies

---

## Comparison to Existing Contrasts

### vs. Acc0 vs Acc1 Analysis
- **Acc0 vs Acc1**: Error processing (all numerosities mixed)
- **Within-Small vs Large**: Numerosity range effect (orthogonal to accuracy)
- **Complementarity**: Tests whether Cluster 4 is **error-specific** or **numerosity-specific**

### vs. 13_31 Direction Analysis
- **13_31**: Direction effect within small range only
- **Small vs Large**: Magnitude range effect across directions
- **Complementarity**: Separates direction from magnitude processing

---

## Expected Outcomes

### Outcome A: Strong Right Parietal Effect (Supports ANS Hypothesis)
If within-large shows significantly stronger Cluster 4 activation:
- **Confirms** Cluster 4 as ANS proxy
- **Supports** dual-system model
- **Suggests** right parietal recruitment scales with need for magnitude estimation

### Outcome B: No Cluster 4 Difference (Challenges ANS Hypothesis)
If no difference in Cluster 4:
- **Suggests** Cluster 4 is not primarily magnitude-driven
- May reflect general task difficulty or attention
- Would need alternative explanation for Cluster 4 role

### Outcome C: Bilateral Posterior Differences
If effects appear in **both hemispheres**:
- **Suggests** different processing streams for small vs large
- Left hemisphere may dominate for small (verbal/precise)
- Right hemisphere may dominate for large (spatial/approximate)

---

## Statistical Approach

### Primary Analysis
- **Cluster-based permutation test** (whole scalp, 0-496 ms)
- **5000 permutations**, cluster alpha = 0.05
- Identifies **any** spatiotemporal clusters showing small vs large differences

### Targeted ROI Analysis (Optional)
Restrict to **Cluster 4 electrodes** (252-364 ms window):
- Increases sensitivity for testing specific Cluster 4 hypothesis
- Can be run as follow-up or a priori test

### Complementary Analyses
1. **Trial-level modeling**: Does Cluster 4 amplitude predict RT/accuracy differently for small vs large?
2. **Distance effects**: Within each range, do larger distances (e.g., 13 vs 12) modulate Cluster 4?
3. **Weber's law test**: For large numbers only, does Cluster 4 scale with numerical ratio?

---

## Integration with Brain-Behavior Analysis

This analysis **complements** the planned Cluster 2 brain-behavior correlation:

### Sequential Analysis Plan

**Phase 1**: Run Within-Small vs Within-Large sensor analysis
- Identify clusters differentiating magnitude ranges

**Phase 2**: Extract Cluster 4 amplitudes by numerosity range
- Add to brain-behavior dataset
- Test: `RT ~ Cluster2 * Cluster4 * NumerosityRange * Accuracy`

**Phase 3**: Interpret interaction patterns
- Does left temporal (Cluster 2) predict RT more for small numbers?
- Does right parietal (Cluster 4) predict RT more for large numbers?
- Are these systems **complementary** or **competitive**?

---

## Significance for Numerical Cognition

This analysis addresses a **fundamental question**:

**Are small and large numerosities processed by distinct neural systems?**

### If YES (predicted):
- Provides **neural evidence** for dual-system theories
- Localizes PI to bilateral posterior (object-tracking) systems
- Localizes ANS to right parietal (magnitude) systems
- Explains why subitizing is fast/accurate (doesn't rely on noisy ANS)

### If NO (alternative):
- Suggests a **unified** magnitude system across all numerosities
- Challenges traditional PI vs ANS distinction
- Would require rethinking why 1-3 are special

---

## Next Steps After Sensor Analysis

If significant clusters emerge:

1. **Source localization**: Which cortical regions drive the effects?
   - Expected: Right IPS for large, bilateral posterior for small

2. **Time-frequency analysis**: Are different oscillatory signatures involved?
   - PI might show theta (object tracking)
   - ANS might show alpha/beta (magnitude estimation)

3. **Connectivity analysis**: Do small and large recruit different networks?

4. **Individual differences**: Link to behavioral ANS acuity measures

---

## Summary

This **Within-Small vs Within-Large** analysis is a **critical test** of the dual-system hypothesis. By contrasting transitions entirely within the subitizing range vs entirely within the estimation range, we can:

✅ Test whether Cluster 4 is an ANS proxy
✅ Identify distinct neural signatures of PI vs ANS
✅ Link brain systems to behavioral performance
✅ Advance theoretical understanding of numerical cognition

**Expected runtime**: ~10 minutes (sensor analysis only)
**Power**: High (balanced conditions, strong theoretical predictions)
**Novelty**: Direct neural test of PI vs ANS with EEG

---

## References

Hyde, D. C., & Spelke, E. S. (2009). All numbers are not equal: An electrophysiological investigation of small and large number representations. *Journal of Cognitive Neuroscience, 21*(6), 1039-1053.

Hyde, D. C., & Spelke, E. S. (2012). Spatiotemporal dynamics of processing nonsymbolic number: An event-related potential source localization study. *Human Brain Mapping, 33*(9), 2189-2203.

Piazza, M., Izard, V., Pinel, P., Le Bihan, D., & Dehaene, S. (2004). Tuning curves for approximate numerosity in the human intraparietal sulcus. *Neuron, 44*(3), 547-555.
