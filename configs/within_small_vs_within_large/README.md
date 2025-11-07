# Within-Small vs Within-Large Analysis

## Quick Start

### Run the Analysis

```powershell
# Activate environment
conda activate numbers_eeg_source

# Run sensor analysis only
python -m code.run_sensor_analysis_pipeline --config configs\within_small_vs_within_large\sensor_within_small_vs_within_large.yaml --accuracy all

# Or run full pipeline (sensor + source, if source configs exist)
python -m code.run_full_analysis_pipeline --config configs\within_small_vs_within_large\sensor_within_small_vs_within_large.yaml --accuracy all
```

### Check Trial Counts (Optional)

Before running, verify you have sufficient trials:

```powershell
python temp\check_within_small_large_trials.py
```

---

## What This Analysis Tests

**Research Question**: Do numerical changes within the subitizing range (1-2-3) produce different brain responses than changes within the estimation range (4-5-6)?

**Hypothesis**: If Cluster 4 (right parietal, 252-364 ms) reflects ANS-based magnitude estimation, within-large transitions should show stronger activation than within-small transitions.

---

## Conditions

### Condition A: Within-Small (1-2-3)
**All changes starting AND ending in 1-2-3 range**:
- 1→2, 2→1
- 1→3, 3→1
- 2→3, 3→2

**Total**: 6 unique transition types (balanced for direction)

### Condition B: Within-Large (4-5-6)
**All changes starting AND ending in 4-5-6 range**:
- 4→5, 5→4
- 4→6, 6→4
- 5→6, 6→5

**Total**: 6 unique transition types (balanced for direction)

**Note**: Cardinality (no-change) trials like 11, 22, 33, 44, 55, 66 are **excluded** from both conditions.

---

## Key Predictions

### 1. Right Parietal (Cluster 4 Region)
**Prediction**: Within-large > Within-small

**Time window**: 252-364 ms
**Channels**: E55, E62, E77-E80, E85-E87, E90-E93, E95-E98, E100-E101, E103, E109-E110

**Rationale**: Right parietal cortex implements the ANS, which is required for large but not small numerosity discrimination.

### 2. Bilateral Posterior (P1/N1 Regions)
**Prediction**: May show within-small > within-large if precise individuation recruits object-tracking systems more strongly.

**Time window**: 100-200 ms
**Channels**: Posterior occipital-parietal electrodes

### 3. P3b (Decision/Confidence)
**Prediction**: Within-small > within-large if PI allows more confident decisions.

**Time window**: 368-496 ms
**Channels**: Bilateral posterior-parietal, midline central

---

## Outputs

### Location
```
derivatives/sensor/<timestamp>-sensor_within_small_vs_within_large/
```

### Files
- `sensor_within_small_vs_within_large_report.txt` - Statistical results
- `sensor_within_small_vs_within_large_grand_average-ave.fif` - Grand average contrast
- `sensor_within_small_vs_within_large_erp_cluster_*.png` - ERP plots for each significant cluster
- `sensor_within_small_vs_within_large_topomap_cluster_*.png` - Topomaps for each cluster
- `sensor_within_small_vs_within_large_roi_*.png` - Condition ERPs over canonical ROIs (P1, N1, P3b)

---

## Interpreting Results

### If Cluster 4 Shows Within-Large > Within-Small:
✅ **Supports** Cluster 4 as ANS proxy
✅ **Confirms** dual-system hypothesis
✅ Right parietal recruits more strongly for magnitude estimation

### If No Cluster 4 Difference:
⚠️ **Challenges** ANS interpretation of Cluster 4
⚠️ May indicate Cluster 4 reflects general difficulty, not magnitude-specific processing

### If Multiple Clusters Emerge:
🔍 **Suggests** different processing streams for small vs large
🔍 Map clusters to theoretical predictions:
- Early posterior: Perceptual encoding differences
- Mid-latency parietal: Magnitude vs object-file systems
- Late central: Decision confidence differences

---

## Follow-Up Analyses

### 1. ROI-Restricted Analysis
Uncomment the ROI section in the YAML to restrict testing to Cluster 4 electrodes only:

```yaml
stats:
  roi:
    channels:
      - E55
      - E62
      # ... (full list in YAML)
```

This increases sensitivity for testing the specific Cluster 4 hypothesis.

### 2. Distance Effects Within Each Range
Test whether numerical distance modulates the effect:
- Within-small: Compare distance-1 (12, 23) vs distance-2 (13)
- Within-large: Compare distance-1 (45, 56) vs distance-2 (46)

### 3. Integration with Brain-Behavior Analysis
Extract Cluster 4 amplitudes for each subject and condition, then correlate with:
- RT patterns
- Accuracy rates
- ANS acuity (Weber fraction)

---

## Related Analyses

### Complementary
- **Acc0 vs Acc1**: Tests error processing (orthogonal to this)
- **13_31**: Tests direction within small range only
- **Change vs No-Change**: Tests detection vs baseline

### Sequential
1. Run this analysis first → Identify magnitude-range effects
2. Run brain-behavior correlation → Link clusters to RT/accuracy
3. Run source localization → Identify cortical generators

---

## Scientific Background

See [ANALYSIS_RATIONALE.md](ANALYSIS_RATIONALE.md) for:
- Detailed theoretical background
- Dual-system hypothesis overview
- Comprehensive predictions
- Integration with existing literature

---

## Questions?

This analysis was designed to test Cluster 4 from the Acc0 vs Acc1 results. If you're unsure about:
- The theoretical motivation → See ANALYSIS_RATIONALE.md
- How to run it → See Quick Start above
- How to interpret results → See Interpreting Results above

**Expected runtime**: ~10 minutes for sensor analysis only
