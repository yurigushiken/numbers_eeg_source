# Centralized Sensor ROI System

## Overview

This document describes the new centralized sensor ROI (Region of Interest) system implemented for the EEG analysis pipeline. The system provides a clean, maintainable way to define and use sensor channel groups for statistical analysis.

## Key Changes

### Before
- Each sensor config file had **~50 lines** of hardcoded channel lists
- ROI definitions were duplicated across all sensor YAML files
- No single source of truth for channel groups
- Difficult to maintain and error-prone

### After
- **Centralized definitions** in `configs/sensor_roi_definitions.yaml`
- Sensor configs reference ROI groups by name (e.g., `N1_bilateral`, `posterior_visual_parietal`)
- **~10 lines** in each config (80% reduction)
- Single source of truth for all sensor ROIs
- Easy to maintain and update

---

## File Structure

### Core Files

1. **`configs/sensor_roi_definitions.yaml`** (NEW)
   - Central repository for all sensor ROI definitions
   - Defines channel groups: `N1_bilateral`, `P1_Oz`, `P3b_midline`, `posterior_visual_parietal`
   - Supports nested groups (e.g., `posterior_visual_parietal` is union of N1 + P1 + P3b)

2. **`code/utils/cluster_stats.py`** (UPDATED)
   - Added `_load_sensor_roi_definitions()` function
   - Added `_resolve_sensor_roi_channels()` function
   - Updated `run_sensor_cluster_test()` to use centralized definitions
   - Maintains backward compatibility with inline `roi_channels` section

3. **All sensor config YAMLs** (UPDATED)
   - Removed ~50 lines of hardcoded channel lists
   - Added clean commented ROI examples
   - Now reference centralized definitions

---

## Usage

### How to Use Sensor ROIs in Analysis Configs

To restrict cluster-based permutation testing to specific sensor regions, uncomment and modify the `roi` section in your sensor config:

#### Option 1: Use Combined Posterior ROI (Recommended)
```yaml
stats:
  # ... other stats parameters ...
  roi:
    channel_groups:
      - posterior_visual_parietal  # All N1 + P1 + P3b channels
```

#### Option 2: Select Specific Individual ROIs
```yaml
stats:
  # ... other stats parameters ...
  roi:
    channel_groups:
      - N1_bilateral    # 14 channels over bilateral occipital regions
      - P1_Oz          # 8 channels around Oz (parieto-occipital)
      - P3b_midline    # 9 channels over centro-parietal midline
```

#### Option 3: Mix Groups and Direct Channels
```yaml
stats:
  # ... other stats parameters ...
  roi:
    channel_groups:
      - N1_bilateral
    channels:           # Add extra channels not in predefined groups
      - E100
      - E101
```

---

## Available ROI Definitions

### Individual ROIs

| ROI Name | Channels | Description |
|----------|----------|-------------|
| `N1_bilateral` | 14 | Bilateral N1 regions (posterior-occipital) |
| `P1_Oz` | 8 | P1 component (occipital midline around Oz) |
| `P3b_midline` | 9 | P3b component (centro-parietal midline) |

### Combined ROIs

| ROI Name | Components | Total Channels | Description |
|----------|------------|----------------|-------------|
| `posterior_visual_parietal` | N1 + P1 + P3b | 25 unique | Union of all posterior visual/parietal sensors |

**Note:** Total is less than sum (14+8+9=31) due to overlap between groups (e.g., E71, E77 appear in multiple regions).

---

## Channel Lists

### N1_bilateral (14 channels)
```
E59, E60, E65, E66, E67, E70, E71, E76, E77, E83, E84, E85, E90, E91
```

### P1_Oz (8 channels)
```
E70, E71, E74, E75, E76, E81, E82, E83
```

### P3b_midline (9 channels)
```
E54, E55, E61, E62, E67, E72, E77, E78, E79
```

### posterior_visual_parietal (25 unique channels)
```
E54, E55, E59, E60, E61, E62, E65, E66, E67, E70, E71, E72, E74, E75,
E76, E77, E78, E79, E81, E82, E83, E84, E85, E90, E91
```

---

## Scientific Rationale

### Channel Selection Basis

The sensor ROI definitions are based on:

1. **Functionally-defined regions** used for ERP visualization (see `code/utils/electrodes.py`)
2. **Topographic correspondence** to expected component distributions:
   - **N1**: Bilateral posterior-occipital regions (visual processing)
   - **P1**: Parieto-occipital midline (early visual response)
   - **P3b**: Centro-parietal midline (attentional/cognitive processing)

3. **Literature precedent**: Hyde & Spelke (2012) and related studies on numerosity processing

### When to Use Sensor ROIs

**Use sensor ROI restriction when:**
- You have strong *a priori* hypotheses about specific scalp regions
- You want to reduce multiple comparisons and increase statistical power
- Your source-level analysis identified specific regions of interest

**Run whole-scalp analysis when:**
- Conducting exploratory analyses
- Testing novel contrasts without strong spatial priors
- You want maximum sensitivity to unexpected effects

---

## Testing

A comprehensive test suite is provided to verify the system:

```bash
python test_sensor_roi_system.py
```

**Tests include:**
- Loading centralized ROI definitions
- Resolving individual ROI groups
- Resolving multiple ROI groups
- Nested ROI resolution (posterior_visual_parietal)
- Direct channel specification
- Mixed group + channel specification
- Config file format verification

All tests should pass ✓

---

## Comparison with Source Space ROIs

### Similarities
- Both use named ROI groups for clarity and maintainability
- Both support referencing ROI definitions by name
- Both are optional (can run full-space analysis)

### Differences

| Aspect | Source Space | Sensor Space |
|--------|-------------|--------------|
| **Basis** | Anatomical labels (FreeSurfer parcellation) | Sensor channel topography |
| **Definition** | `parc: aparc` + label names | Channel groups from central config |
| **Resolution** | Automatic vertex mapping from labels | Explicit channel lists |
| **Scientific grounding** | Direct neuroanatomy (e.g., `lateraloccipital`) | Functional/topographic correspondence |

### Configuration Format Comparison

**Source space ROI:**
```yaml
stats:
  roi:
    parc: aparc
    labels:
      - lateraloccipital
      - fusiform
      - inferiorparietal
```

**Sensor space ROI:**
```yaml
stats:
  roi:
    channel_groups:
      - N1_bilateral
      - P1_Oz
      - P3b_midline
```

Both systems now use clean, named references instead of hardcoded lists!

---

## Implementation Details

### Code Architecture

The ROI resolution system follows this flow:

1. **Config Loading** → Analysis config specifies `stats.roi.channel_groups`
2. **Definition Loading** → `_load_sensor_roi_definitions()` reads `sensor_roi_definitions.yaml`
3. **Channel Resolution** → `_resolve_sensor_roi_channels()` converts groups to channel sets
4. **Nested Resolution** → Handles `posterior_visual_parietal` → `[N1, P1, P3b]` → channels
5. **Deduplication** → Uses sets to handle overlapping channels
6. **Application** → Filters adjacency matrix and data array in `run_sensor_cluster_test()`

### Backward Compatibility

The system maintains **full backward compatibility**:
- If `roi_channels` section exists in config, it's used as fallback
- Old configs will continue to work without modification
- Allows gradual migration to centralized system

---

## For Scientific Reviewers

### Reproducibility
- All sensor ROI definitions are version-controlled in `configs/sensor_roi_definitions.yaml`
- Channel groups are identical to those used for ERP visualization
- Test suite (`test_sensor_roi_system.py`) verifies correct implementation

### Transparency
- Clear mapping from ROI name → specific channels
- No hidden or implicit channel selections
- All definitions documented with descriptions and channel counts

### Flexibility
- Supports hypothesis-driven ROI restriction
- Also supports whole-scalp exploratory analysis
- Method choice clearly documented in config files

---

## Maintenance

### Adding a New ROI

To add a new sensor ROI definition:

1. Edit `configs/sensor_roi_definitions.yaml`
2. Add new entry under `sensor_rois:`:
   ```yaml
   your_new_roi:
     description: "Clear description of the ROI"
     channels:
       - E1
       - E2
       - E3
   ```
3. Run `python test_sensor_roi_system.py` to verify
4. Update this README if needed

### Modifying Existing ROIs

1. Edit channel list in `configs/sensor_roi_definitions.yaml`
2. Document rationale in git commit message
3. Run test suite to verify no breaking changes
4. Consider scientific impact on existing analyses

---

## Summary

The centralized sensor ROI system provides:

✓ **Maintainability** - Single source of truth for channel groups
✓ **Clarity** - Clean, readable config files
✓ **Flexibility** - Individual, combined, or custom ROIs
✓ **Consistency** - Same channels used for stats and visualization
✓ **Reproducibility** - Version-controlled, tested, documented
✓ **Scientific rigor** - Clear rationale and transparency

This system brings sensor space ROI handling to the same level of organization and clarity as the established source space ROI system, while respecting the fundamental differences between the two domains.

---

**Author**: Claude (Anthropic)
**Date**: 2025-10-10
**Version**: 1.0
**Status**: Production-ready, fully tested
