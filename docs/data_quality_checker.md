# Data Quality Checker

## Overview

The Data Quality Checker is a module that extracts and validates preprocessing parameters from HAPPE outputs to ensure accurate reporting and data consistency in your EEG analysis pipeline.

## Purpose

When running EEG analyses, it's critical to know exactly what preprocessing was applied to your data. Instead of relying on directory names, the Data Quality Checker:

1. **Extracts actual preprocessing parameters** from HAPPE's MAT and CSV files
2. **Validates consistency** between expected and actual parameters
3. **Generates comprehensive reports** for inclusion in analysis outputs
4. **Provides QC metrics** like channel retention and data quality statistics

## Features

### Parameter Extraction

The checker extracts the following information from HAPPE outputs:

- **High-pass filter** (Hz)
- **Low-pass filter** (Hz)
- **Baseline correction** (on/off)
- **Average reference** (on/off)
- **Number of subjects** processed
- **Mean channels retained** per subject
- **Mean segments retained** (%) per subject
- **HAPPE version** used

### Data Sources

The checker reads from:
- `data/data_input_from_happe/[preprocessing_variant]/input_parameters/inputParameters_*.mat`
- `data/data_input_from_happe/[preprocessing_variant]/6 - quality_assessment_outputs/HAPPE_dataQC_*.csv`

### Integration

The checker is automatically integrated into `run_full_analysis_pipeline.py` and will:

1. Read the preprocessing variant from `configs/common.yaml`
2. Extract actual parameters from HAPPE outputs
3. Generate an HTML section showing data preprocessing information
4. Include this section in the final analysis report

## Usage

### Programmatic Use

```python
from pathlib import Path
from code.utils.data_quality_checker import get_preprocessing_info_from_config

# Get data quality info from common.yaml
report = get_preprocessing_info_from_config(project_root=Path('.'))

if report:
    print(f"Preprocessing: {report.preprocessing_name}")
    print(f"HPF: {report.highpass_hz} Hz")
    print(f"LPF: {report.lowpass_hz} Hz")
    print(f"Subjects: {report.n_subjects}")

    # Generate HTML for report
    html = report.to_html()

    # Generate text for logging
    text = report.to_text()
```

### Direct Parameter Extraction

```python
from pathlib import Path
from code.utils.data_quality_checker import generate_data_quality_report

# Generate report for a specific preprocessing variant
report = generate_data_quality_report(
    preprocessing_name="hpf_1.0_lpf_35_baseline-on",
    project_root=Path('.')
)
```

### Consistency Checking

```python
from pathlib import Path
from code.utils.data_quality_checker import check_data_consistency

# Verify that preprocessing name matches actual parameters
is_consistent, issues = check_data_consistency(
    preprocessing_name="hpf_1.0_lpf_35_baseline-on",
    project_root=Path('.')
)

if not is_consistent:
    for issue in issues:
        print(f"⚠️ {issue}")
```

## Example Output

When integrated into the analysis pipeline, the data quality section appears in HTML reports:

```
DATA PREPROCESSING INFORMATION
==============================

Preprocessing Dataset: hpf_1.0_lpf_35_baseline-on
High-pass Filter: 1.0 Hz
Low-pass Filter: 35.0 Hz
Baseline Correction: Yes
Reference: Average reference
Number of Subjects: 24
Mean Channels Retained: 113.4
Mean Segments Retained: 56.5%
```

## Benefits for Your Research

1. **Transparency**: Exact preprocessing parameters are documented in every report
2. **Reproducibility**: Anyone reading your report knows exactly what was done
3. **Quality Control**: QC metrics show data quality at a glance
4. **Error Prevention**: Catches mismatches between directory names and actual parameters
5. **Career Protection**: Reviewers can see you've been thorough and rigorous

## Testing

The module includes comprehensive tests in `tests/test_data_quality_checker.py`:

```bash
# Run all data quality checker tests
python -m pytest tests/test_data_quality_checker.py -v
```

Tests cover:
- Parameter extraction from MAT files
- QC metric extraction from CSV files
- Consistency validation
- HTML/text report generation
- Integration with the analysis pipeline

## File Structure

```
code/
  utils/
    data_quality_checker.py    # Main module
    data_loader.py              # Added load_common_config()
    report_generator.py         # Integrated data quality section

tests/
  test_data_quality_checker.py # Comprehensive test suite

data/
  data_input_from_happe/
    [preprocessing_variant]/
      input_parameters/
        inputParameters_*.mat   # Source of truth for parameters
      6 - quality_assessment_outputs/
        HAPPE_dataQC_*.csv      # Source of QC metrics
```

## Future Enhancements

Potential improvements:
- Add warnings if QC metrics fall below thresholds
- Export QC summary to standalone CSV for cross-analysis comparisons
- Add visualization of QC metrics across subjects
- Support for additional preprocessing tools beyond HAPPE

## Author

Language and Cognitive Neuroscience Lab
Teachers College, Columbia University
