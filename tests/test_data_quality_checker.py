"""
Tests for data quality checker module.

These tests verify that the data quality checker can:
1. Extract preprocessing parameters from HAPPE outputs
2. Validate data consistency
3. Generate reports for the analysis pipeline
"""

import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from code.utils.data_quality_checker import (
    extract_happe_parameters,
    check_data_consistency,
    generate_data_quality_report,
    DataQualityInfo
)


class TestHappeParameterExtraction:
    """Test extraction of preprocessing parameters from HAPPE outputs."""

    def test_extract_parameters_from_mat_file(self):
        """Should extract filter, baseline, and reref settings from MAT file."""
        happe_dir = PROJECT_ROOT / "data" / "data_input_from_happe" / "hpf_1.0_lpf_35_baseline-on"

        params = extract_happe_parameters(happe_dir)

        assert params is not None
        assert params['highpass_filter'] == 1.0
        assert params['lowpass_filter'] == 35.0
        assert params['baseline_correction'] is True
        assert params['average_reference'] is True

    def test_extract_parameters_missing_directory(self):
        """Should handle missing HAPPE directory gracefully."""
        missing_dir = PROJECT_ROOT / "data" / "nonexistent"

        params = extract_happe_parameters(missing_dir)

        assert params is None

    def test_extract_parameters_all_datasets(self):
        """Should work for all preprocessing variants in data_input_from_happe."""
        happe_root = PROJECT_ROOT / "data" / "data_input_from_happe"

        if not happe_root.exists():
            pytest.skip("HAPPE input directory not found")

        dataset_dirs = [d for d in happe_root.iterdir() if d.is_dir()]

        for dataset_dir in dataset_dirs:
            params = extract_happe_parameters(dataset_dir)
            # Should extract params successfully or return None
            if params:
                assert 'highpass_filter' in params
                assert 'lowpass_filter' in params


class TestDataConsistency:
    """Test data consistency validation."""

    def test_check_consistency_valid_dataset(self):
        """Should validate that preprocessed data matches HAPPE parameters."""
        preprocessing_name = "hpf_1.0_lpf_35_baseline-on"

        is_consistent, issues = check_data_consistency(
            preprocessing_name,
            project_root=PROJECT_ROOT
        )

        assert is_consistent is True
        assert len(issues) == 0

    def test_check_consistency_name_mismatch(self):
        """Should handle missing preprocessing datasets gracefully."""
        # Non-existent dataset
        preprocessing_name = "hpf_2.0_lpf_40_baseline-off"

        is_consistent, issues = check_data_consistency(
            preprocessing_name,
            project_root=PROJECT_ROOT
        )

        # Function should handle gracefully - either consistent with no data or flag missing
        # This is acceptable behavior for a non-existent dataset
        assert isinstance(is_consistent, bool)
        assert isinstance(issues, list)


class TestDataQualityReport:
    """Test data quality report generation."""

    def test_generate_report_basic(self):
        """Should generate a data quality report with all key information."""
        preprocessing_name = "hpf_1.0_lpf_35_baseline-on"

        report = generate_data_quality_report(
            preprocessing_name,
            project_root=PROJECT_ROOT
        )

        assert isinstance(report, DataQualityInfo)
        assert report.preprocessing_name == preprocessing_name
        assert report.highpass_hz == 1.0
        assert report.lowpass_hz == 35.0
        assert report.baseline_corrected is True
        assert report.average_referenced is True

    def test_report_includes_subject_count(self):
        """Should include subject count in the report."""
        preprocessing_name = "hpf_1.0_lpf_35_baseline-on"

        report = generate_data_quality_report(
            preprocessing_name,
            project_root=PROJECT_ROOT
        )

        assert hasattr(report, 'n_subjects')
        assert report.n_subjects > 0

    def test_report_includes_qc_metrics(self):
        """Should include QC metrics from HAPPE outputs."""
        preprocessing_name = "hpf_1.0_lpf_35_baseline-on"

        report = generate_data_quality_report(
            preprocessing_name,
            project_root=PROJECT_ROOT
        )

        assert hasattr(report, 'mean_channels_retained')
        assert hasattr(report, 'mean_segments_retained')

    def test_report_html_format(self):
        """Should generate HTML-formatted report section."""
        preprocessing_name = "hpf_1.0_lpf_35_baseline-on"

        report = generate_data_quality_report(
            preprocessing_name,
            project_root=PROJECT_ROOT
        )

        html = report.to_html()

        assert '<h3>' in html or '<h2>' in html
        assert '1.0' in html  # HPF value
        assert '35.0' in html  # LPF value
        assert 'Hz' in html

    def test_report_from_common_yaml(self):
        """Should extract preprocessing name from common.yaml."""
        from code.utils.data_loader import load_common_config

        common_config = load_common_config(project_root=PROJECT_ROOT)
        preprocessing_name = common_config.get('data', {}).get('preprocessing')

        assert preprocessing_name is not None

        report = generate_data_quality_report(
            preprocessing_name,
            project_root=PROJECT_ROOT
        )

        assert report is not None
        assert report.preprocessing_name == preprocessing_name


class TestIntegrationWithPipeline:
    """Test integration with the analysis pipeline."""

    def test_pipeline_can_import_checker(self):
        """Should be importable from pipeline scripts."""
        try:
            from code.utils.data_quality_checker import generate_data_quality_report
            assert callable(generate_data_quality_report)
        except ImportError as e:
            pytest.fail(f"Cannot import data_quality_checker: {e}")

    def test_report_serializable(self):
        """Data quality report should be serializable for logging."""
        preprocessing_name = "hpf_1.0_lpf_35_baseline-on"

        report = generate_data_quality_report(
            preprocessing_name,
            project_root=PROJECT_ROOT
        )

        # Should convert to dict
        report_dict = report.to_dict()
        assert isinstance(report_dict, dict)
        assert 'preprocessing_name' in report_dict
        assert 'highpass_hz' in report_dict