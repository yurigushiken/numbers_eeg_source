"""
TDD Tests for Combined Epoch File Loading

These tests verify that the new data loader can:
1. Load combined epoch files with metadata
2. Filter by condition using metadata
3. Maintain backward compatibility with split files
4. Apply baseline correction properly
5. Return same results as old approach
"""

import pytest
import numpy as np
import mne
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.utils.data_loader import (
    load_subject_combined_epochs,
    get_evoked_for_condition,
    CONDITION_SETS
)


class TestLoadCombinedEpochs:
    """Test suite for loading combined epoch files."""

    @pytest.fixture
    def test_data_paths(self):
        """Provide paths to test data."""
        return {
            'combined_data_root': Path(r"D:\numbers_eeg_source\data\data_preprocessed\hpf_1.0_lpf_35_baseline-on"),
            'split_data_root': Path(r"D:\numbers_eeg_source\data\all"),
            'test_subject': 'sub-02'
        }

    def test_load_combined_epochs_exists(self, test_data_paths):
        """
        TEST 1: Function load_subject_combined_epochs should exist and be callable.

        This test will FAIL initially because we haven't implemented the function yet.
        """
        from code.utils.data_loader import load_subject_combined_epochs

        # Should be callable
        assert callable(load_subject_combined_epochs), \
            "load_subject_combined_epochs should be a callable function"

    def test_load_combined_epochs_returns_epochs_object(self, test_data_paths):
        """
        TEST 2: load_subject_combined_epochs should return an MNE Epochs object.

        Expected: MNE Epochs object with metadata
        """
        from code.utils.data_loader import load_subject_combined_epochs

        epochs = load_subject_combined_epochs(
            subject_id='02',
            data_root=test_data_paths['combined_data_root']
        )

        # Should return Epochs object
        assert epochs is not None, "Should return epochs object for valid subject"
        assert isinstance(epochs, mne.BaseEpochs), \
            f"Should return MNE Epochs object, got {type(epochs)}"

    def test_loaded_epochs_has_metadata(self, test_data_paths):
        """
        TEST 3: Loaded epochs should have metadata with 'Condition' column.

        This is critical - we need metadata to filter by condition!
        """
        from code.utils.data_loader import load_subject_combined_epochs

        epochs = load_subject_combined_epochs(
            subject_id='02',
            data_root=test_data_paths['combined_data_root']
        )

        # Should have metadata
        assert epochs.metadata is not None, \
            "Epochs should have metadata attached"

        # Metadata should have 'Condition' column
        assert 'Condition' in epochs.metadata.columns, \
            "Metadata should have 'Condition' column"

        # Should have reasonable number of epochs
        assert len(epochs) > 0, "Should have at least one epoch"
        assert len(epochs) == len(epochs.metadata), \
            "Metadata rows should match epoch count"

    def test_metadata_has_expected_conditions(self, test_data_paths):
        """
        TEST 4: Metadata should contain expected condition codes.

        We expect conditions like "11", "12", "13", etc.
        """
        from code.utils.data_loader import load_subject_combined_epochs

        epochs = load_subject_combined_epochs(
            subject_id='02',
            data_root=test_data_paths['combined_data_root']
        )

        # Get unique conditions
        unique_conditions = epochs.metadata['Condition'].unique()

        # Should have multiple conditions
        assert len(unique_conditions) > 10, \
            f"Should have >10 unique conditions, got {len(unique_conditions)}"

        # Conditions should be numeric-ish (11, 12, etc.)
        # Could be int or str depending on metadata format
        sample_condition = str(unique_conditions[0])
        assert sample_condition.isdigit() or len(sample_condition) == 2, \
            f"Conditions should be 2-digit codes, got '{sample_condition}'"

    def test_load_nonexistent_subject_returns_none(self, test_data_paths):
        """
        TEST 5: Loading non-existent subject should return None gracefully.
        """
        from code.utils.data_loader import load_subject_combined_epochs

        epochs = load_subject_combined_epochs(
            subject_id='999',  # Non-existent
            data_root=test_data_paths['combined_data_root']
        )

        assert epochs is None, \
            "Should return None for non-existent subject"


class TestGetEvokedForConditionWithCombined:
    """Test suite for filtering conditions from combined files."""

    @pytest.fixture
    def test_data_paths(self):
        """Provide paths to test data."""
        return {
            'combined_data_root': Path(r"D:\numbers_eeg_source\data\data_preprocessed\hpf_1.0_lpf_35_baseline-on"),
            'split_data_root': Path(r"D:\numbers_eeg_source\data\all"),
            'test_subject': 'sub-02'
        }

    @pytest.fixture
    def sample_condition_info(self):
        """Sample condition configuration."""
        return {
            'condition_set_name': 'CARDINALITY_1',
            'name': 'Cardinality 1'
        }

    def test_get_evoked_accepts_use_combined_parameter(self, test_data_paths, sample_condition_info):
        """
        TEST 6: get_evoked_for_condition should accept use_combined parameter.

        This maintains backward compatibility.
        """
        from code.utils.data_loader import get_evoked_for_condition

        subject_dir = test_data_paths['split_data_root'] / test_data_paths['test_subject']

        # Should accept use_combined parameter without error
        try:
            evoked_list, epochs_list = get_evoked_for_condition(
                subject_dir=subject_dir,
                condition_info=sample_condition_info,
                baseline=(-0.1, 0.0),
                use_combined=False  # Use old split files
            )
            test_passed = True
        except TypeError as e:
            if "use_combined" in str(e):
                test_passed = False
            else:
                raise

        assert test_passed, \
            "get_evoked_for_condition should accept 'use_combined' parameter"

    def test_get_evoked_returns_correct_structure(self, test_data_paths, sample_condition_info):
        """
        TEST 7: get_evoked_for_condition should return (evoked_list, epochs_list).
        """
        from code.utils.data_loader import get_evoked_for_condition

        # For combined files, we need to pass a path with subject ID
        subject_dir = test_data_paths['combined_data_root'] / test_data_paths['test_subject']

        evoked_list, epochs_list = get_evoked_for_condition(
            subject_dir=subject_dir,
            condition_info=sample_condition_info,
            baseline=(-0.2, 0.0),  # New longer baseline
            use_combined=True
        )

        # Should return lists
        assert isinstance(evoked_list, list), "Should return list of evoked objects"
        assert isinstance(epochs_list, list), "Should return list of epochs objects"

        # Should have data (CARDINALITY_1 has condition "11")
        assert len(evoked_list) > 0, "Should have at least one evoked response"
        assert len(epochs_list) > 0, "Should have at least one epochs object"

    def test_filtered_epochs_match_expected_condition(self, test_data_paths):
        """
        TEST 8: Filtered epochs should only contain the requested condition.

        If we ask for CARDINALITY_1 (condition "11"), all epochs should be condition "11".
        """
        from code.utils.data_loader import get_evoked_for_condition

        condition_info = {
            'condition_set_name': 'CARDINALITY_1',
            'name': 'Cardinality 1'
        }

        subject_dir = test_data_paths['combined_data_root'] / test_data_paths['test_subject']

        evoked_list, epochs_list = get_evoked_for_condition(
            subject_dir=subject_dir,
            condition_info=condition_info,
            baseline=(-0.2, 0.0),
            use_combined=True
        )

        # Get the condition code from CONDITION_SETS
        expected_conditions = CONDITION_SETS['CARDINALITY_1']
        expected_codes = [num for nums in expected_conditions.values() for num in nums]

        # Check that all epochs match expected conditions
        for epochs in epochs_list:
            if hasattr(epochs, 'metadata') and epochs.metadata is not None:
                actual_conditions = epochs.metadata['Condition'].unique()
                for cond in actual_conditions:
                    assert str(cond) in expected_codes, \
                        f"Unexpected condition {cond}, expected one of {expected_codes}"

    def test_epoch_count_reasonable(self, test_data_paths):
        """
        TEST 9: Number of epochs should be reasonable for a single condition.

        Each condition should have ~5-10 epochs based on our earlier analysis.
        """
        from code.utils.data_loader import get_evoked_for_condition

        condition_info = {
            'condition_set_name': 'CARDINALITY_2',
            'name': 'Cardinality 2'
        }

        subject_dir = test_data_paths['combined_data_root'] / test_data_paths['test_subject']

        evoked_list, epochs_list = get_evoked_for_condition(
            subject_dir=subject_dir,
            condition_info=condition_info,
            baseline=(-0.2, 0.0),
            use_combined=True
        )

        total_epochs = sum(len(ep) for ep in epochs_list)

        # Should have reasonable number (5-15 epochs for single condition)
        assert 3 <= total_epochs <= 20, \
            f"Expected 3-20 epochs for single condition, got {total_epochs}"


class TestBackwardCompatibility:
    """Test that old split-file approach still works."""

    @pytest.fixture
    def test_data_paths(self):
        """Provide paths to test data."""
        return {
            'split_data_root': Path(r"D:\numbers_eeg_source\data\all"),
            'test_subject': 'sub-02'
        }

    def test_old_split_file_approach_still_works(self, test_data_paths):
        """
        TEST 10: Old split-file loading should still work (backward compatibility).

        When use_combined=False, should use old approach.
        """
        from code.utils.data_loader import get_evoked_for_condition

        condition_info = {
            'condition_set_name': 'CARDINALITY_1',
            'name': 'Cardinality 1'
        }

        subject_dir = test_data_paths['split_data_root'] / test_data_paths['test_subject']

        evoked_list, epochs_list = get_evoked_for_condition(
            subject_dir=subject_dir,
            condition_info=condition_info,
            baseline=(-0.1, 0.0),
            use_combined=False
        )

        # Should successfully load from split files
        assert evoked_list is not None and len(evoked_list) > 0, \
            "Should load evoked from split files"
        assert epochs_list is not None and len(epochs_list) > 0, \
            "Should load epochs from split files"


class TestDataQuality:
    """Test that loaded data has expected quality characteristics."""

    @pytest.fixture
    def test_data_paths(self):
        """Provide paths to test data."""
        return {
            'combined_data_root': Path(r"D:\numbers_eeg_source\data\data_preprocessed\hpf_1.0_lpf_35_baseline-on"),
            'test_subject': 'sub-02'
        }

    def test_baseline_applied_correctly(self, test_data_paths):
        """
        TEST 11: Baseline correction should be applied when specified.

        Data in baseline period should be centered near zero.
        """
        from code.utils.data_loader import get_evoked_for_condition

        condition_info = {
            'condition_set_name': 'CARDINALITY_1',
            'name': 'Cardinality 1'
        }

        baseline = (-0.2, 0.0)
        subject_dir = test_data_paths['combined_data_root']

        evoked_list, epochs_list = get_evoked_for_condition(
            subject_dir=subject_dir,
            condition_info=condition_info,
            baseline=baseline,
            use_combined=True
        )

        # Check evoked response
        for evoked in evoked_list:
            # Get data in baseline period
            baseline_mask = (evoked.times >= baseline[0]) & (evoked.times <= baseline[1])
            baseline_data = evoked.data[:, baseline_mask]

            # Mean should be close to zero (within 1e-12 for numerical precision)
            mean_baseline = np.abs(baseline_data.mean())
            assert mean_baseline < 1e-10, \
                f"Baseline mean should be ~0 after correction, got {mean_baseline}"

    def test_has_expected_time_range(self, test_data_paths):
        """
        TEST 12: Epochs should have expected time range (-200ms to 496ms).
        """
        from code.utils.data_loader import load_subject_combined_epochs

        epochs = load_subject_combined_epochs(
            subject_id='02',
            data_root=test_data_paths['combined_data_root']
        )

        # Time range should be approximately -0.2 to 0.496
        assert epochs.tmin <= -0.19, \
            f"Expected tmin around -0.2, got {epochs.tmin}"
        assert epochs.tmax >= 0.49, \
            f"Expected tmax around 0.496, got {epochs.tmax}"

    def test_has_129_channels(self, test_data_paths):
        """
        TEST 13: Should have 129 EEG channels as expected.
        """
        from code.utils.data_loader import load_subject_combined_epochs

        epochs = load_subject_combined_epochs(
            subject_id='02',
            data_root=test_data_paths['combined_data_root']
        )

        # Should have 129 channels
        n_channels = len(epochs.ch_names)
        assert n_channels == 129, \
            f"Expected 129 channels, got {n_channels}"


class TestComparisonWithOldData:
    """Compare results from new vs old approach."""

    @pytest.fixture
    def test_data_paths(self):
        """Provide paths to test data."""
        return {
            'combined_data_root': Path(r"D:\numbers_eeg_source\data\data_preprocessed\hpf_1.0_lpf_35_baseline-on"),
            'split_data_root': Path(r"D:\numbers_eeg_source\data\all"),
            'test_subject': 'sub-02'
        }

    def test_new_has_more_or_equal_epochs_than_old(self, test_data_paths):
        """
        TEST 14: New data should have >= epochs than old (better artifact rejection).

        Our analysis showed new data retains ~19% more epochs.
        """
        from code.utils.data_loader import get_evoked_for_condition

        condition_info = {
            'condition_set_name': 'CARDINALITY_1',
            'name': 'Cardinality 1'
        }

        # Load from old split files
        subject_dir_old = test_data_paths['split_data_root'] / test_data_paths['test_subject']
        _, epochs_list_old = get_evoked_for_condition(
            subject_dir=subject_dir_old,
            condition_info=condition_info,
            baseline=(-0.1, 0.0),
            use_combined=False
        )

        # Load from new combined files
        subject_dir_new = test_data_paths['combined_data_root'] / test_data_paths['test_subject']
        _, epochs_list_new = get_evoked_for_condition(
            subject_dir=subject_dir_new,
            condition_info=condition_info,
            baseline=(-0.2, 0.0),
            use_combined=True
        )

        old_count = sum(len(ep) for ep in epochs_list_old)
        new_count = sum(len(ep) for ep in epochs_list_new)

        # New should have >= epochs than old
        assert new_count >= old_count, \
            f"New data should have >= epochs (got new={new_count}, old={old_count})"


# Run instructions
if __name__ == "__main__":
    print("="*80)
    print("TDD TEST SUITE: Combined Epoch Loading")
    print("="*80)
    print()
    print("Run with: pytest tests/test_data_loader_combined.py -v")
    print()
    print("Expected: ALL TESTS SHOULD FAIL initially (TDD approach)")
    print("After implementation: All tests should PASS")
    print("="*80)
