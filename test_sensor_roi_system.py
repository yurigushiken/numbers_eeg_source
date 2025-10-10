#!/usr/bin/env python
"""
Test script to verify the new centralized sensor ROI system.

This script tests:
1. Loading sensor ROI definitions from the central config file
2. Resolving channel groups to channel lists
3. Nested channel group resolution (e.g., posterior_visual_parietal)
"""

import yaml
from pathlib import Path
from code.utils.cluster_stats import _load_sensor_roi_definitions, _resolve_sensor_roi_channels


def test_load_sensor_roi_definitions():
    """Test loading the sensor ROI definitions file."""
    print("=" * 80)
    print("TEST 1: Loading Sensor ROI Definitions")
    print("=" * 80)

    roi_defs = _load_sensor_roi_definitions()

    if not roi_defs:
        print("[FAIL] FAILED: No ROI definitions loaded")
        return False

    print(f"[PASS] Loaded {len(roi_defs)} ROI definitions:")
    for roi_name, roi_info in roi_defs.items():
        desc = roi_info.get('description', 'No description')
        if 'channels' in roi_info:
            n_channels = len(roi_info['channels'])
            print(f"  - {roi_name}: {n_channels} channels ({desc})")
        elif 'channel_groups' in roi_info:
            groups = ', '.join(roi_info['channel_groups'])
            print(f"  - {roi_name}: groups [{groups}] ({desc})")

    # Verify expected ROIs exist
    expected_rois = ['N1_bilateral', 'P1_Oz', 'P3b_midline', 'posterior_visual_parietal']
    missing = [roi for roi in expected_rois if roi not in roi_defs]
    if missing:
        print(f"[FAIL] FAILED: Missing expected ROIs: {missing}")
        return False

    print("[PASS] All expected ROI definitions found\n")
    return True


def test_resolve_individual_roi():
    """Test resolving a single ROI group."""
    print("=" * 80)
    print("TEST 2: Resolving Individual ROI (N1_bilateral)")
    print("=" * 80)

    roi_cfg = {
        'channel_groups': ['N1_bilateral']
    }

    channels = _resolve_sensor_roi_channels(roi_cfg)

    if not channels:
        print("[FAIL] FAILED: No channels resolved")
        return False

    print(f"[PASS] Resolved {len(channels)} channels:")
    print(f"  {sorted(channels)}\n")

    # Verify expected channels
    expected_channels = {'E66', 'E65', 'E59', 'E60', 'E67', 'E71', 'E70', 'E84',
                        'E76', 'E77', 'E85', 'E91', 'E90', 'E83'}
    if channels != expected_channels:
        print(f"[FAIL] FAILED: Channel mismatch")
        print(f"  Expected: {sorted(expected_channels)}")
        print(f"  Got: {sorted(channels)}")
        return False

    print("[PASS] All expected channels present\n")
    return True


def test_resolve_multiple_rois():
    """Test resolving multiple ROI groups."""
    print("=" * 80)
    print("TEST 3: Resolving Multiple ROIs (N1 + P1 + P3b)")
    print("=" * 80)

    roi_cfg = {
        'channel_groups': ['N1_bilateral', 'P1_Oz', 'P3b_midline']
    }

    channels = _resolve_sensor_roi_channels(roi_cfg)

    if not channels:
        print("[FAIL] FAILED: No channels resolved")
        return False

    print(f"[PASS] Resolved {len(channels)} channels from 3 ROI groups:")
    print(f"  {sorted(channels)}\n")

    # The union should have some overlap (e.g., E71, E77 appear in multiple groups)
    # So total should be less than sum of individual groups
    if len(channels) > 30:  # Sanity check
        print(f"[FAIL] FAILED: Too many channels ({len(channels)}), possible duplication issue")
        return False

    print(f"[PASS] Reasonable channel count (handles overlaps correctly)\n")
    return True


def test_resolve_nested_roi():
    """Test resolving nested ROI (posterior_visual_parietal)."""
    print("=" * 80)
    print("TEST 4: Resolving Nested ROI (posterior_visual_parietal)")
    print("=" * 80)

    roi_cfg = {
        'channel_groups': ['posterior_visual_parietal']
    }

    channels = _resolve_sensor_roi_channels(roi_cfg)

    if not channels:
        print("[FAIL] FAILED: No channels resolved")
        return False

    print(f"[PASS] Resolved {len(channels)} channels from nested ROI:")
    print(f"  {sorted(channels)}\n")

    # This should equal the union of N1 + P1 + P3b
    roi_cfg_manual = {
        'channel_groups': ['N1_bilateral', 'P1_Oz', 'P3b_midline']
    }
    channels_manual = _resolve_sensor_roi_channels(roi_cfg_manual)

    if channels != channels_manual:
        print(f"[FAIL] FAILED: Nested ROI doesn't match manual union")
        print(f"  Nested: {len(channels)} channels")
        print(f"  Manual: {len(channels_manual)} channels")
        return False

    print(f"[PASS] Nested ROI correctly resolves to union of component groups\n")
    return True


def test_direct_channels():
    """Test direct channel specification."""
    print("=" * 80)
    print("TEST 5: Direct Channel Specification")
    print("=" * 80)

    roi_cfg = {
        'channels': ['E71', 'E75', 'E76']
    }

    channels = _resolve_sensor_roi_channels(roi_cfg)

    expected = {'E71', 'E75', 'E76'}
    if channels != expected:
        print(f"[FAIL] FAILED: Direct channel specification mismatch")
        print(f"  Expected: {expected}")
        print(f"  Got: {channels}")
        return False

    print(f"[PASS] Direct channels correctly resolved: {sorted(channels)}\n")
    return True


def test_mixed_specification():
    """Test mixing channel groups and direct channels."""
    print("=" * 80)
    print("TEST 6: Mixed Specification (groups + direct channels)")
    print("=" * 80)

    roi_cfg = {
        'channel_groups': ['N1_bilateral'],
        'channels': ['E1', 'E2', 'E3']  # Add some extra channels
    }

    channels = _resolve_sensor_roi_channels(roi_cfg)

    # Should include both N1 channels and the direct channels
    if 'E1' not in channels or 'E66' not in channels:
        print(f"[FAIL] FAILED: Missing channels from mixed specification")
        return False

    print(f"[PASS] Mixed specification resolved to {len(channels)} channels")
    print(f"  Includes both group channels and direct channels\n")
    return True


def test_example_config():
    """Test loading an actual sensor config and verifying ROI format."""
    print("=" * 80)
    print("TEST 7: Example Config File Format")
    print("=" * 80)

    config_path = Path("d:/numbers_eeg_source/configs/small_to_small_direction/sensor_small_to_small_direction.yaml")

    if not config_path.exists():
        print(f"[FAIL] FAILED: Config file not found: {config_path}")
        return False

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check that roi_channels section has been removed
    if 'roi_channels' in config:
        print(f"[FAIL] FAILED: Old roi_channels section still present in config")
        return False

    print(f"[PASS] Config file cleaned up (no roi_channels section)")

    # Check for new ROI comment format
    with open(config_path, 'r') as f:
        content = f.read()

    if 'centrally managed in configs/sensor_roi_definitions.yaml' not in content:
        print(f"[FAIL] FAILED: New ROI comment format not found")
        return False

    print(f"[PASS] New ROI comment format present")
    print(f"[PASS] Config references centralized sensor_roi_definitions.yaml\n")
    return True


def main():
    """Run all tests."""
    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " " * 78 + "|")
    print("|" + "  SENSOR ROI SYSTEM VERIFICATION TESTS".center(78) + "|")
    print("|" + " " * 78 + "|")
    print("+" + "=" * 78 + "+")
    print("\n")

    tests = [
        ("Load ROI Definitions", test_load_sensor_roi_definitions),
        ("Resolve Individual ROI", test_resolve_individual_roi),
        ("Resolve Multiple ROIs", test_resolve_multiple_rois),
        ("Resolve Nested ROI", test_resolve_nested_roi),
        ("Direct Channel Spec", test_direct_channels),
        ("Mixed Specification", test_mixed_specification),
        ("Example Config Format", test_example_config),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"[FAIL] EXCEPTION in {name}: {e}\n")
            results.append((name, False))

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, passed in results:
        status = "[PASS] PASS" if passed else "[FAIL] FAIL"
        print(f"{status}: {name}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print("\n" + "=" * 80)
    print(f"TOTAL: {total_passed}/{total_tests} tests passed")
    print("=" * 80 + "\n")

    if total_passed == total_tests:
        print("SUCCESS: All tests passed! The centralized sensor ROI system is working correctly.\n")
        return 0
    else:
        print("WARNING: Some tests failed. Please review the output above.\n")
        return 1


if __name__ == "__main__":
    exit(main())
