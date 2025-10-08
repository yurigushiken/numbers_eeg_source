from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import mne


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONTAGE_PATH = (
    PROJECT_ROOT
    / "assets"
    / "Channel Location - Net128_v1.sfp"
    / "AdultAverageNet128_v1.sfp"
)
EPOCH_PATH = (
    PROJECT_ROOT
    / "data"
    / "data_preprocessed"
    / "hpf_1.0_lpf_35_baseline-on"
    / "sub-02_preprocessed-epo.fif"
)


def _load_montages():
    custom = mne.channels.read_custom_montage(MONTAGE_PATH)
    builtin = mne.channels.make_standard_montage("GSN-HydroCel-129")
    return custom, builtin


class TestMontageAlignment(unittest.TestCase):
    """Regression checks for montage compatibility with the project data."""

    def test_custom_montage_matches_preprocessed_channels(self):
        """Ensure the custom HydroCel montage aligns with the recorded channel labels."""
        custom, _ = _load_montages()
        epochs = mne.read_epochs(EPOCH_PATH, preload=False, verbose=False)

        self.assertEqual(len(custom.ch_names), 129)
        self.assertEqual(len(epochs.ch_names), 129)
        self.assertSetEqual(set(custom.ch_names), set(epochs.ch_names))

    def test_custom_and_builtin_montage_positions_are_close(self):
        """Validate that the custom Net128_v1 file is a close match to MNE's 129 template."""
        custom, builtin = _load_montages()
        custom_pos = custom.get_positions()["ch_pos"]
        builtin_pos = builtin.get_positions()["ch_pos"]

        distances = np.array(
            [
                np.linalg.norm(
                    np.array(custom_pos[ch]) - np.array(builtin_pos[ch])
                )
                for ch in custom.ch_names
            ]
        )

        # Allow small mismatches (few millimeters) due to template differences.
        self.assertLess(distances.max(), 0.025)
        self.assertLess(distances.mean(), 0.02)


if __name__ == "__main__":
    unittest.main()
