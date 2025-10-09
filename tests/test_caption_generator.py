"""
Test-Driven Development for Enhanced Caption Generation

This module tests the generation of scientifically informative figure captions
that include directional interpretation and functional context.
"""

import pytest
from pathlib import Path


class TestCaptionGeneration:
    """Test suite for caption generation functions."""

    def test_sensor_caption_negative_cluster_interpretation(self):
        """Test that negative t-value sensor clusters are correctly interpreted."""
        from code.utils.caption_generator import generate_sensor_caption

        cluster_info = {
            'p_value': 0.0004,
            'peak_t': -6.654,
            'n_channels': 42,
            'time_window': '104.0 ms to 176.0 ms',
            'cluster_mass': -1613.67,
            'topography': 'posterior'  # occipital-parietal
        }
        contrast_name = "All Increasing vs. All Decreasing (Full 1-6 Range)"

        caption = generate_sensor_caption(
            cluster_id=1,
            cluster_info=cluster_info,
            contrast_name=contrast_name,
            time_window_label="N1"
        )

        # Caption should explain negative t-values
        assert "Negative t-values" in caption or "negative t-values" in caption
        assert "Increasing" in caption
        assert "Decreasing" in caption

        # Should indicate stronger activation for first condition
        assert "stronger activation for increasing" in caption.lower() or \
               "increasing > decreasing" in caption.lower()

        # Should mention the functional interpretation
        assert "perceptual" in caption.lower() or "processing" in caption.lower()

        # Should include statistical details
        assert "p=0.0004" in caption or "p = 0.0004" in caption
        assert "-6.65" in caption or "-6.654" in caption
        assert "42 channels" in caption


    def test_sensor_caption_positive_cluster_interpretation(self):
        """Test that positive t-value sensor clusters are correctly interpreted."""
        from code.utils.caption_generator import generate_sensor_caption

        cluster_info = {
            'p_value': 0.0014,
            'peak_t': 6.994,
            'n_channels': 44,
            'time_window': '108.0 ms to 160.0 ms',
            'cluster_mass': 1293.51,
            'topography': 'frontal'
        }
        contrast_name = "All Increasing vs. All Decreasing (Full 1-6 Range)"

        caption = generate_sensor_caption(
            cluster_id=2,
            cluster_info=cluster_info,
            contrast_name=contrast_name,
            time_window_label="N1"
        )

        # Caption should explain positive t-values
        assert "Positive t-values" in caption or "positive t-values" in caption

        # Should indicate stronger activation for second condition (Decreasing)
        assert "stronger activation for decreasing" in caption.lower() or \
               "decreasing > increasing" in caption.lower()

        # Should mention frontal regions or cognitive control
        assert "frontal" in caption.lower() or "cognitive" in caption.lower()


    def test_source_caption_with_anatomical_context(self):
        """Test that source captions include anatomical and functional context."""
        from code.utils.caption_generator import generate_source_caption

        cluster_info = {
            'p_value': 0.0168,
            'peak_t': -5.585,
            'n_vertices': 322,
            'peak_mni': '(-22.6, -97.4, 10.8)',
            'primary_region': 'lateraloccipital-lh',
            'top_regions': ['lateraloccipital-lh', 'inferiorparietal-lh', 'superiorparietal-lh']
        }
        contrast_name = "All Increasing vs. All Decreasing (Full 1-6 Range)"

        caption = generate_source_caption(
            cluster_id=1,
            cluster_info=cluster_info,
            contrast_name=contrast_name,
            method="dSPM",
            time_window_label="N1",
            analysis_window="80-200 ms"
        )

        # Should include anatomical region
        assert "lateral occipital" in caption.lower() or "extrastriate" in caption.lower()

        # Should explain directional effect
        assert "Negative t-values" in caption or "negative t-values" in caption

        # Should mention MNI coordinates
        assert "MNI" in caption
        assert "-22.6" in caption or "(-22.6" in caption

        # Should include functional interpretation
        assert "visual" in caption.lower() or "perceptual" in caption.lower()


    def test_caption_time_window_labeling(self):
        """Test that captions use informative time window labels (N1, P3b)."""
        from code.utils.caption_generator import generate_sensor_caption

        # N1 window
        cluster_info_n1 = {
            'p_value': 0.001,
            'peak_t': -5.0,
            'n_channels': 30,
            'time_window': '100.0 ms to 180.0 ms',
            'topography': 'posterior'
        }

        caption_n1 = generate_sensor_caption(
            cluster_id=1,
            cluster_info=cluster_info_n1,
            contrast_name="Test Contrast",
            time_window_label="N1"
        )

        assert "N1" in caption_n1

        # P3b window
        cluster_info_p3b = {
            'p_value': 0.001,
            'peak_t': 4.0,
            'n_channels': 25,
            'time_window': '300.0 ms to 500.0 ms',
            'topography': 'parietal'
        }

        caption_p3b = generate_sensor_caption(
            cluster_id=1,
            cluster_info=cluster_info_p3b,
            contrast_name="Test Contrast",
            time_window_label="P3b"
        )

        assert "P3b" in caption_p3b


    def test_caption_handles_missing_topography(self):
        """Test that captions gracefully handle missing topography information."""
        from code.utils.caption_generator import generate_sensor_caption

        cluster_info = {
            'p_value': 0.01,
            'peak_t': -3.5,
            'n_channels': 20,
            'time_window': '100.0 ms to 200.0 ms'
            # No 'topography' key
        }

        caption = generate_sensor_caption(
            cluster_id=1,
            cluster_info=cluster_info,
            contrast_name="Test Contrast"
        )

        # Should still generate a valid caption
        assert len(caption) > 50
        assert "p=" in caption.lower()


    def test_source_caption_positive_t_values(self):
        """Test source caption generation for positive t-values (Decreasing > Increasing)."""
        from code.utils.caption_generator import generate_source_caption

        cluster_info = {
            'p_value': 0.02,
            'peak_t': 5.2,  # Positive
            'n_vertices': 250,
            'peak_mni': '(-30.0, -50.0, 45.0)',
            'primary_region': 'inferiorparietal-lh',
            'top_regions': ['inferiorparietal-lh', 'superiorparietal-lh']
        }

        caption = generate_source_caption(
            cluster_id=1,
            cluster_info=cluster_info,
            contrast_name="All Increasing vs. All Decreasing (Full 1-6 Range)",
            method="dSPM",
            time_window_label="P3b",
            analysis_window="300-500 ms"
        )

        # Should explain positive t-values correctly
        assert "Positive t-values" in caption or "positive t-values" in caption
        assert "Decreasing" in caption

        # P3b context should mention parietal/updating
        assert "parietal" in caption.lower() or "updating" in caption.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
