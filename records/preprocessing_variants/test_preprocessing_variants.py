import pandas as pd

from pathlib import Path

from temp_tests.evaluate_preprocessing_variants import evaluate_preprocessing_variants


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_preprocessing_variants_subset(tmp_path):
    # Use a small subset of datasets for a quick diagnostic run
    datasets = [
        "hpf_1.0_lpf_35_baseline-on",
        "hpf_1.5_lpf_35_baseline-on",
    ]
    df = evaluate_preprocessing_variants(
        config_path=PROJECT_ROOT / "configs/all_direction_effect/sensor_all_direction_effect.yaml",
        accuracy="acc1",
        datasets=datasets,
        max_subjects=2,
    )

    # Basic sanity checks
    assert set(df["dataset"]) == set(datasets)
    for column in [
        "mean_N1_snr",
        "mean_P3b_snr",
        "mean_N1_peak_uv",
        "mean_P3b_peak_uv",
    ]:
        assert column in df.columns

    # Ensure CSV output works through helper
    output_dir = tmp_path / "out"
    from temp_tests.evaluate_preprocessing_variants import _write_outputs

    _write_outputs(df, output_dir)
    assert (output_dir / "preprocessing_metrics.csv").exists()
    assert (output_dir / "preprocessing_metrics.md").exists()
