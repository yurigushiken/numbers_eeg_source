import os
from pathlib import Path

from code.utils.config_utils import load_common_defaults, merge_common_into_config, resolve_data_source_dir


def test_merge_common_baseline_applied(tmp_path: Path):
    # Arrange: write a minimal common.yaml in a temp project
    project = tmp_path
    (project / "configs").mkdir(parents=True)
    (project / "configs" / "common.yaml").write_text(
        """
data:
  source: new
  preprocessing: hpf_1.0_lpf_35_baseline-on
epoch_defaults:
  baseline: [-0.2, 0.0]
""",
        encoding="utf-8",
    )

    analysis_cfg = {
        "analysis_name": "sensor_example",
        # epoch_window provided but missing baseline
        "epoch_window": {"tmin": -0.2, "tmax": 0.6},
    }

    common = load_common_defaults(project)
    merged = merge_common_into_config(common, analysis_cfg)

    assert merged["epoch_window"]["baseline"] == [-0.2, 0.0]


def test_merge_common_does_not_overwrite_analysis_baseline(tmp_path: Path):
    # Arrange: same common
    project = tmp_path
    (project / "configs").mkdir(parents=True)
    (project / "configs" / "common.yaml").write_text(
        """
data:
  source: new
  preprocessing: hpf_1.0_lpf_35_baseline-on
epoch_defaults:
  baseline: [-0.2, 0.0]
""",
        encoding="utf-8",
    )

    analysis_cfg = {
        "analysis_name": "sensor_example",
        "epoch_window": {"tmin": -0.1, "tmax": 0.5, "baseline": [-0.1, 0.0]},
    }

    common = load_common_defaults(project)
    merged = merge_common_into_config(common, analysis_cfg)

    assert merged["epoch_window"]["baseline"] == [-0.1, 0.0]


def test_resolve_dir_new_uses_preprocessing(tmp_path: Path):
    project = tmp_path
    base, mode = resolve_data_source_dir(project, data_source="new", preprocessing="variantA")
    assert str(base).endswith(os.path.join("data", "data_preprocessed", "variantA"))
    assert mode == "combined"


def test_resolve_dir_custom_path(tmp_path: Path):
    custom = tmp_path / "custom_root"
    base, mode = resolve_data_source_dir(tmp_path, data_source=str(custom), preprocessing=None)
    assert base == custom
    assert mode == "combined"

