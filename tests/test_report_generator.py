import json
import textwrap
from pathlib import Path

import pytest

from code.utils import report_generator


def _write_dummy_png(path: Path) -> None:
    """Write a minimal valid 1x1 PNG so relpaths resolve in the report."""
    path.write_bytes(
        bytes(
            [
                0x89,
                0x50,
                0x4E,
                0x47,
                0x0D,
                0x0A,
                0x1A,
                0x0A,
                0x00,
                0x00,
                0x00,
                0x0D,
                0x49,
                0x48,
                0x44,
                0x52,
                0x00,
                0x00,
                0x00,
                0x01,
                0x00,
                0x00,
                0x00,
                0x01,
                0x08,
                0x02,
                0x00,
                0x00,
                0x00,
                0x90,
                0x77,
                0x53,
                0xDE,
                0x00,
                0x00,
                0x00,
                0x0B,
                0x49,
                0x44,
                0x41,
                0x54,
                0x08,
                0xD7,
                0x63,
                0x60,
                0x00,
                0x00,
                0x00,
                0x02,
                0x00,
                0x01,
                0xE2,
                0x21,
                0xBC,
                0x33,
                0x00,
                0x00,
                0x00,
                0x00,
                0x49,
                0x45,
                0x4E,
                0x44,
                0xAE,
                0x42,
                0x60,
                0x82,
            ]
        )
    )


def _write_sensor_config(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            analysis_name: "sensor_dummy_all_direction_effect"
            domain: "sensor"
            contrast:
              name: "Dummy contrast"
              condition_A:
                name: "CondA"
                condition_set_name: "SET_A"
              condition_B:
                name: "CondB"
                condition_set_name: "SET_B"
              combination_weights: [1, -1]
            epoch_window:
              tmin: -0.1
              tmax: 0.6
              baseline: [-0.1, 0.0]
            stats:
              analysis_window: [0.0, 0.5]
              p_threshold: 0.001
              cluster_alpha: 0.05
              n_permutations: 1024
              tail: 0
              connectivity: "eeg"
            """
        ).strip()
    )


def _write_source_config(path: Path, method: str, window: tuple[float, float]) -> None:
    path.write_text(
        textwrap.dedent(
            f"""
            analysis_name: "{path.stem}"
            domain: "source"
            contrast:
              name: "Dummy contrast"
              condition_A:
                name: "CondA"
                condition_set_name: "SET_A"
              condition_B:
                name: "CondB"
                condition_set_name: "SET_B"
              combination_weights: [1, -1]
            epoch_window:
              tmin: -0.2
              tmax: 0.6
              baseline: [-0.2, 0.0]
            source:
              method: "{method}"
              snr: 2.0
            stats:
              analysis_window: [{window[0]}, {window[1]}]
              p_threshold: 0.05
              cluster_alpha: 0.05
              n_permutations: 1000
              tail: 0
            """
        ).strip()
    )


def _write_sensor_outputs(output_dir: Path) -> None:
    analysis_name = "sensor_dummy_all_direction_effect"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{analysis_name}_report.txt"
    report_path.write_text(
        textwrap.dedent(
            """
            ================================================================================
            RESULTS
            ================================================================================

            Found 1 significant cluster(s).

            ----------------------------------------
            Cluster #1 (p-value = 0.0005)
            ----------------------------------------
              - Cluster mass (sum of t-values): -123.45
              - Peak t-value: -5.000 at 125.0 ms on channel E1
              - Time window: 100.0 ms to 150.0 ms
              - Number of channels: 12
            """
        ).strip()
    )

    for suffix in ["erp_cluster", "topomap_cluster"]:
        _write_dummy_png(output_dir / f"{analysis_name}_{suffix}.png")


def _write_source_outputs(output_dir: Path, analysis_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{analysis_name}_report.txt"
    report_path.write_text(
        textwrap.dedent(
            """
            RESULTS
            ================================================================================

            Found 1 significant cluster(s).

            ----------------------------------------
            Cluster #1 (p-value = 0.0123)
            ----------------------------------------
              - Cluster mass (sum of t-values): -456.7
              - Peak t-value: -5.585 at 172.0 ms (vertex #1536)
              - Time window: 80.0 ms to 200.0 ms
              - Number of vertices: 322
            """
        ).strip()
    )

    _write_dummy_png(output_dir / f"{analysis_name}_source_cluster.png")
    (output_dir / "aux").mkdir(exist_ok=True)
    (output_dir / "inverse_provenance.json").write_text(json.dumps([]))


@pytest.mark.parametrize("source_is_list", [True, False])
def test_create_html_report_with_multiple_source_sections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_is_list: bool) -> None:
    monkeypatch.setattr(report_generator, "_export_html_to_pdf", lambda _: None)

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    sensor_config_path = configs_dir / "sensor_dummy_all_direction_effect.yaml"
    _write_sensor_config(sensor_config_path)

    # Matching source configs for analysis table discovery
    _write_source_config(configs_dir / "source_dspm_n1_all_direction_effect.yaml", "dSPM", (0.08, 0.20))
    _write_source_config(configs_dir / "source_dspm_p3b_all_direction_effect.yaml", "dSPM", (0.30, 0.50))
    _write_source_config(configs_dir / "source_loreta_n1_all_direction_effect.yaml", "eLORETA", (0.08, 0.20))
    _write_source_config(configs_dir / "source_loreta_p3b_all_direction_effect.yaml", "eLORETA", (0.30, 0.50))

    sensor_output_dir = tmp_path / "derivatives" / "sensor" / "20250101_000000-sensor_dummy_all_direction_effect"
    _write_sensor_outputs(sensor_output_dir)

    dspm_n1_dir = tmp_path / "derivatives" / "source" / "20250101_000100-source_dspm_n1_all_direction_effect"
    dspm_p3b_dir = tmp_path / "derivatives" / "source" / "20250101_000200-source_dspm_p3b_all_direction_effect"
    loreta_n1_dir = tmp_path / "derivatives" / "source" / "20250101_000300-source_loreta_n1_all_direction_effect"
    loreta_p3b_dir = tmp_path / "derivatives" / "source" / "20250101_000400-source_loreta_p3b_all_direction_effect"

    _write_source_outputs(dspm_n1_dir, "source_dspm_n1_all_direction_effect")
    _write_source_outputs(dspm_p3b_dir, "source_dspm_p3b_all_direction_effect")
    _write_source_outputs(loreta_n1_dir, "source_loreta_n1_all_direction_effect")
    _write_source_outputs(loreta_p3b_dir, "source_loreta_p3b_all_direction_effect")

    report_output_path = tmp_path / "derivatives" / "reports" / "dummy_report.html"
    report_output_path.parent.mkdir(parents=True, exist_ok=True)

    dspm_arg = [dspm_n1_dir, dspm_p3b_dir] if source_is_list else dspm_n1_dir
    loreta_arg = [loreta_n1_dir, loreta_p3b_dir] if source_is_list else loreta_n1_dir

    report_generator.create_html_report(
        sensor_config_path=sensor_config_path,
        sensor_output_dir=sensor_output_dir,
        source_output_dir=dspm_arg,
        loreta_output_dir=loreta_arg,
        report_output_path=report_output_path,
        run_command="python -m code.run_full_analysis_pipeline",
        accuracy="acc1",
        data_source="new",
    )

    html = report_output_path.read_text(encoding="utf-8")

    assert html.count("Source-Space Localization (dSPM)") == (2 if source_is_list else 1)
    assert html.count("Source-Space Localization (eLORETA)") == (2 if source_is_list else 1)

    # Ensure the analysis-specific assets are referenced in the HTML
    assert "source_dspm_n1_all_direction_effect_source_cluster.png" in html
    assert ("source_dspm_p3b_all_direction_effect_source_cluster.png" in html) == source_is_list
    assert "source_loreta_n1_all_direction_effect_source_cluster.png" in html
    assert ("source_loreta_p3b_all_direction_effect_source_cluster.png" in html) == source_is_list
