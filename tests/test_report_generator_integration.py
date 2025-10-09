import re
from pathlib import Path

from code.utils.report_generator import create_html_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _write_png(path: Path) -> None:
    path.write_bytes(
        bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
            0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0x0F, 0x04, 0x00,
            0x09, 0xFB, 0x03, 0xFD, 0xA7, 0x80, 0xAD, 0x0E,
            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44,
            0xAE, 0x42, 0x60, 0x82,
        ])
    )

def test_data_quality_section_present(tmp_path: Path):
    sensor_config = PROJECT_ROOT / 'configs' / 'small_to_small_direction' / 'sensor_small_to_small_direction.yaml'

    sensor_output = tmp_path / 'sensor'
    sensor_output.mkdir()

    (sensor_output / 'sensor_small_to_small_direction_report.txt').write_text('RESULTS', encoding='utf-8')
    _write_png(sensor_output / 'sensor_small_to_small_direction_erp_cluster.png')
    _write_png(sensor_output / 'sensor_small_to_small_direction_topomap_cluster.png')

    report_path = tmp_path / 'report.html'

    create_html_report(
        sensor_config_path=sensor_config,
        sensor_output_dir=sensor_output,
        source_output_dir=None,
        loreta_output_dir=None,
        report_output_path=report_path,
    )

    html = report_path.read_text('utf-8')
    assert 'Data Preprocessing Information' in html
