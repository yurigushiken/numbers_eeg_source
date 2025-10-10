"""
Regenerate report for timestamp 20251009_215424
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from code.utils.report_generator import create_html_report

# Define paths
sensor_config = Path("configs/small_to_small_direction/sensor_small_to_small_direction.yaml")
sensor_output = Path("derivatives/sensor/20251009_215424-sensor_small_to_small_direction")

# dSPM source outputs
dspm_outputs = [
    Path("derivatives/source/20251009_215537-source_dspm_n1_small_to_small_direction"),
    Path("derivatives/source/20251009_215928-source_dspm_p1_small_to_small_direction"),
    Path("derivatives/source/20251009_220055-source_dspm_p3b_small_to_small_direction"),
]

# eLORETA source outputs
eloreta_outputs = [
    Path("derivatives/source/20251009_220931-source_loreta_n1_small_to_small_direction"),
    Path("derivatives/source/20251009_222106-source_loreta_p1_small_to_small_direction"),
    Path("derivatives/source/20251009_222636-source_loreta_p3b_small_to_small_direction"),
]

# Output report path
report_output = Path("derivatives/reports/20251009_215424-small_to_small_direction_report.html")

# Run command for documentation
run_command = "python regenerate_report_20251009_215424.py"

print("=" * 60)
print("REGENERATING REPORT WITH NEW IMPROVEMENTS")
print("=" * 60)
print(f"Sensor config: {sensor_config}")
print(f"Sensor output: {sensor_output}")
print(f"dSPM outputs: {len(dspm_outputs)} directories")
print(f"eLORETA outputs: {len(eloreta_outputs)} directories")
print(f"Report output: {report_output}")
print("=" * 60)

# Generate the report
create_html_report(
    sensor_config_path=sensor_config,
    sensor_output_dir=sensor_output,
    source_output_dir=dspm_outputs,
    loreta_output_dir=eloreta_outputs,
    report_output_path=report_output,
    run_command=run_command,
    accuracy="acc1",
    data_source="new"
)

print("\n" + "=" * 60)
print("REPORT GENERATION COMPLETE!")
print("=" * 60)
print(f"HTML report: {report_output}")
print(f"PDF report: {report_output.with_suffix('.pdf')}")
print("=" * 60)
