import argparse
import logging
import re
from pathlib import Path
import sys
import yaml

# Add the project root to the path to allow for module imports
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from code.run_sensor_analysis_pipeline import main as run_sensor_analysis
from code.run_source_analysis_pipeline import main as run_source_analysis
from code.utils.report_generator import create_html_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()


def main():
    """
    Orchestrates the full sensor and source analysis pipeline.
    1. Runs sensor analysis.
    2. Checks for significant clusters.
    3. If significant, runs source analysis.
    4. Generates a final HTML report.
    """
    parser = argparse.ArgumentParser(description="Run the full analysis pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the SENSOR-space YAML configuration file.",
    )
    parser.add_argument(
        "--accuracy",
        type=str,
        choices=['all', 'acc1'],
        required=True,
        help="Dataset to use ('all' or 'acc1' for correct trials)."
    )
    args = parser.parse_args()

    # --- Step 1: Run Sensor Analysis ---
    log.info(f"Starting sensor-space analysis for config: {args.config}")
    if "sensor" not in args.config:
        log.error("The provided config file must be a sensor-space config file.")
        return

    # The pipeline scripts return the output directory path
    sensor_output_dir = run_sensor_analysis(config_path=args.config, accuracy=args.accuracy)
    if not sensor_output_dir:
        log.error("Sensor analysis pipeline failed to return an output directory. Aborting.")
        return
    
    log.info(f"Sensor analysis complete. Outputs are in: {sensor_output_dir}")

    # --- Step 2: The "Gatekeeper" - Check for Significance ---
    log.info("Checking for significant clusters at the sensor level...")
    
    # Extract the timestamped name from the sensor output directory path
    timestamped_sensor_name = sensor_output_dir.name
    timestamp_match = re.match(r"(\d{8}_\d{6})-", timestamped_sensor_name)
    timestamp = timestamp_match.group(1) if timestamp_match else "no_timestamp"

    # Derive names from the YAML's analysis_name (authoritative)
    with open(args.config, 'r') as _f:
        _cfg = yaml.safe_load(_f)
    analysis_name = _cfg['analysis_name']
    # For the combined report, use a neutral base (strip leading domain prefixes)
    base_name = analysis_name
    if base_name.startswith('sensor_'):
        base_name = base_name[len('sensor_'):]
    elif base_name.startswith('source_'):
        base_name = base_name[len('source_'):]

    sensor_report_path = sensor_output_dir / f"{analysis_name}_report.txt"

    if not sensor_report_path.exists():
        log.error(f"Could not find sensor report file at: {sensor_report_path}. Aborting.")
        return

    with open(sensor_report_path, 'r') as f:
        report_content = f.read()

    match = re.search(r"Found (\d+) significant cluster\(s\)", report_content)
    
    num_significant_clusters = 0
    if match:
        num_significant_clusters = int(match.group(1))

    # --- Step 3: The Decision ---
    source_output_dir = None
    if num_significant_clusters > 0:
        log.info(f"Found {num_significant_clusters} significant sensor-level cluster(s). Proceeding to source analysis.")
        
        # Derive the source config path
        source_config_path = args.config.replace("sensor", "source")
        
        if not Path(source_config_path).exists():
            log.error(f"Could not find corresponding source config file at: {source_config_path}. Skipping source analysis.")
        else:
            source_output_dir = run_source_analysis(config_path=source_config_path, accuracy=args.accuracy)
            log.info(f"Source analysis complete. Outputs are in: {source_output_dir}")

    else:
        log.info("No significant sensor-level effect found. Skipping source analysis.")

    # --- Step 4: Generate the Final Report ---
    log.info("Generating final analysis report...")

    # Create the dedicated reports directory
    reports_dir = Path("derivatives/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the final report path using the neutral base name and timestamp
    timestamped_base_name = f"{timestamp}-{base_name}"
    report_output_path = reports_dir / f"{timestamped_base_name}_report.html"

    create_html_report(
        sensor_config_path=args.config,
        sensor_output_dir=sensor_output_dir,
        source_output_dir=source_output_dir,  # This will be None if source was skipped
        report_output_path=report_output_path
    )
    log.info(f"Analysis and reporting complete. Final report at: {report_output_path}")


if __name__ == "__main__":
    main()
