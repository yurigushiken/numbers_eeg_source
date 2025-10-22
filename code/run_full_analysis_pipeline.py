import argparse
import logging
import os
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
        choices=['all', 'acc1', 'acc0'],
        required=False,
        default=None,
        help=("Optional override for trial filtering. Use 'all', 'acc1' (correct only), or 'acc0' (incorrect only). "
              "If omitted, per-condition YAML 'accuracy' fields are used (fallback 'all').")
    )
    # Data source flag removed; combined preprocessed data is the standard
    args = parser.parse_args()

    # --- Step 1: Run Sensor Analysis ---
    log.info(f"Starting sensor-space analysis for config: {args.config}")
    # No data-source distinction; combined preprocessed data is standard
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
    sensor_config_path = Path(args.config)
    # Collect all source outputs per method to embed multiple sections (e.g., N1 and P3b)
    dspm_output_dirs = []
    eloreta_output_dirs = []
    if num_significant_clusters > 0:
        log.info(f"Found {num_significant_clusters} significant sensor-level cluster(s). Proceeding to source analysis.")

        # Identify matching source-domain configs that share the analysis slug
        analysis_slug = sensor_config_path.stem
        if analysis_slug.startswith("sensor_"):
            analysis_slug = analysis_slug[len("sensor_"):]
        candidate_paths = sorted(sensor_config_path.parent.glob(f"*_{analysis_slug}.yaml"))
        if not candidate_paths:
            log.warning("No companion source-space configs found in directory; skipping source analyses.")
        else:
            for candidate in candidate_paths:
                if candidate.resolve() == sensor_config_path.resolve():
                    continue
                try:
                    with open(candidate, 'r') as cf:
                        candidate_cfg = yaml.safe_load(cf)
                except Exception as exc:
                    log.warning(f"Failed to load candidate source config {candidate}: {exc}")
                    continue

                if (candidate_cfg or {}).get("domain") != "source":
                    continue

                method = (candidate_cfg.get("source") or {}).get("method", "unknown")
                log.info(f"--- Running source analysis ({method}) for {candidate} ---")
                output_dir = run_source_analysis(
                    config_path=str(candidate),
                    accuracy=args.accuracy,
                )
                log.info(f"Source analysis ({method}) complete. Outputs are in: {output_dir}")

                method_lower = str(method).lower()
                if method_lower == "dspm":
                    dspm_output_dirs.append(output_dir)
                elif method_lower == "eloreta":
                    eloreta_output_dirs.append(output_dir)
                else:
                    log.info(f"Recorded outputs for additional method '{method}' at {output_dir}")

    else:
        log.info("No significant sensor-level effect found. Skipping all source analyses.")

    # --- Step 4: Generate the Final Report ---
    log.info("Generating final analysis report...")

    # Create the dedicated reports directory
    # Allow custom derivatives root via environment variable (default: "derivatives")
    derivatives_root = os.environ.get("DERIVATIVES_ROOT", "derivatives")
    reports_dir = Path(derivatives_root) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the final report path using the neutral base name and timestamp
    timestamped_base_name = f"{timestamp}-{base_name}"
    report_output_path = reports_dir / f"{timestamped_base_name}_report.html"

    # Build a reproducible command string for the report
    if args.accuracy is not None:
        run_cmd = (
            f"python -m code.run_full_analysis_pipeline --config {args.config} "
            f"--accuracy {args.accuracy}"
        )
    else:
        run_cmd = f"python -m code.run_full_analysis_pipeline --config {args.config}"

    create_html_report(
        sensor_config_path=args.config,
        sensor_output_dir=sensor_output_dir,
        source_output_dir=dspm_output_dirs,   # List of dSPM runs (e.g., N1 + P3b)
        loreta_output_dir=eloreta_output_dirs, # List of eLORETA runs
        report_output_path=report_output_path,
        run_command=run_cmd,
        accuracy=args.accuracy,
        data_source=None
    )
    log.info(f"Analysis and reporting complete. Final report at: {report_output_path}")


if __name__ == "__main__":
    main()
