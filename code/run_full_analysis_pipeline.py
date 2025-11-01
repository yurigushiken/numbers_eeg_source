import argparse
import logging
import os
import re
from pathlib import Path
import sys
import yaml
from datetime import datetime

from code.utils.logging_utils import setup_run_logger

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
    """Run sensor + source pipeline and generate the combined report."""
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
        help=(
            "Optional override for trial filtering. Use 'all', 'acc1' (correct only), or 'acc0' (incorrect only). "
            "If omitted, per-condition YAML 'accuracy' fields are used (fallback 'all')."
        ),
    )
    args = parser.parse_args()

    try:
        with open(args.config, 'r', encoding='utf-8') as cfg_file:
            config_data = yaml.safe_load(cfg_file) or {}
    except Exception as exc:
        log.error(f"Failed to read config {args.config}: {exc}")
        return

    if "sensor" not in args.config:
        log.error("The provided config file must be a sensor-space config file.")
        return

    analysis_name = config_data.get('analysis_name')
    if not analysis_name:
        log.error("Config missing 'analysis_name'; aborting.")
        return

    base_name = analysis_name
    if base_name.startswith('sensor_'):
        base_name = base_name[len('sensor_'):]
    elif base_name.startswith('source_'):
        base_name = base_name[len('source_'):]

    derivatives_root = Path(os.environ.get("DERIVATIVES_ROOT", "derivatives"))
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = derivatives_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_timestamp}-{analysis_name}.log"
    setup_run_logger(log_path)
    log.info(f"File logging enabled: {log_path}")

    log.info(f"Starting sensor-space analysis for config: {args.config}")
    sensor_output_dir = run_sensor_analysis(config_path=args.config, accuracy=args.accuracy)
    if not sensor_output_dir:
        log.error("Sensor analysis pipeline failed to return an output directory. Aborting.")
        return

    log.info(f"Sensor analysis complete. Outputs are in: {sensor_output_dir}")

    log.info("Checking for significant clusters at the sensor level...")
    timestamped_sensor_name = sensor_output_dir.name
    timestamp_match = re.match(r"(\d{8}_\d{6})-", timestamped_sensor_name)
    timestamp = timestamp_match.group(1) if timestamp_match else "no_timestamp"

    sensor_report_path = sensor_output_dir / f"{analysis_name}_report.txt"
    if not sensor_report_path.exists():
        log.error(f"Could not find sensor report file at: {sensor_report_path}. Aborting.")
        return

    with open(sensor_report_path, 'r', encoding='utf-8') as report_file:
        report_content = report_file.read()

    match = re.search(r"Found (\d+) significant cluster\(s\)", report_content)
    num_significant_clusters = int(match.group(1)) if match else 0

    sensor_config_path = Path(args.config)
    dspm_output_dirs = []
    eloreta_output_dirs = []

    if num_significant_clusters > 0:
        log.info(f"Found {num_significant_clusters} significant sensor-level cluster(s). Proceeding to source analysis.")
        analysis_slug = sensor_config_path.stem
        if analysis_slug.startswith("sensor_"):
            analysis_slug = analysis_slug[len("sensor_"):]
        target_group = config_data.get('report_group', analysis_slug)
        candidate_paths = []
        for candidate in sensor_config_path.parent.glob("*.yaml"):
            if candidate.resolve() == sensor_config_path.resolve():
                continue
            try:
                with open(candidate, 'r', encoding='utf-8') as cf:
                    candidate_cfg = yaml.safe_load(cf) or {}
            except Exception as exc:
                log.warning(f"Failed to load candidate source config {candidate}: {exc}")
                continue

            if (candidate_cfg or {}).get("domain") != "source":
                continue

            candidate_group = candidate_cfg.get('report_group')
            candidate_slug_match = candidate.stem.endswith(f"_{analysis_slug}")
            group_matches = (
                (candidate_group is not None and candidate_group == target_group)
                or (candidate_group is None and candidate_slug_match)
            )
            if not group_matches:
                continue

            order_val = candidate_cfg.get('report_order')
            sort_key = (1, candidate.name.lower())
            if isinstance(order_val, (int, float)):
                sort_key = (0, float(order_val))
            elif isinstance(order_val, str):
                try:
                    sort_key = (0, float(order_val))
                except ValueError:
                    sort_key = (0, order_val.lower())
            candidate_paths.append((sort_key, candidate, candidate_cfg))

        if not candidate_paths:
            log.warning("No companion source-space configs found in directory; skipping source analyses.")
        else:
            for _, candidate, candidate_cfg in sorted(candidate_paths, key=lambda x: x[0]):
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

    log.info("Generating final analysis report...")
    reports_dir = derivatives_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamped_base_name = f"{timestamp}-{base_name}"
    report_output_path = reports_dir / f"{timestamped_base_name}_report.html"

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
        source_output_dir=dspm_output_dirs,
        loreta_output_dir=eloreta_output_dirs,
        report_output_path=report_output_path,
        run_command=run_cmd,
        accuracy=args.accuracy,
        data_source=None,
    )
    log.info(f"Analysis and reporting complete. Final report at: {report_output_path}")



if __name__ == "__main__":
    main()
