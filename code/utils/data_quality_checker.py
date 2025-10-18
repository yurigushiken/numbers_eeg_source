"""
Data Quality Checker for EEG Preprocessing Pipeline

Extracts and validates preprocessing parameters from HAPPE outputs to ensure
accurate reporting and data consistency.

Author: Language and Cognitive Neuroscience Lab, Teachers College, Columbia University
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import pandas as pd
import scipy.io
import numpy as np


@dataclass
class DataQualityInfo:
    """Container for data quality and preprocessing information."""

    preprocessing_name: str
    highpass_hz: float
    lowpass_hz: float
    baseline_corrected: bool
    average_referenced: bool
    n_subjects: int = 0
    mean_channels_retained: float = 0.0
    mean_segments_retained: float = 0.0
    happe_version: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_html(self) -> str:
        """Generate HTML section for inclusion in reports."""
        html = """
        <section class="data-quality-section">
            <h2>Data Preprocessing Information</h2>
            <table class="data-quality-table">
                <tr>
                    <th>Preprocessing Dataset</th>
                    <td>{preprocessing_name}</td>
                </tr>
                <tr>
                    <th>High-pass Filter</th>
                    <td>{highpass_hz} Hz</td>
                </tr>
                <tr>
                    <th>Low-pass Filter</th>
                    <td>{lowpass_hz} Hz</td>
                </tr>
                <tr>
                    <th>Baseline Correction</th>
                    <td>{baseline_status}</td>
                </tr>
                <tr>
                    <th>Reference</th>
                    <td>{reference_status}</td>
                </tr>
                <tr>
                    <th>Number of Subjects</th>
                    <td>{n_subjects}</td>
                </tr>
                <tr>
                    <th>Mean Channels Retained</th>
                    <td>{mean_channels:.1f}</td>
                </tr>
                <tr>
                    <th>Mean Segments Retained (%)</th>
                    <td>{mean_segments:.1f}%</td>
                </tr>
            </table>
        </section>
        """.format(
            preprocessing_name=self.preprocessing_name,
            highpass_hz=self.highpass_hz,
            lowpass_hz=self.lowpass_hz,
            baseline_status="Yes" if self.baseline_corrected else "No",
            reference_status="Average reference" if self.average_referenced else "Unknown",
            n_subjects=self.n_subjects,
            mean_channels=self.mean_channels_retained,
            mean_segments=self.mean_segments_retained
        )

        return html

    def to_text(self) -> str:
        """Generate text section for inclusion in text reports."""
        text = f"""
DATA PREPROCESSING INFORMATION
==============================

Preprocessing Dataset: {self.preprocessing_name}
High-pass Filter: {self.highpass_hz} Hz
Low-pass Filter: {self.lowpass_hz} Hz
Baseline Correction: {'Yes' if self.baseline_corrected else 'No'}
Reference: {'Average reference' if self.average_referenced else 'Unknown'}
Number of Subjects: {self.n_subjects}
Mean Channels Retained: {self.mean_channels_retained:.1f}
Mean Segments Retained: {self.mean_segments_retained:.1f}%
"""
        if self.happe_version:
            text += f"HAPPE Version: {self.happe_version}\n"

        return text


def extract_happe_parameters(happe_dir: Path) -> Optional[Dict]:
    """
    Extract preprocessing parameters from HAPPE output directory.

    Parameters
    ----------
    happe_dir : Path
        Path to HAPPE output directory (e.g., data/data_input_from_happe/hpf_1.0_lpf_35_baseline-on)

    Returns
    -------
    dict or None
        Dictionary with keys: highpass_filter, lowpass_filter, baseline_correction,
        average_reference, happe_version. Returns None if parameters cannot be extracted.
    """
    # Look for input parameters MAT file
    input_params_dir = happe_dir / "input_parameters"

    if not input_params_dir.exists():
        return None

    mat_files = list(input_params_dir.glob("inputParameters_*.mat"))

    if not mat_files:
        return None

    try:
        mat_data = scipy.io.loadmat(str(mat_files[0]))
        params = mat_data['params'][0, 0]

        # Extract filter parameters
        filt = params['filt'][0, 0]
        highpass = float(filt['highpass'][0, 0])
        lowpass = float(filt['lowpass'][0, 0])

        # Extract baseline correction
        baseCorr = params['baseCorr'][0, 0]
        baseline_on = bool(int(baseCorr['on'][0, 0]))

        # Extract re-reference
        reref = params['reref'][0, 0]
        reref_on = bool(int(reref['on'][0, 0]))
        reref_method = str(reref['method'][0]) if reref_on else ""

        # Extract HAPPE version if available
        happe_version = ""
        if 'HAPPEver' in params.dtype.names:
            try:
                happe_version = str(params['HAPPEver'][0])
            except:
                pass

        return {
            'highpass_filter': highpass,
            'lowpass_filter': lowpass,
            'baseline_correction': baseline_on,
            'average_reference': reref_on and reref_method.lower() == 'average',
            'happe_version': happe_version
        }

    except Exception as e:
        print(f"Warning: Could not extract parameters from {mat_files[0]}: {e}")
        return None


def extract_qc_metrics(happe_dir: Path) -> Dict[str, float]:
    """
    Extract quality control metrics from HAPPE QC outputs.

    Parameters
    ----------
    happe_dir : Path
        Path to HAPPE output directory

    Returns
    -------
    dict
        Dictionary with QC metrics (channels_retained, segments_retained, etc.)
    """
    qc_dir = happe_dir / "6 - quality_assessment_outputs"

    metrics = {
        'mean_channels_retained': 0.0,
        'mean_segments_retained': 0.0,
        'n_subjects': 0
    }

    if not qc_dir.exists():
        return metrics

    # Look for data QC CSV
    data_qc_files = list(qc_dir.glob("HAPPE_dataQC_*.csv"))

    if not data_qc_files:
        return metrics

    try:
        qc_df = pd.read_csv(data_qc_files[0])

        # Extract number of subjects
        metrics['n_subjects'] = len(qc_df)

        # Extract channel retention if available (try multiple column name variants)
        channel_cols = [
            'Number_Good_Chans_Selected',
            'Number Good Channels Post-Wav',
            'Number Good Channels Post-Processing'
        ]
        for col in channel_cols:
            if col in qc_df.columns:
                metrics['mean_channels_retained'] = float(qc_df[col].mean())
                break

        # Extract segment retention if available (try multiple column name variants)
        segment_cols = [
            'Percent_Var_Retained_Post-Wav',
            'Percent Var Post-Wav',
            'Percent Good Segments',
            'Percent_Good_Segments'
        ]
        for col in segment_cols:
            if col in qc_df.columns:
                metrics['mean_segments_retained'] = float(qc_df[col].mean())
                break

    except Exception as e:
        print(f"Warning: Could not extract QC metrics from {data_qc_files[0]}: {e}")

    return metrics


def check_data_consistency(
    preprocessing_name: str,
    project_root: Path = None
) -> Tuple[bool, List[str]]:
    """
    Check consistency between preprocessing name and actual parameters.

    Parameters
    ----------
    preprocessing_name : str
        Name of preprocessing variant (e.g., "hpf_1.0_lpf_35_baseline-on")
    project_root : Path, optional
        Project root directory

    Returns
    -------
    tuple
        (is_consistent, list_of_issues)
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    happe_dir = project_root / "data" / "data_input_from_happe" / preprocessing_name

    if not happe_dir.exists():
        return False, [f"HAPPE directory not found: {happe_dir}"]

    params = extract_happe_parameters(happe_dir)

    if params is None:
        return False, ["Could not extract preprocessing parameters"]

    issues = []

    # Parse expected values from name
    hpf_match = re.search(r'hpf[_-](\d+\.?\d*)', preprocessing_name)
    lpf_match = re.search(r'lpf[_-](\d+\.?\d*)', preprocessing_name)
    baseline_match = re.search(r'baseline[_-](on|off)', preprocessing_name)

    if hpf_match:
        expected_hpf = float(hpf_match.group(1))
        if abs(params['highpass_filter'] - expected_hpf) > 0.01:
            issues.append(
                f"HPF mismatch: name says {expected_hpf} Hz, actual is {params['highpass_filter']} Hz"
            )

    if lpf_match:
        expected_lpf = float(lpf_match.group(1))
        if abs(params['lowpass_filter'] - expected_lpf) > 0.01:
            issues.append(
                f"LPF mismatch: name says {expected_lpf} Hz, actual is {params['lowpass_filter']} Hz"
            )

    if baseline_match:
        expected_baseline = baseline_match.group(1) == 'on'
        if params['baseline_correction'] != expected_baseline:
            issues.append(
                f"Baseline mismatch: name says {baseline_match.group(1)}, "
                f"actual is {'on' if params['baseline_correction'] else 'off'}"
            )

    return len(issues) == 0, issues


def generate_data_quality_report(
    preprocessing_name: str,
    project_root: Path = None
) -> Optional[DataQualityInfo]:
    """
    Generate comprehensive data quality report for a preprocessing variant.

    Parameters
    ----------
    preprocessing_name : str
        Name of preprocessing variant (from common.yaml or config)
    project_root : Path, optional
        Project root directory

    Returns
    -------
    DataQualityInfo or None
        Data quality information object, or None if unavailable
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    happe_dir = project_root / "data" / "data_input_from_happe" / preprocessing_name

    if not happe_dir.exists():
        print(f"Warning: HAPPE directory not found: {happe_dir}")
        return None

    # Extract preprocessing parameters
    params = extract_happe_parameters(happe_dir)

    if params is None:
        print(f"Warning: Could not extract parameters for {preprocessing_name}")
        return None

    # Extract QC metrics
    qc_metrics = extract_qc_metrics(happe_dir)

    # Create report object
    report = DataQualityInfo(
        preprocessing_name=preprocessing_name,
        highpass_hz=params['highpass_filter'],
        lowpass_hz=params['lowpass_filter'],
        baseline_corrected=params['baseline_correction'],
        average_referenced=params['average_reference'],
        n_subjects=qc_metrics['n_subjects'],
        mean_channels_retained=qc_metrics['mean_channels_retained'],
        mean_segments_retained=qc_metrics['mean_segments_retained'],
        happe_version=params.get('happe_version', '')
    )

    return report


def get_preprocessing_info_from_config(
    config_path: Path = None,
    project_root: Path = None,
    data_source: Optional[str] = None,
) -> Optional[DataQualityInfo]:
    """
    Extract preprocessing info based on common.yaml configuration.

    Parameters
    ----------
    config_path : Path, optional
        Path to common.yaml (defaults to configs/common.yaml)
    project_root : Path, optional
        Project root directory
    data_source : str, optional
        Optional custom combined data path. Legacy 'old' split paths are not supported.

    Returns
    -------
    DataQualityInfo or None
        Data quality information from the configured preprocessing variant
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    if config_path is None:
        config_path = project_root / "configs" / "common.yaml"

    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        preprocessing_name = (config.get('data') or {}).get('preprocessing')

        # Optional runtime override if a custom combined path was used
        if data_source:
            if data_source == "old":
                raise ValueError("Legacy 'old' data pipeline is no longer supported.")
            ds_path = Path(data_source)
            if not ds_path.is_absolute():
                ds_path = (project_root / ds_path).resolve()
            if ds_path.is_file():
                ds_path = ds_path.parent
            candidate = ds_path.name
            if candidate:
                preprocessing_name = candidate

        if not preprocessing_name:
            print("Warning: No preprocessing name found in common.yaml")
            return None

        return generate_data_quality_report(preprocessing_name, project_root=project_root)

    except Exception as e:
        print(f"Error reading config: {e}")
        return None
