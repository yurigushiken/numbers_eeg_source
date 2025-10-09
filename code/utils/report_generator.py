import yaml
from pathlib import Path
import logging
from datetime import datetime
import os
import pandas as pd
import shutil
import subprocess
import json as _json
from code.utils.caption_generator import (
    generate_sensor_caption,
    generate_source_caption,
    infer_time_window_label,
    infer_topography_from_channels
)
from code.utils.data_quality_checker import get_preprocessing_info_from_config

log = logging.getLogger()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 20px auto; padding: 0 20px; font-size: 14px; }}
        header {{ text-align: center; border-bottom: 2px solid #eee; padding-bottom: 16px; margin-bottom: 24px; }}
        h1 {{ color: #111; font-size: 26px; margin-bottom: 4px; }}
        h2 {{ color: #333; border-bottom: 1px solid #eee; padding-bottom: 8px; margin-top: 28px; font-size: 18px; }}
        h3 {{ color: #333; font-size: 16px; margin-top: 18px; }}
        p, li {{ color: #555; font-size: 14px; }}
        .meta-info {{ color: #777; font-size: 0.9em; }}
        .summary, .findings {{ background-color: #f9f9f9; border: 1px solid #eee; padding: 16px; border-radius: 5px; margin-bottom: 16px; }}
        .figure-container {{ text-align: center; margin: 20px 0; }}
        .figure-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: center; }}
        .figure-grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; align-items: start; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        figcaption {{ font-size: 12px; color: #666; margin-top: 10px; }}
        .null-result {{ font-style: italic; color: #888; }}
        footer {{ text-align: center; margin-top: 32px; padding-top: 16px; border-top: 2px solid #eee; font-size: 12px; color: #999; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }}
        th, td {{ padding: 8px 12px; border: 1px solid #ddd; text-align: left; vertical-align: top; }}
        th {{ background-color: #f2f2f2; font-size: 13px; }}
        .data-quality-table {{ max-width: 600px; margin: 0 auto; }}
        .data-quality-table th {{ width: 50%; }}
        .data-quality-table td {{ width: 50%; }}
        pre, code {{ white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere; }}
        pre {{ background-color: #fafafa; padding: 12px; border-radius: 4px; overflow-x: auto; font-size: 12px; }}
    </style>
</head>
<body>
    <header>
        <h1>{title}</h1>
        <div class="meta-info">
            <p><strong>Lab:</strong> Language and Cognitive Neuroscience Lab</p>
            <p><strong>Institution:</strong> Teachers College, Columbia University</p>
            <p><strong>Date:</strong> {date}</p>
        </div>
    </header>

    <main>
        <section id="summary">
            <h2>Brief analysis description</h2>
            <div class="summary">
                <p>This report shows the sensor- and source-space analysis of the contrast: <strong>{contrast_name}</strong>. The goal is to identify the spatiotemporal characteristics of the neural response differences between these conditions.</p>
            </div>
        </section>

        <section id="run-details">
            <h2>Run Details</h2>
            {run_details_section}
        </section>

        {data_quality_section}

        <section id="analysis-parameters">
            <h2>Analysis Parameters</h2>
            {analysis_parameters_section}
        </section>

        <section id="condition-erps">
            <h2>Condition ERPs (Canonical ROIs)</h2>
            {sensor_roi_erp_section}
        </section>

        <section id="methods-summary">
            <h2>Methods Summary</h2>
            <p>{methods_summary_paragraph}</p>
        </section>

        <section id="sensor-results">
            <h2>Sensor-Space Results</h2>
            <div class="findings">
                <h3>Statistical Findings</h3>
                <pre><code>{sensor_stats}</code></pre>
            </div>
            <div class="figure-container">
                <div class="figure-grid">
                    <div>
                        <img src="{erp_plot_path}" alt="ERP Cluster Plot">
                    </div>
                    <div>
                        <img src="{topo_plot_path}" alt="Topomap Cluster Plot">
                    </div>
                </div>
                <figcaption><strong>Figure 1.</strong>{sensor_main_caption_detail}</figcaption>
            </div>
            {sensor_extra_clusters_section}
        </section>

        {dSPM_section}
        
        {eloreta_section}

    </main>

    <footer>
        <p>EEG Analysis Pipeline</p>
    </footer>
</body>
</html>
"""
SENSOR_ROI_ERP_SECTION_TEMPLATE = """
<div class=\"figure-container\">
  <div class=\"figure-grid-3\">
    <div><img src=\"{roi_p1_path}\" alt=\"P1 ROI Condition ERPs\"></div>
    <div><img src=\"{roi_n1_path}\" alt=\"N1 ROI Condition ERPs\"></div>
    <div><img src=\"{roi_p3b_path}\" alt=\"P3b ROI Condition ERPs\"></div>
  </div>
  <figcaption><strong>Figure 0.</strong> Condition ERPs over canonical ROIs: P1 (Oz), N1 (bilateral), P3b (midline).</figcaption>
  <br/>
</div>
"""

def _build_run_details_section(run_command: str | None, accuracy: str | None, data_source: str | None, sensor_config_path: Path, report_dir: Path) -> str:
    rows = []
    if run_command:
        rows.append(("Command Invoked", run_command))
    if accuracy:
        rows.append(("Accuracy", accuracy))
    if data_source:
        rows.append(("Data Source", data_source))
    try:
        cfg_rel = os.path.relpath(sensor_config_path.resolve(), start=report_dir)
    except Exception:
        cfg_rel = str(sensor_config_path)
    rows.append(("Sensor Config", cfg_rel))

    html = ["<div class=\"summary\"><table>", "<tbody>"]
    for k, v in rows:
        html.append(f"<tr><th style='width:180px'>{k}</th><td>{v}</td></tr>")
    html.extend(["</tbody>", "</table>", "</div>"])
    return "\n".join(html)


SOURCE_SECTION_TEMPLATE = """
<div class="findings">
    <h3>Statistical Findings (Source Space)</h3>
    <pre><code>{source_stats}</code></pre>
</div>
{anatomical_table}
<div class="figure-container">
    <img src="{source_plot_path}" alt="Source Cluster Plot">
    <figcaption><strong>Figure 2.</strong> Source localization of the significant cluster, demonstrating the estimated anatomical origin of the sensor-level effect.</figcaption>
</div>
"""

NULL_SOURCE_SECTION_TEMPLATE = """
<p class="null-result">Source statistics were computed for the specified window, but no clusters survived multiple-comparisons correction. No source plot is shown.</p>
{label_ts_section}
"""

LABEL_TS_SECTION_TEMPLATE = """
<div class="findings">
  <h3>Auxiliary ROI Time-Series Result</h3>
  <pre><code>{label_ts_summary}</code></pre>
</div>
"""

SOURCE_SECTION_STATS_ONLY_TEMPLATE = """
<div class="findings">
    <h3>Statistical Findings (Source Space)</h3>
    <pre><code>{source_stats}</code></pre>
</div>
{anatomical_table}
{label_ts_section}
"""

def _format_seconds(value) -> str:
    """Format seconds with up to 3 decimals, stripping trailing zeros."""
    try:
        return f"{float(value):.3f}".rstrip('0').rstrip('.')
    except Exception:
        return str(value)

def _format_tail_descriptor(tail_value: int) -> str:
    """Return a human-readable tail descriptor for display."""
    if tail_value == 0:
        return "two-tailed"
    if tail_value == 1:
        return "one-tailed, positive"
    if tail_value == -1:
        return "one-tailed, negative"
    return "two-tailed"

def _build_analysis_parameters_table(sensor_config_path: Path, sensor_config: dict, report_dir: Path) -> str:
    """Construct an HTML table summarizing key analysis parameters from sensor and source YAMLs."""
    # Sensor parameters
    sensor_tmin = sensor_config.get('tmin')
    sensor_tmax = sensor_config.get('tmax')
    sensor_baseline = sensor_config.get('baseline')
    sensor_contrast_name = sensor_config.get('contrast', {}).get('name')
    s_stats = sensor_config.get('stats', {}) or {}
    s_p = s_stats.get('p_threshold')
    s_alpha = s_stats.get('cluster_alpha')
    s_nperm = s_stats.get('n_permutations')
    s_tail = _format_tail_descriptor(int(s_stats.get('tail', 0)))

    # Discover matching source-space configs in the same directory
    source_entries: list[dict] = []
    try:
        analysis_slug = sensor_config_path.stem
        if analysis_slug.startswith("sensor_"):
            analysis_slug = analysis_slug[len("sensor_"):]
        candidate_paths = sorted(sensor_config_path.parent.glob(f"*_{analysis_slug}.yaml"))
        for candidate in candidate_paths:
            if candidate.resolve() == sensor_config_path.resolve():
                continue
            with open(candidate, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            if cfg.get("domain") != "source":
                continue
            src_epoch = cfg.get("epoch_window") or {}
            src_stats = cfg.get("stats") or {}
            roi = (src_stats.get("roi") or {}).get("labels")
            if isinstance(roi, list):
                roi_str = ", ".join(str(x) for x in roi)
            elif isinstance(roi, str):
                roi_str = roi
            else:
                roi_str = None
            source_entries.append({
                "method": (cfg.get("source") or {}).get("method"),
                "snr": (cfg.get("source") or {}).get("snr"),
                "analysis_window": src_stats.get("analysis_window"),
                "p_threshold": src_stats.get("p_threshold"),
                "cluster_alpha": src_stats.get("cluster_alpha"),
                "n_permutations": src_stats.get("n_permutations"),
                "tail": _format_tail_descriptor(int(src_stats.get("tail", 0))) if "tail" in src_stats else None,
                "roi": roi_str,
                "tmin": src_epoch.get("tmin"),
                "tmax": src_epoch.get("tmax")
            })
    except Exception as exc:
        log.warning(f"Failed to gather source configuration details: {exc}")
        source_entries = []

    # Build rows
    rows = []
    if sensor_contrast_name:
        rows.append(("Data", "Contrast", str(sensor_contrast_name)))
    if sensor_tmin is not None and sensor_tmax is not None:
        rows.append(("Time Window (Sensor)", "Epoch Window", f"{_format_seconds(sensor_tmin)} to {_format_seconds(sensor_tmax)} s"))
    if isinstance(sensor_baseline, (list, tuple)) and len(sensor_baseline) == 2:
        rows.append(("Baseline Period", "Baseline", f"{_format_seconds(sensor_baseline[0])} to {_format_seconds(sensor_baseline[1])} s"))
    # Sensor stats
    rows.append(("Sensor Statistics", "Test Type", "Spatio-temporal Cluster Permutation"))
    if s_p is not None:
        rows.append(("Sensor Statistics", "Cluster-forming Threshold (p)", f"{s_p} ({s_tail})"))
    if s_alpha is not None:
        rows.append(("Sensor Statistics", "Cluster Significance (alpha)", f"{s_alpha}"))
    if s_nperm is not None:
        rows.append(("Sensor Statistics", "Permutations", f"{s_nperm}"))

    for entry in source_entries:
        method = entry.get("method")
        label_prefix = "Source Localization"
        if method:
            label_prefix += f" ({method})"
        if method:
            rows.append((label_prefix, "Method", str(method)))
        snr = entry.get("snr")
        if snr is not None:
            rows.append((label_prefix, "SNR Estimate", str(snr)))
        rows.append((label_prefix.replace("Localization", "Statistics"), "Test Type", "Spatio-temporal Cluster Permutation"))
        tmin, tmax = entry.get("tmin"), entry.get("tmax")
        if tmin is not None and tmax is not None:
            rows.append((label_prefix.replace("Localization", "Statistics"), "Epoch Window", f"{_format_seconds(tmin)} to {_format_seconds(tmax)} s"))
        analysis_window = entry.get("analysis_window")
        if isinstance(analysis_window, (list, tuple)) and len(analysis_window) == 2:
            rows.append((label_prefix.replace("Localization", "Statistics"), "Analysis Window", f"{_format_seconds(analysis_window[0])} to {_format_seconds(analysis_window[1])} s"))
        roi = entry.get("roi")
        if roi:
            rows.append((label_prefix.replace("Localization", "Statistics"), "Region of Interest (ROI)", roi))
        p_thr = entry.get("p_threshold")
        if p_thr is not None:
            tail = entry.get("tail") or ""
            rows.append((label_prefix.replace("Localization", "Statistics"), "Cluster-forming Threshold (p)", f"{p_thr} ({tail})".strip()))
        cluster_alpha = entry.get("cluster_alpha")
        if cluster_alpha is not None:
            rows.append((label_prefix.replace("Localization", "Statistics"), "Cluster Significance (alpha)", str(cluster_alpha)))
        n_perm = entry.get("n_permutations")
        if n_perm is not None:
            rows.append((label_prefix.replace("Localization", "Statistics"), "Permutations", str(n_perm)))

    # Render table
    html = [
        "<table>",
        "  <thead>",
        "    <tr><th>Parameter Domain</th><th>Parameter Name</th><th>Value</th></tr>",
        "  </thead>",
        "  <tbody>"
    ]
    for domain, name, value in rows:
        html.append(f"    <tr><td>{domain}</td><td>{name}</td><td>{value}</td></tr>")
    html.extend(["  </tbody>", "</table>"])
    return "\n".join(html)

def read_report_file(report_path):
    """Reads the content of a stats report file."""
    if report_path and report_path.exists():
        with open(report_path, 'r') as f:
            # Skip the header to get to the results
            content = f.read()
            results_marker = "RESULTS"
            results_pos = content.find(results_marker)
            if results_pos != -1:
                return content[results_pos:]
    return "Report file not found."


def _generate_anatomical_table_html(anatomical_report_path, top_n=7):
    """Generates an HTML table from the anatomical report CSV."""
    if not anatomical_report_path or not anatomical_report_path.exists():
        return ""
    
    try:
        df = pd.read_csv(anatomical_report_path)
        if df.empty:
            return ""

        # FIX: Convert contribution column to a numeric type for sorting.
        # The original CSV stores this as a string (e.g., "8.5%"), which
        # prevents the use of numerical methods like nlargest().
        df['Contribution_numeric'] = pd.to_numeric(
            df['Region Contribution (%)'].astype(str).str.rstrip('%'),
            errors='coerce'
        ).fillna(0)

        html = "<h3>Anatomical Localization Summary</h3>"
        
        for cluster_id in sorted(df['Cluster ID'].unique()):
            cluster_df = df[df['Cluster ID'] == cluster_id]
            
            # Use the new numeric column to find the top N rows
            top_df = cluster_df.nlargest(top_n, 'Contribution_numeric')

            if top_df.empty:
                continue

            # p-value is already a formatted string in the CSV
            p_value = top_df['p-value'].iloc[0]
            peak_mni = top_df['Peak Activation MNI (mm)'].iloc[0]

            html += f"<h4>Cluster #{cluster_id} (p={p_value}, Peak MNI: {peak_mni})</h4>"
            
            # Select, rename, and format columns for the final display table
            table_df = top_df[['Anatomical Region', 'Contribution_numeric']].copy()
            table_df.rename(columns={
                'Anatomical Region': 'Region', 
                'Contribution_numeric': 'Contribution (%)'
            }, inplace=True)
            
            table_df['Contribution (%)'] = table_df['Contribution (%)'].map('{:.1f}%'.format)
            
            html += table_df.to_html(index=False, classes='anatomical-table')

        return html
    except Exception as e:
        log.error(f"Failed to generate anatomical table: {e}")
        return "<p><em>Error generating anatomical summary table.</em></p>"


def _build_source_section_html(
    source_output_dir: Path,
    analysis_name_base: str,
    report_dir: Path,
    section_title: str,
    figure_number_prefix: str,
    contrast_name: str = "",
    method: str = "dSPM",
    analysis_window: str = ""
) -> str:
    """Helper function to build the HTML for a generic source analysis section (dSPM or eLORETA)."""
    if not source_output_dir or not source_output_dir.exists():
        return f"""
        <section>
            <h2>{section_title}</h2>
            <p class="null-result">{analysis_name_base} analysis was not run or produced no output.</p>
        </section>
        """

    source_report_path = source_output_dir / f"{analysis_name_base}_report.txt"
    source_plot_path = source_output_dir / f"{analysis_name_base}_source_cluster.png"
    anatomical_report_path = source_output_dir / f"{analysis_name_base}_anatomical_report.csv"
    hs_summary_path = source_output_dir / f"{analysis_name_base}_anatomical_summary_hs.csv"
    label_ts_summary_path = source_output_dir / "aux" / "label_cluster_summary.txt"

    source_stats = read_report_file(source_report_path) if source_report_path.exists() else "No significant clusters found."
    cluster_details = _parse_source_report_for_cluster_details(source_report_path, source_output_dir)

    # --- Build each component separately ---

    # 1. Statistical Findings
    statistical_findings_html = f"""
        <div class="findings">
            <h3>Statistical Findings</h3>
            <pre><code>{source_stats}</code></pre>
        </div>
    """

    # 2. Inverse provenance summary & table (if present)
    inverse_block_html = ""
    try:
        prov_path = Path(source_output_dir) / "inverse_provenance.json"
        if prov_path.exists():
            rows = _json.loads(prov_path.read_text())
            pre = sum(1 for r in rows if r.get('used') == 'precomputed')
            tpl = sum(1 for r in rows if r.get('used') == 'template')
            try:
                prov_rel = os.path.relpath(prov_path.resolve(), start=report_dir)
            except Exception:
                prov_rel = str(prov_path)
            inverse_block_html = f"""
                <div class=\"findings\">
                    <h3>Inverse Operators</h3>
                    <p>Precomputed: {pre}, Template: {tpl}. Full details: <code>{prov_rel}</code></p>
                </div>
            """
    except Exception:
        inverse_block_html = ""

    # 3. Cortical Cluster Localization Summary (HS Table)
    hs_summary_html = ""
    if hs_summary_path.exists():
        try:
            hs_df = pd.read_csv(hs_summary_path)
            if not hs_df.empty:
                hs_summary_html = (
                    "<h3>Cortical Cluster Localization Summary</h3>"
                    + hs_df.to_html(index=False, classes='anatomical-table')
                )
        except Exception:
            pass
    
    # 4. Detailed Anatomical Localization Summary
    anatomical_table_html = _generate_anatomical_table_html(anatomical_report_path)

    # 5. Plots (Main and additional)
    plots_html = ""
    if source_plot_path.exists():
        rel_path = os.path.relpath(source_plot_path.resolve(), start=report_dir)
        caption_full = "T-Values for Cluster #1."
        if 1 in cluster_details:
            info = cluster_details[1]
            # Infer time window label from analysis window
            time_window_label = infer_time_window_label(analysis_window) if analysis_window else None

            try:
                # Generate enhanced source caption
                caption_full = generate_source_caption(
                    cluster_id=1,
                    cluster_info=info,
                    contrast_name=contrast_name if contrast_name else "Condition A vs. Condition B",
                    method=method,
                    time_window_label=time_window_label,
                    analysis_window=analysis_window
                )
            except Exception as e:
                # Fallback to simple caption
                log.warning(f"Failed to generate enhanced source caption: {e}")
                caption_full = f"T-Values for Cluster #1 (p={info['p_value']:.4f}, peak t={info['peak_t']:.2f}, {info['n_vertices']} vertices)."

        plots_html += f"""
        <div class="figure-container">
            <img src="{rel_path}" alt="Source Plot for {analysis_name_base}">
            <figcaption><strong>Figure {figure_number_prefix}.</strong> {caption_full}</figcaption>
        </div>
        """
    else:
         label_ts_summary = label_ts_summary_path.read_text() if label_ts_summary_path.exists() else ""
         label_ts_section = LABEL_TS_SECTION_TEMPLATE.format(label_ts_summary=label_ts_summary) if label_ts_summary else ""
         plots_html += NULL_SOURCE_SECTION_TEMPLATE.format(label_ts_section=label_ts_section)

    try:
        import re as _re
        for plot_file in sorted(source_output_dir.glob(f"{analysis_name_base}_source_cluster_*.png")):
            m = _re.search(r"_cluster_(\d+)\.png$", plot_file.name)
            if not m: continue
            rank = int(m.group(1))
            if rank <= 1: continue

            rel_path = os.path.relpath(plot_file.resolve(), start=report_dir)
            caption_detail = ""
            if rank in cluster_details:
                info = cluster_details[rank]
                caption_detail = f" (p={info['p_value']:.4f}, peak t={info['peak_t']:.2f}, {info['n_vertices']} vertices)"
            
            plots_html += f"""
            <div class="figure-container">
                <img src="{rel_path}" alt="Source Cluster {rank}">
                <figcaption><strong>Figure {figure_number_prefix}.{rank-1}. Additional significant source cluster #{rank}{caption_detail}.</strong></figcaption>
            </div>
            """
    except Exception:
        pass

    # --- Assemble the final section in the correct order ---
    return f"""
    <section>
        <h2>{section_title}</h2>
        {statistical_findings_html}
        {inverse_block_html}
        {hs_summary_html}
        {anatomical_table_html}
        {plots_html}
    </section>
    """


def _parse_source_report_for_cluster_details(report_path: Path, source_output_dir: Path = None) -> dict:
    """Parses the source stats report to extract key details for each cluster, including anatomical info."""
    details = {}
    if not report_path or not report_path.exists():
        return details

    try:
        text = report_path.read_text()
        # Use regex to find all cluster blocks
        import re
        pattern = re.compile(
            r"Cluster #(\d+)\s*\(p-value = ([\d.]+)\).*?"
            r"Peak t-value: ([-\d.]+).*?"
            r"Number of vertices: (\d+)",
            re.DOTALL
        )
        matches = pattern.findall(text)
        for match in matches:
            cluster_id = int(match[0])
            details[cluster_id] = {
                "p_value": float(match[1]),
                "peak_t": float(match[2]),
                "n_vertices": int(match[3]),
                "peak_mni": "",
                "primary_region": ""
            }

        # Try to augment with anatomical info from HS summary CSV
        if source_output_dir:
            try:
                analysis_name = source_output_dir.name.split("-", 1)[1] if "-" in source_output_dir.name else source_output_dir.name
                hs_path = source_output_dir / f"{analysis_name}_anatomical_summary_hs.csv"
                if hs_path.exists():
                    import pandas as pd
                    hs_df = pd.read_csv(hs_path)
                    for _, row in hs_df.iterrows():
                        cluster_id = int(row['Cluster ID'])
                        if cluster_id in details:
                            details[cluster_id]['peak_mni'] = str(row['Peak MNI (mm)'])
                            # Extract first region from "Top regions"
                            top_regions = str(row.get('Top regions', ''))
                            if top_regions and top_regions != 'nan':
                                first_region = top_regions.split(',')[0].strip()
                                details[cluster_id]['primary_region'] = first_region
            except Exception as e:
                log.debug(f"Could not parse anatomical summary for enhanced source captions: {e}")

    except Exception as e:
        log.warning(f"Could not parse source report for cluster details: {e}")
    return details


def _parse_sensor_report_for_cluster_details(report_path: Path) -> dict:
    """Parses the sensor stats report to extract key details for each cluster."""
    details = {}
    if not report_path or not report_path.exists():
        return details

    try:
        text = report_path.read_text()
        import re
        # Pattern to extract cluster details including channels
        pattern = re.compile(
            r"Cluster #(\d+)\s*\(p-value = ([\d.]+)\).*?"
            r"Peak t-value: ([-\d.]+).*?"
            r"Time window: ([\d.]+\s*ms to [\d.]+\s*ms).*?"
            r"Number of channels: (\d+).*?"
            r"Channels involved: ([^\n]+)",
            re.DOTALL
        )
        matches = pattern.findall(text)
        for match in matches:
            cluster_id = int(match[0])
            # Parse channel list
            channels_str = match[5].strip()
            channels = [ch.strip() for ch in channels_str.split(',') if ch.strip()]

            details[cluster_id] = {
                "p_value": float(match[1]),
                "peak_t": float(match[2]),
                "time_window": str(match[3]),
                "n_channels": int(match[4]),
                "channels": channels
            }
    except Exception as e:
        log.warning(f"Could not parse sensor report for cluster details: {e}")
    return details


def create_html_report(sensor_config_path, sensor_output_dir, source_output_dir, loreta_output_dir, report_output_path, run_command: str | None = None, accuracy: str | None = None, data_source: str | None = None):
    """
    Generates a standalone HTML report from the analysis outputs.
    """
    sensor_config_path = Path(sensor_config_path)
    sensor_output_dir = Path(sensor_output_dir)
    report_output_path = Path(report_output_path)
    report_dir = report_output_path.parent.resolve()
    
    # --- 1. Gather Data and Paths ---
    with open(sensor_config_path, 'r') as f:
        config = yaml.safe_load(f)

    analysis_name = config['analysis_name']
    # Neutral base name for combined report title (strip leading domain prefixes)
    base_name = analysis_name
    if base_name.startswith('sensor_'):
        base_name = base_name[len('sensor_'):]
    elif base_name.startswith('source_'):
        base_name = base_name[len('source_'):]
    base_title = base_name.replace("_", " ").title()
    
    # Sensor paths
    sensor_report_path = sensor_output_dir / f"{analysis_name}_report.txt"
    erp_plot_path = sensor_output_dir / f"{analysis_name}_erp_cluster.png"
    topo_plot_path = sensor_output_dir / f"{analysis_name}_topomap_cluster.png"
    # ROI condition ERP individual images
    roi_p1_path = sensor_output_dir / f"{analysis_name}_roi_P1_erp.png"
    roi_n1_path = sensor_output_dir / f"{analysis_name}_roi_N1_erp.png"
    roi_p3b_path = sensor_output_dir / f"{analysis_name}_roi_P3b_erp.png"
    
    # Read sensor stats
    sensor_stats = read_report_file(sensor_report_path)
    sensor_cluster_details = _parse_sensor_report_for_cluster_details(sensor_report_path)

    # Discover additional sensor cluster figure pairs (rank >= 2)
    sensor_extra_clusters_section = ""
    try:
        import re as _re
        extra_pairs = []
        for erp_img in sorted(sensor_output_dir.glob(f"{analysis_name}_erp_cluster_*.png")):
            m = _re.search(r"_cluster_(\d+)\.png$", erp_img.name)
            if not m:
                continue
            rank = int(m.group(1))
            if rank <= 1:
                continue
            topo_img = sensor_output_dir / f"{analysis_name}_topomap_cluster_{rank}.png"
            if topo_img.exists():
                extra_pairs.append((rank, erp_img, topo_img))

        if extra_pairs:
            blocks = []
            for rank, erp_img, topo_img in sorted(extra_pairs, key=lambda x: x[0]):
                erp_rel = os.path.relpath(erp_img.resolve(), start=report_dir)
                topo_rel = os.path.relpath(topo_img.resolve(), start=report_dir)

                # Generate enhanced caption for extra sensor clusters
                extra_caption_full = f"Additional significant sensor cluster #{rank}."
                if rank in sensor_cluster_details:
                    info = sensor_cluster_details[rank]
                    # Infer topography from channels if available
                    channels = info.get('channels', [])
                    topography = infer_topography_from_channels(channels) if channels else ""
                    # Infer time window label
                    time_window_label = infer_time_window_label(info.get('time_window', ''))
                    # Add topography to info dict for caption generation
                    info_with_topo = {**info, 'topography': topography}

                    try:
                        extra_caption_full = generate_sensor_caption(
                            cluster_id=rank,
                            cluster_info=info_with_topo,
                            contrast_name=config['contrast']['name'],
                            time_window_label=time_window_label
                        )
                    except Exception:
                        # Fallback to simple caption
                        extra_caption_full = f"Additional significant sensor cluster #{rank} (p={info['p_value']:.4f}, peak t={info['peak_t']:.2f}, {info['n_channels']} channels, {info['time_window']})."

                blocks.append(
                    f"""
                    <div class="figure-container">
                        <div class="figure-grid">
                            <div><img src="{erp_rel}" alt="ERP Cluster {rank}"></div>
                            <div><img src="{topo_rel}" alt="Topomap Cluster {rank}"></div>
                        </div>
                        <figcaption><strong>Figure 1.{rank-1}.</strong> {extra_caption_full}</figcaption>
                    </div>
                    """
                )
            sensor_extra_clusters_section = "\n".join(blocks)
    except Exception:
        sensor_extra_clusters_section = ""

    # --- Build dSPM Section(s) ---
    source_section_html = ""
    sods = source_output_dir
    if sods and not isinstance(sods, (list, tuple)):
        sods = [sods]
    if sods:
        for sod in sods:
            try:
                dSPM_analysis_name = Path(sod).name.split("-", 1)[1]
            except Exception:
                dSPM_analysis_name = analysis_name.replace("sensor", "source")
            source_section_html += _build_source_section_html(
                source_output_dir=Path(sod),
                analysis_name_base=dSPM_analysis_name,
                report_dir=report_dir,
                section_title="Source-Space Localization (dSPM)",
                figure_number_prefix="2",
                contrast_name=config['contrast']['name'],
                method="dSPM",
                analysis_window=config.get('stats', {}).get('analysis_window', '')
            )
    
    # --- Build eLORETA Section(s) ---
    eloreta_section_html = ""
    lods = loreta_output_dir
    if lods and not isinstance(lods, (list, tuple)):
        lods = [lods]
    if lods:
        for lod in lods:
            try:
                loreta_analysis_name = Path(lod).name.split("-", 1)[1]
            except Exception:
                loreta_analysis_name = analysis_name.replace("sensor", "loreta")
            eloreta_section_html += _build_source_section_html(
                source_output_dir=Path(lod),
                analysis_name_base=loreta_analysis_name,
                report_dir=report_dir,
                section_title="Source-Space Localization (eLORETA)",
                figure_number_prefix="3",
                contrast_name=config['contrast']['name'],
                method="eLORETA",
                analysis_window=config.get('stats', {}).get('analysis_window', '')
            )

    # --- Populate Template ---
    # Calculate relative paths for sensor images from the new report location
    erp_plot_rel_path = os.path.relpath(erp_plot_path.resolve(), start=report_dir)
    topo_plot_rel_path = os.path.relpath(topo_plot_path.resolve(), start=report_dir)

    # Sensor ROI ERP section (if plots exist)
    sensor_roi_erp_section = ""
    if roi_p1_path.exists() and roi_n1_path.exists() and roi_p3b_path.exists():
        rel_p1 = os.path.relpath(roi_p1_path.resolve(), start=report_dir)
        rel_n1 = os.path.relpath(roi_n1_path.resolve(), start=report_dir)
        rel_p3b = os.path.relpath(roi_p3b_path.resolve(), start=report_dir)
        sensor_roi_erp_section = SENSOR_ROI_ERP_SECTION_TEMPLATE.format(
            roi_p1_path=rel_p1,
            roi_n1_path=rel_n1,
            roi_p3b_path=rel_p3b
        )

    analysis_parameters_section = _build_analysis_parameters_table(sensor_config_path, config, report_dir)

    # Generate data quality section
    data_quality_section = ""
    try:
        project_root = sensor_config_path.resolve().parents[2]
        data_quality_info = get_preprocessing_info_from_config(project_root=project_root)
        if data_quality_info:
            data_quality_section = data_quality_info.to_html()
    except Exception as e:
        log.warning(f"Could not generate data quality section: {e}")
        data_quality_section = ""

    methods_summary_paragraph = (
        "We performed group-level statistical analysis using a non-parametric cluster-based permutation test to control for multiple comparisons across space and time. "
        "When sensor-space effects reached significance, they were localized to the cortical surface using the configured source estimation methods (e.g., dSPM, eLORETA). "
        "We then conducted region-of-interest (ROI) cluster-based permutation testing on the source data to identify the anatomical origins of the effect."
    )

    # Generate enhanced caption for the main sensor plot (Cluster #1)
    sensor_caption_detail = ""
    if 1 in sensor_cluster_details:
        info = sensor_cluster_details[1]
        # Infer topography from channels
        channels = info.get('channels', [])
        topography = infer_topography_from_channels(channels) if channels else "posterior"
        # Infer time window label
        time_window_label = infer_time_window_label(info.get('time_window', ''))
        # Add topography to info dict
        info_with_topo = {**info, 'topography': topography}

        try:
            # Generate full enhanced caption
            full_caption = generate_sensor_caption(
                cluster_id=1,
                cluster_info=info_with_topo,
                contrast_name=config['contrast']['name'],
                time_window_label=time_window_label
            )
            # Use the full caption as the detail
            sensor_caption_detail = f" {full_caption}"
        except Exception as e:
            # Fallback to simple caption
            log.warning(f"Failed to generate enhanced sensor caption: {e}")
            sensor_caption_detail = f" (p={info['p_value']:.4f}, peak t={info['peak_t']:.2f}, {info['n_channels']} channels, {info['time_window']})"

    final_html = HTML_TEMPLATE.format(
        title=f"{base_title} — Sensor + Source Report",
        date=datetime.now().strftime("%Y-%m-%d"),
        contrast_name=config['contrast']['name'],
        run_details_section=_build_run_details_section(run_command, accuracy, data_source, sensor_config_path, report_dir),
        data_quality_section=data_quality_section,
        sensor_stats=sensor_stats,
        erp_plot_path=erp_plot_rel_path,
        topo_plot_path=topo_plot_rel_path,
        sensor_main_caption_detail=sensor_caption_detail, # Add this
        sensor_roi_erp_section=sensor_roi_erp_section,
        sensor_extra_clusters_section=sensor_extra_clusters_section,
        dSPM_section=source_section_html,
        eloreta_section=eloreta_section_html,
        analysis_parameters_section=analysis_parameters_section,
        methods_summary_paragraph=methods_summary_paragraph
    )

    # --- 3. Write Report ---
    with open(report_output_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
        
    log.info(f"Successfully generated HTML report: {report_output_path}")

    # --- 4. Optionally export a PDF alongside the HTML ---
    try:
        # Always overwrite PDF by removing any existing one first
        pdf_path = report_output_path.with_suffix('.pdf')
        try:
            if pdf_path.exists():
                pdf_path.unlink()
        except Exception:
            pass
        _export_html_to_pdf(report_output_path)
    except Exception as e:
        log.warning(f"Failed to export report PDF: {e}")


def _export_html_to_pdf(html_path: Path) -> None:
    """Best-effort HTML→PDF export using wkhtmltopdf, WeasyPrint, or pdfkit.

    Saves the PDF next to the HTML. If no backend is available, logs a warning.
    """
    pdf_path = html_path.with_suffix('.pdf')

    # 1) Use wkhtmltopdf CLI if available
    wkhtml = shutil.which('wkhtmltopdf')
    if wkhtml:
        try:
            subprocess.run([wkhtml, '--quiet', str(html_path), str(pdf_path)], check=True)
            log.info(f"Saved PDF report via wkhtmltopdf: {pdf_path}")
            return
        except Exception as e:
            log.warning(f"wkhtmltopdf failed: {e}")

    # 2) Try WeasyPrint if installed
    try:
        from weasyprint import HTML  # type: ignore
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        log.info(f"Saved PDF report via WeasyPrint: {pdf_path}")
        return
    except Exception as e:
        log.warning(f"WeasyPrint not available or failed: {e}")

    # 3) Try pdfkit if installed (may still require wkhtmltopdf on PATH)
    try:
        import pdfkit  # type: ignore
        configuration = pdfkit.configuration(wkhtmltopdf=wkhtml) if wkhtml else None
        pdfkit.from_file(str(html_path), str(pdf_path), configuration=configuration)
        log.info(f"Saved PDF report via pdfkit: {pdf_path}")
        return
    except Exception as e:
        log.warning(f"pdfkit not available or failed: {e}")

    log.warning("No HTML→PDF backend available. Install 'wkhtmltopdf' or 'weasyprint' to enable PDF export.")
