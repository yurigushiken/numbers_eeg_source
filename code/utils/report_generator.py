import yaml
from pathlib import Path
import logging
from datetime import datetime
import os
import pandas as pd
import shutil
import subprocess

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
                <figcaption><strong>Figure 1.</strong> The grand average difference wave over the significant posterior cluster (left) and the topographical distribution of T-values during the significant time window (right).</figcaption>
            </div>
        </section>

        {hs_summary_section}

        <section id="source-results">
            <h2>Source-Space Localization</h2>
            {source_section}
        </section>
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

    # Try to find matching source YAML by filename substitution (sensor->source)
    source_cfg = None
    try:
        candidate_name = sensor_config_path.name.replace('sensor_', 'source_').replace('sensor', 'source')
        candidate = sensor_config_path.with_name(candidate_name)
        if candidate.exists():
            with open(candidate, 'r') as f:
                source_cfg = yaml.safe_load(f)
        else:
            alt_path = Path(str(sensor_config_path).replace('sensor_', 'source_').replace('sensor', 'source'))
            if alt_path.exists():
                with open(alt_path, 'r') as f:
                    source_cfg = yaml.safe_load(f)
    except Exception:
        source_cfg = None

    # Source parameters (if available)
    src_tmin = src_tmax = src_method = src_snr = src_p = src_alpha = src_nperm = src_tail = src_roi = None
    if isinstance(source_cfg, dict):
        src_tmin = source_cfg.get('tmin')
        src_tmax = source_cfg.get('tmax')
        src_method = (source_cfg.get('source') or {}).get('method')
        src_snr = (source_cfg.get('source') or {}).get('snr')
        src_stats = (source_cfg.get('stats') or {})
        src_p = src_stats.get('p_threshold')
        src_alpha = src_stats.get('cluster_alpha')
        src_nperm = src_stats.get('n_permutations')
        src_tail = _format_tail_descriptor(int(src_stats.get('tail', 0)))
        roi = (src_stats.get('roi') or {}).get('labels')
        if isinstance(roi, list):
            src_roi = ", ".join(str(x) for x in roi)
        elif isinstance(roi, str):
            src_roi = roi

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

    # Source localization (if present)
    if any(v is not None for v in [src_method, src_snr, src_tmin, src_tmax, src_p, src_alpha, src_nperm, src_roi]):
        if src_method is not None:
            rows.append(("Source Localization", "Method", str(src_method)))
        if src_snr is not None:
            rows.append(("Source Localization", "SNR Estimate", str(src_snr)))
        # Source stats
        rows.append(("Source Statistics", "Test Type", "Spatio-temporal Cluster Permutation"))
        if src_tmin is not None and src_tmax is not None:
            rows.append(("Source Statistics", "Analysis Window", f"{_format_seconds(src_tmin)} to {_format_seconds(src_tmax)} s"))
        if src_roi is not None:
            rows.append(("Source Statistics", "Region of Interest (ROI)", src_roi))
        if src_p is not None:
            rows.append(("Source Statistics", "Cluster-forming Threshold (p)", f"{src_p} ({src_tail})"))
        if src_alpha is not None:
            rows.append(("Source Statistics", "Cluster Significance (alpha)", f"{src_alpha}"))
        if src_nperm is not None:
            rows.append(("Source Statistics", "Permutations", f"{src_nperm}"))

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


def create_html_report(sensor_config_path, sensor_output_dir, source_output_dir, report_output_path):
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

    # Source paths and logic
    source_section_html = NULL_SOURCE_SECTION_TEMPLATE.format(label_ts_section="")
    hs_summary_section = ""
    if source_output_dir:
        source_output_dir = Path(source_output_dir)
        source_analysis_name = analysis_name.replace("sensor", "source")
        source_report_path = source_output_dir / f"{source_analysis_name}_report.txt"
        source_plot_path = source_output_dir / f"{source_analysis_name}_source_cluster.png"
        anatomical_report_path = source_output_dir / f"{source_analysis_name}_anatomical_report.csv"
        hs_summary_path = source_output_dir / f"{source_analysis_name}_anatomical_summary_hs.csv"
        label_ts_summary_path = source_output_dir / "aux" / "label_cluster_summary.txt"
        
        if source_report_path.exists():
            source_stats = read_report_file(source_report_path)
            anatomical_table_html = _generate_anatomical_table_html(anatomical_report_path)
            # Build HS summary section if present
            if hs_summary_path.exists():
                try:
                    hs_df = pd.read_csv(hs_summary_path)
                    # Drop any Brodmann's Area related columns if present
                    drop_cols = [c for c in hs_df.columns if 'brodmann' in str(c).lower()]
                    if drop_cols:
                        hs_df = hs_df.drop(columns=drop_cols)
                    if not hs_df.empty:
                        hs_summary_section = (
                            "<section id=\"hs-summary\">"
                            "<h2>Cortical Cluster Localization Summary</h2>"
                            + hs_df.to_html(index=False, classes='anatomical-table') +
                            "</section>"
                        )
                except Exception:
                    hs_summary_section = ""

            if source_plot_path.exists():
                # Make path relative to the final report's location for portability
                source_plot_rel_path = os.path.relpath(source_plot_path.resolve(), start=report_dir)
                source_section_html = SOURCE_SECTION_TEMPLATE.format(
                    source_stats=source_stats,
                    anatomical_table=anatomical_table_html,
                    source_plot_path=source_plot_rel_path
                )
            else:
                # Show stats and anatomical table even if plot is missing
                label_ts_summary = None
                if label_ts_summary_path.exists():
                    try:
                        label_ts_summary = label_ts_summary_path.read_text()
                    except Exception:
                        label_ts_summary = None
                label_ts_section = LABEL_TS_SECTION_TEMPLATE.format(label_ts_summary=label_ts_summary) if label_ts_summary else ""
                source_section_html = SOURCE_SECTION_STATS_ONLY_TEMPLATE.format(
                    source_stats=source_stats,
                    anatomical_table=anatomical_table_html,
                    label_ts_section=label_ts_section
                )
        else:
            # Neither stats nor plot available; keep null message, possibly with label TS
            label_ts_summary = None
            if label_ts_summary_path.exists():
                try:
                    label_ts_summary = label_ts_summary_path.read_text()
                except Exception:
                    label_ts_summary = None
            label_ts_section = LABEL_TS_SECTION_TEMPLATE.format(label_ts_summary=label_ts_summary) if label_ts_summary else ""
            source_section_html = NULL_SOURCE_SECTION_TEMPLATE.format(label_ts_section=label_ts_section)

    # --- 2. Populate Template ---
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

    methods_summary_paragraph = (
        "We performed group-level statistical analysis using a non-parametric cluster-based permutation test to control for multiple comparisons across space and time. "
        "Significant sensor-space effects were then localized to the cortical surface using dSPM. "
        "We performed a subsequent region-of-interest (ROI) cluster-based permutation test on the source data to identify the anatomical origins of the effect."
    )

    final_html = HTML_TEMPLATE.format(
        title=f"{base_title} — Sensor + Source Report",
        date=datetime.now().strftime("%Y-%m-%d"),
        contrast_name=config['contrast']['name'],
        sensor_stats=sensor_stats,
        erp_plot_path=erp_plot_rel_path,
        topo_plot_path=topo_plot_rel_path,
        sensor_roi_erp_section=sensor_roi_erp_section,
        source_section=source_section_html,
        hs_summary_section=hs_summary_section,
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
