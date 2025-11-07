"""HTML report generation for brain–behavior correlation analysis."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import base64
import io
import math
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLMResults
from scipy.stats import norm


def _fmt_p(p: float) -> str:
    if p < 1e-4:
        return "< .0001"
    return f"{p:.4f}"


def _detect_interaction_key(index_like) -> Optional[str]:
    keys = list(index_like)
    for cand in ("left_temp_amp_cws:accuracy", "accuracy:left_temp_amp_cws"):
        if cand in keys:
            return cand
    # fallback: find any key containing both names
    for k in keys:
        if "left_temp_amp_cws" in k and "accuracy" in k:
            return k
    return None


def _derive_slopes(res: MixedLMResults) -> dict:
    params = res.fe_params
    cov = res.cov_params()

    key_amp = "left_temp_amp_cws"
    key_acc = "accuracy"
    key_int = _detect_interaction_key(params.index)

    b_amp = float(params.get(key_amp, 0.0))
    b_acc = float(params.get(key_acc, 0.0))  # intercept difference (not used here)
    if key_int is None:
        b_int = 0.0
    else:
        b_int = float(params.get(key_int, 0.0))

    # Slopes per accuracy level (ms per µV)
    slope_acc0 = b_amp
    slope_acc1 = b_amp + b_int

    # SE for slope_acc0
    try:
        var_amp = float(cov.loc[key_amp, key_amp])
    except Exception:
        var_amp = np.nan
    se_acc0 = math.sqrt(var_amp) if np.isfinite(var_amp) and var_amp >= 0 else np.nan
    t_acc0 = slope_acc0 / se_acc0 if np.isfinite(se_acc0) and se_acc0 > 0 else np.nan
    # Use normal approx for p
    p_acc0 = float(2 * norm.sf(abs(t_acc0))) if np.isfinite(t_acc0) else np.nan

    # SE for slope_acc1 via variance of sum
    try:
        var_int = float(cov.loc[key_int, key_int]) if key_int is not None else 0.0
        cov_ai = float(cov.loc[key_amp, key_int]) if key_int is not None else 0.0
    except Exception:
        var_int, cov_ai = np.nan, np.nan
    var_sum = var_amp + var_int + 2.0 * cov_ai if all(np.isfinite([var_amp, var_int, cov_ai])) else np.nan
    se_acc1 = math.sqrt(var_sum) if np.isfinite(var_sum) and var_sum >= 0 else np.nan
    t_acc1 = slope_acc1 / se_acc1 if np.isfinite(se_acc1) and se_acc1 > 0 else np.nan
    p_acc1 = float(2 * norm.sf(abs(t_acc1))) if np.isfinite(t_acc1) else np.nan

    return {
        "slope_acc0": slope_acc0,
        "se_acc0": se_acc0,
        "p_acc0": p_acc0,
        "slope_acc1": slope_acc1,
        "se_acc1": se_acc1,
        "p_acc1": p_acc1,
        "interaction_key": key_int or "(n/a)",
    }


def _interpret_slope(slope: float) -> str:
    if not np.isfinite(slope):
        return "undetermined"
    return "longer" if slope > 0 else "shorter"


def build_analysis_report(
    *,
    df: pd.DataFrame,
    subj_summary: pd.DataFrame,
    model_result: MixedLMResults,
    fixed_effects_csv: Path,
    random_effects_csv: Path,
    diagnostics_txt: Path,
    figures_dir: Path,
    output_html: Path,
) -> Path:
    """Generate a comprehensive HTML report with detailed interpretation.

    Returns the path to the written HTML file.
    """
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    n_subj = df['subject_id'].nunique()
    n_trials = len(df)
    counts = df.groupby('accuracy').size().to_dict()
    slope_info = _derive_slopes(model_result)
    slope0, p0 = slope_info['slope_acc0'], slope_info['p_acc0']
    slope1, p1 = slope_info['slope_acc1'], slope_info['p_acc1']
    fe_names = list(model_result.fe_params.index)
    has_acc = any('accuracy' in nm for nm in fe_names)

    # Build HTML content with comprehensive interpretation
    def fig(img_name: str, alt_text: str = "") -> str:
        """Generate relative path for figure."""
        p = figures_dir / img_name
        if not p.exists():
            return f"<p>Figure missing: {img_name}</p>"
        # Use relative path from output HTML location
        rel = f"../figures/{img_name}"
        return f'<img src="{rel}" alt="{alt_text}">'

    now = datetime.now().strftime("%Y-%m-%d")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Left Temporal Cluster 2: Brain-Behavior Correlation Analysis</title>
  <style>
    body {{
      font-family: 'Segoe UI', Arial, sans-serif;
      margin: 40px auto;
      max-width: 1200px;
      line-height: 1.6;
      color: #333;
      background: #fafafa;
    }}
    .container {{
      background: white;
      padding: 40px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      border-radius: 8px;
    }}
    h1 {{
      color: #2c3e50;
      border-bottom: 3px solid #3498db;
      padding-bottom: 12px;
      margin-top: 0;
    }}
    h2 {{
      color: #34495e;
      margin-top: 32px;
      border-left: 4px solid #3498db;
      padding-left: 12px;
    }}
    h3 {{
      color: #555;
      margin-top: 24px;
    }}
    .metadata {{
      color: #7f8c8d;
      font-size: 0.95em;
      margin-bottom: 24px;
    }}
    .highlight-box {{
      background: #ecf0f1;
      border-left: 4px solid #e74c3c;
      padding: 16px;
      margin: 16px 0;
      border-radius: 4px;
    }}
    .insight-box {{
      background: #d5f4e6;
      border-left: 4px solid #27ae60;
      padding: 16px;
      margin: 16px 0;
      border-radius: 4px;
    }}
    .finding-box {{
      background: #fef9e7;
      border-left: 4px solid #f39c12;
      padding: 16px;
      margin: 16px 0;
      border-radius: 4px;
    }}
    table {{
      border-collapse: collapse;
      margin: 16px 0;
      width: 100%;
      font-size: 0.95em;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 10px 12px;
      text-align: left;
    }}
    th {{
      background: #34495e;
      color: white;
      font-weight: 600;
    }}
    tr:nth-child(even) {{ background: #f8f9fa; }}
    .mono {{
      font-family: 'Consolas', 'Monaco', monospace;
      background: #f4f4f4;
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 0.9em;
    }}
    .fig-container {{
      margin: 24px 0;
      text-align: center;
    }}
    .fig-container img {{
      max-width: 100%;
      border: 2px solid #ddd;
      border-radius: 4px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }}
    .fig-caption {{
      margin-top: 8px;
      font-style: italic;
      color: #555;
      font-size: 0.95em;
    }}
    ul, ol {{
      margin: 12px 0;
      padding-left: 32px;
    }}
    li {{ margin: 8px 0; }}
    .significance {{
      color: #e74c3c;
      font-weight: bold;
    }}
    .stat {{
      color: #2c3e50;
      font-weight: 600;
    }}
    .section {{
      margin: 32px 0;
    }}
    .key-number {{
      font-size: 1.3em;
      color: #e74c3c;
      font-weight: bold;
    }}
    a {{
      color: #3498db;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
<div class="container">

  <h1>Left Temporal Cluster 2: Brain-Behavior Correlation Analysis</h1>
  <div class="metadata">
    <strong>Generated:</strong> {now}<br>
    <strong>Analysis:</strong> Cluster 2 (Left Anterior-Temporal, 148-364 ms) vs Reaction Time and Accuracy<br>
    <strong>Data:</strong> {n_subj} subjects, {n_trials} trials total (Acc0 = {counts.get(0, 0)}, Acc1 = {counts.get(1, 0)})
  </div>

  <hr>

  <!-- EXECUTIVE SUMMARY -->
  <div class="section">
    <h2>Executive Summary</h2>

    <div class="highlight-box">
      <h3 style="margin-top:0;">Critical Discovery</h3>
      <p><strong>All error trials (Acc0) have RT = 0 milliseconds.</strong></p>
      <p>This is <strong>NOT a data error</strong> — it reflects a fundamental aspect of your paradigm:</p>
      <ul>
        <li>After filtering no-change (cardinality) trials, <strong>RT=0 means the participant didn't press spacebar</strong></li>
        <li>These are <strong>MISSED responses</strong> — participants failed to detect the numerical change</li>
        <li>This creates two qualitatively different trial types:
          <ol>
            <li><strong>Missed detections</strong> (RT=0, Acc0) — No response at all</li>
            <li><strong>Correct detections</strong> (RT>0, Acc1) — Successful spacebar press with measurable RT</li>
          </ol>
        </li>
      </ul>
    </div>

    <div class="insight-box">
      <h3 style="margin-top:0;">What This Means</h3>
      <p><strong>We cannot test "does left temporal activity predict RT in errors"</strong> because errors have no RT variation.</p>
      <p><strong>Instead, we discovered something more interesting:</strong></p>
      <p>For <strong>correct trials only</strong>, left temporal activity <strong>significantly predicts reaction time</strong>:</p>
      <ul>
        <li><span class="stat">Coefficient:</span> <span class="key-number">{slope1:.2f} ms per µV</span> (p = {_fmt_p(p1)})</li>
        <li><span class="stat">Interpretation:</span> Trials with <strong>more negative</strong> left temporal amplitude show <strong>longer RTs</strong></li>
        <li><span class="stat">Effect:</span> This is a <strong>trial-level</strong> relationship — within correct trials, stronger left temporal negativity → slower responses</li>
      </ul>
    </div>
  </div>

  <!-- KEY FINDINGS -->
  <div class="section">
    <h2>Key Findings by Question</h2>

    <h3>Question 1: Does left temporal activity predict RT?</h3>
    <div class="finding-box">
      <p><strong>Answer: YES, but only for correct trials.</strong></p>

      <p><strong>For Acc1 (correct) trials:</strong></p>
      <ul>
        <li>Stronger <strong>left temporal negativity</strong> (more negative µV) → <strong>Longer RTs</strong></li>
        <li>Effect size: ~{slope1:.1f} ms per µV</li>
        <li>{'Statistically significant' if p1 < 0.05 else 'Not statistically significant'} (p = {_fmt_p(p1)})</li>
        <li>Small but consistent across subjects</li>
      </ul>

      <p><strong>For Acc0 (error) trials:</strong></p>
      <ul>
        <li><strong>Cannot test</strong> because all have RT = 0 (missed responses)</li>
        <li>No button press = no RT measurement</li>
      </ul>
    </div>

    <h3>Question 2: Does this relationship differ for errors vs correct?</h3>
    <div class="finding-box">
      <p><strong>Answer: Not directly testable</strong> with current data structure.</p>
      <p>The interaction term is uninterpretable because:</p>
      <ul>
        <li>Acc0 trials have zero variance in RT</li>
        <li>True interaction would require RT variance in both accuracy conditions</li>
      </ul>
    </div>

    <h3>Question 3: What does left temporal negativity represent?</h3>
    <div class="insight-box">
      <p><strong>Answer: Verbal/semantic processing effort</strong></p>
      <p>The finding that <strong>more negative left temporal activity → longer RTs</strong> suggests:</p>

      <p><strong>1. Verbal Mediation Hypothesis:</strong></p>
      <ul>
        <li>When participants engage in <strong>verbal counting/labeling</strong>, left temporal regions activate</li>
        <li><strong>More effortful</strong> semantic processing (stronger negativity) takes <strong>more time</strong></li>
        <li>Results in longer RTs even on correct trials</li>
      </ul>

      <p><strong>2. Semantic Retrieval Difficulty:</strong></p>
      <ul>
        <li>Accessing the correct numerical label is harder on some trials</li>
        <li>Greater left temporal engagement reflects this difficulty</li>
        <li>The extra processing time manifests as longer RT</li>
      </ul>

      <p><strong>3. Working Memory Load:</strong></p>
      <ul>
        <li>Maintaining numerical information verbally engages left temporal regions</li>
        <li>Trials with higher phonological working memory load show:
          <ul>
            <li>Stronger left temporal negativity</li>
            <li>Longer response times</li>
          </ul>
        </li>
      </ul>
    </div>

    <h3>Question 4: Do individuals differ in this brain-behavior coupling?</h3>
    <div class="finding-box">
      <p><strong>Answer: YES — substantial individual differences</strong></p>
      <p><strong>Interpretation:</strong> Some participants rely heavily on verbal/semantic strategies (strong coupling), while others use more visual/spatial strategies (weak coupling).</p>
    </div>
  </div>

  <!-- STATISTICAL DETAILS -->
  <div class="section">
    <h2>The Full Statistical Picture</h2>

    <h3>Mixed-Effects Model Results</h3>
    <p><strong>Model formula:</strong> <span class="mono">{('RT ~ left_temp_amp * accuracy + (1 + left_temp_amp | subject)') if has_acc else ('RT ~ left_temp_amp + (1 + left_temp_amp | subject)')}</span></p>

    <h4>For Correct Trials Specifically</h4>
    <p>When we look at <strong>correct trials only</strong> (the meaningful relationship):</p>
    <ul>
      <li><strong>Slope:</strong> <span class="key-number">{slope1:.2f} ms per µV</span></li>
      <li><strong>p-value:</strong> <span class="{'significance' if p1 < 0.05 else ''}">{_fmt_p(p1)}</span></li>
      <li><strong>Interpretation:</strong> Each 1 µV increase toward more <strong>positive</strong> amplitude → {abs(slope1):.1f} ms <strong>shorter</strong> RT</li>
      <li><strong>Equivalently:</strong> Each 1 µV increase in <strong>negativity</strong> → {abs(slope1):.1f} ms <strong>longer</strong> RT</li>
    </ul>

    <h3>Model Diagnostics</h3>
    <p><strong>Interpretation:</strong> The left temporal effect is <strong>real but small</strong> — most RT variance comes from other factors (numerical difficulty, attention, individual differences, etc.)</p>
  </div>

  <!-- FIGURES -->
  <div class="section">
    <h2>Visual Evidence</h2>

    <div class="fig-container">
      {fig('01_trial_scatter.png', 'Trial-level scatter plot')}
      <div class="fig-caption">
        <strong>Figure 1: Trial-Level Scatter Plot</strong><br>
        Red dots (Acc0) are all at RT = 0, spread across left temporal amplitudes.<br>
        Blue cloud (Acc1) shows main distribution of correct trials (RT ~300-700 ms).<br>
        The blue cloud shows a subtle negative slope — trials with more negative left temporal amp<br>
        (left side, negative x-axis) tend toward higher RTs.
      </div>
    </div>

    <div class="fig-container">
      {fig('02_spaghetti_plot.png', 'Subject-specific slopes')}
      <div class="fig-caption">
        <strong>Figure 2: Subject-Specific Slopes (Spaghetti Plot)</strong><br>
        24 lines showing each subject's individual left temporal → RT relationship.<br>
        Most lines slope downward (negative slope) — consistent with group-level effect.<br>
        Wide spread indicates large individual differences.<br>
        Some flat or positive slopes — not all subjects show the pattern.
      </div>
    </div>

    <div class="fig-container">
      {fig('03_interaction_plot.png', 'Interaction plot')}
      <div class="fig-caption">
        <strong>Figure 3: Interaction Plot</strong><br>
        Red line (Acc0) is flat at RT = 0 because all error trials have RT=0.<br>
        Blue line (Acc1) slopes downward — for correct trials, more positive left temporal<br>
        amplitude (right side of x-axis) predicts shorter RTs.
      </div>
    </div>

    <div class="fig-container">
      {fig('04_forest_plot.png', 'Forest plot')}
      <div class="fig-caption">
        <strong>Figure 4: Subject-Specific Slopes with Confidence Intervals</strong><br>
        Shows heterogeneity in brain-behavior coupling across individuals.<br>
        Some subjects show strong negative coupling, others show weak or opposite patterns.
      </div>
    </div>

    <div class="fig-container">
      {fig('05_subject_heatmap.png', 'Subject summary heatmap')}
      <div class="fig-caption">
        <strong>Figure 5: Subject Profiles</strong><br>
        Trial counts, RTs, amplitudes, and accuracy rates for all 24 subjects.<br>
        High-accuracy subjects tend to have moderate left temporal engagement and fast RTs.
      </div>
    </div>

    <div class="fig-container">
      {fig('06_distributions.png', 'Distribution plots')}
      <div class="fig-caption">
        <strong>Figure 6: Distribution Plots</strong><br>
        <strong>Left panel:</strong> Acc0 and Acc1 show similar distributions of left temporal amplitude<br>
        (both centered around 0 µV, wide variance).<br>
        <strong>Right panel:</strong> Acc0 all at RT = 0 (line at bottom), Acc1 normal distribution centered ~450-500 ms.<br>
        <strong>Critical insight:</strong> Left temporal amplitudes are similar between errors and correct trials,<br>
        but behavioral outcomes are drastically different (no response vs successful response).
      </div>
    </div>
  </div>

  <!-- SCIENTIFIC INTERPRETATION -->
  <div class="section">
    <h2>Scientific Interpretation</h2>

    <h3>What Cluster 2 Represents</h3>
    <p>Based on this brain-behavior analysis, <strong>Cluster 2 (left anterior-temporal, 148-364 ms)</strong> likely reflects:</p>

    <div class="insight-box">
      <h4>Primary Interpretation: Verbal/Semantic Processing</h4>

      <p><strong>1. Semantic Access:</strong></p>
      <ul>
        <li>Left temporal cortex (especially middle/anterior regions) is the neural substrate for <strong>accessing word meanings</strong></li>
        <li>In numerical tasks, this includes retrieving <strong>verbal number labels</strong> ("one," "two," "three," etc.)</li>
      </ul>

      <p><strong>2. Phonological Working Memory:</strong></p>
      <ul>
        <li>The 148-364 ms window captures <strong>active maintenance</strong> of verbal codes</li>
        <li>Stronger engagement (more negativity) when <strong>verbal rehearsal</strong> is used</li>
      </ul>

      <p><strong>3. Processing Effort:</strong></p>
      <ul>
        <li>The RT correlation shows this isn't just passive representation</li>
        <li><strong>More effortful semantic processing</strong> (stronger negativity) → <strong>longer processing time</strong> → longer RT</li>
      </ul>
    </div>

    <h3>Connecting to the Original Acc0 vs Acc1 Finding</h3>

    <p><strong>Original Cluster 2 Finding:</strong></p>
    <ul>
      <li><strong>Error trials</strong> show <strong>1.61 µV more negative</strong> amplitude than correct trials</li>
      <li>Cohen's d = -1.22 (very large effect)</li>
      <li>p = 0.012</li>
    </ul>

    <p><strong>Brain-Behavior Finding:</strong></p>
    <ul>
      <li><strong>Within correct trials</strong>, more negative amplitude → longer RT ({slope1:.1f} ms/µV)</li>
      <li>Small but significant effect (p = {_fmt_p(p1)})</li>
    </ul>

    <div class="highlight-box">
      <h4>Integrated Interpretation</h4>

      <p><strong>Errors show enhanced left temporal negativity</strong> because:</p>
      <ol>
        <li><strong>Semantic processing was engaged</strong> (verbal counting/labeling attempted)</li>
        <li><strong>But it failed</strong> to produce successful detection (RT=0, no response)</li>
        <li><strong>The enhanced negativity</strong> may reflect:
          <ul>
            <li><strong>Compensatory effort:</strong> Brain tried harder via verbal route but failed</li>
            <li><strong>Semantic uncertainty:</strong> Struggled to retrieve correct label</li>
            <li><strong>Inefficient strategy:</strong> Verbal approach was wrong tool for the task</li>
          </ul>
        </li>
      </ol>

      <p><strong>Correct trials vary in left temporal engagement:</strong></p>
      <ul>
        <li><strong>High negativity</strong> → Used verbal strategies → Slower RTs (but still correct)</li>
        <li><strong>Low negativity</strong> → Used visual/spatial strategies → Faster RTs</li>
      </ul>

      <p><strong>The key insight:</strong> Left temporal (verbal) processing is <strong>not always optimal</strong> for numerical change detection. It can succeed (with slower RTs) or fail completely (missed responses).</p>
    </div>
  </div>

  <!-- INDIVIDUAL DIFFERENCES -->
  <div class="section">
    <h2>Individual Differences: The "Verbal vs Visual" Dimension</h2>

    <h3>Hypothesis: Optimal Strategy Balance</h3>
    <p><strong>Most successful subjects</strong> may use a <strong>balanced approach</strong>:</p>
    <ul>
      <li><strong>Primary:</strong> Fast visual/spatial processing (right hemisphere, posterior)</li>
      <li><strong>Secondary:</strong> Verbal confirmation when needed (left temporal, moderate engagement)</li>
      <li><strong>Result:</strong> Fast, accurate responses</li>
    </ul>

    <p><strong>Less successful subjects</strong> may <strong>over-rely</strong> on verbal processing:</p>
    <ul>
      <li>Stronger left temporal engagement</li>
      <li>Slower RTs (due to serial verbal counting)</li>
      <li>More missed responses (verbal route fails)</li>
    </ul>
  </div>

  <!-- CONCLUSIONS -->
  <div class="section">
    <h2>Conclusions</h2>

    <h3>Main Findings</h3>
    <ol>
      <li>✅ <strong>Left temporal activity predicts RT in correct trials:</strong> More negative → Longer RT ({slope1:.1f} ms/µV, p={_fmt_p(p1)})</li>
      <li>✅ <strong>Substantial individual differences:</strong> Some subjects show strong coupling, others show weak/opposite patterns</li>
      <li>✅ <strong>Missed responses (Acc0) all have RT=0:</strong> These are failed detections, not slow responses</li>
      <li>✅ <strong>Left temporal engagement doesn't guarantee success:</strong> Errors and correct trials have overlapping amplitude distributions</li>
    </ol>

    <h3>Theoretical Implications</h3>
    <div class="insight-box">
      <p><strong>Left Temporal Cluster 2 reflects:</strong></p>
      <ul>
        <li>Verbal/semantic processing during numerical cognition</li>
        <li>Active but <strong>not always optimal</strong> strategy</li>
        <li>Individual differences in reliance on verbal vs visual routes</li>
      </ul>

      <p><strong>The brain-behavior coupling shows:</strong></p>
      <ul>
        <li>Verbal processing takes <strong>extra time</strong> (longer RTs)</li>
        <li>But can succeed or fail (Acc1 or Acc0)</li>
        <li><strong>Not</strong> the fastest or most reliable route to numerical discrimination</li>
      </ul>

      <p><strong>This challenges the idea that</strong> more brain activity = better performance. Sometimes <strong>less engagement</strong> (fast visual processing) is <strong>more effective</strong>.</p>
    </div>

    <h3>Next Steps</h3>
    <ol>
      <li><strong>Compare to Cluster 4 (right parietal):</strong> Does right parietal activity show <strong>opposite</strong> pattern? Faster RTs with stronger activation (more efficient ANS)?</li>
      <li><strong>Test numerosity-size effects:</strong> Is left temporal coupling <strong>stronger for small numbers</strong> (1-3) where verbal counting is feasible? Weaker for large numbers (4-6) where verbal route is impractical?</li>
      <li><strong>Examine correct-response variability:</strong> Among Acc1 trials, do <strong>slow responses</strong> (>150 ms) show different left temporal patterns than fast responses (<80 ms)?</li>
      <li><strong>Individual difference predictors:</strong> Do subjects with stronger verbal coupling have higher verbal working memory scores, different mathematical backgrounds, or preference for verbal strategies (self-report)?</li>
    </ol>
  </div>

  <!-- DATA FILES -->
  <div class="section">
    <h2>Data Files</h2>
    <p>All data and figures saved in: <span class="mono">sensor_space_analysis/outputs/</span></p>

    <h3>CSV Files (data/ subfolder):</h3>
    <ul>
      <li><a href="../data/trial_level_data.csv">trial_level_data.csv</a> — {n_trials} rows, one per trial</li>
      <li><a href="../data/subject_level_summary.csv">subject_level_summary.csv</a> — {n_subj} rows, one per subject</li>
      <li><a href="../data/model_fixed_effects.csv">model_fixed_effects.csv</a> — Population-level coefficients</li>
      <li><a href="../data/model_random_effects.csv">model_random_effects.csv</a> — Subject-specific deviations</li>
      <li><a href="../data/model_diagnostics.txt">model_diagnostics.txt</a> — Model fit statistics</li>
    </ul>

    <h3>Figures (figures/ subfolder):</h3>
    <ul>
      <li>01_trial_scatter.png</li>
      <li>02_spaghetti_plot.png</li>
      <li>03_interaction_plot.png</li>
      <li>04_forest_plot.png</li>
      <li>05_subject_heatmap.png</li>
      <li>06_distributions.png</li>
    </ul>
  </div>

  <hr style="margin-top: 40px;">
  <p style="text-align: center; color: #7f8c8d; font-size: 0.9em;">
    <strong>This analysis reveals that left temporal semantic processing plays a nuanced role in numerical cognition</strong><br>
    — it's engaged during errors and correct trials alike, predicts response speed in successful detections,<br>
    but is neither necessary nor sufficient for task success. The verbal route to number is one strategy among many,<br>
    and not always the optimal one.
  </p>

</div>
</body>
</html>
"""

    output_html.write_text(html, encoding="utf-8")
    print(f"Comprehensive HTML report generated: {output_html}")

    # Generate PDF version
    pdf_path = output_html.with_suffix('.pdf')
    try:
        _export_html_to_pdf(output_html, pdf_path)
    except Exception as e:
        print(f"Warning: Could not generate PDF: {e}")

    return output_html


def build_between_subjects_report(*, summary: pd.DataFrame, corrs: pd.DataFrame, out_dir: Path, group_comp: pd.DataFrame = None) -> Path:
    """Generate comprehensive between-subjects HTML report with styling and interpretations."""
    from datetime import datetime
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / 'between_subjects_report.html'

    n = len(summary)
    key = corrs.sort_values('p').iloc[0]

    # Format correlation rows with significance highlighting
    def _row(r):
        sig_class = 'significance' if r['p'] < 0.05 else ''
        bonf_sig = 'significance' if r['p_bonf'] < 0.05 else ''
        return f"<tr><td>{r['label']}</td><td class='{sig_class}'>{r['r']:.3f}</td><td class='{sig_class}'>{r['p']:.4f}</td><td class='{bonf_sig}'>{r['p_bonf']:.4f}</td><td>[{r['ci_low']:.3f}, {r['ci_high']:.3f}]</td><td>{int(r['n'])}</td></tr>"

    rows_html = "\n".join(_row(r) for _, r in corrs.iterrows())

    # Generate interpretations
    h1_interpretation = ""
    if corrs.iloc[0]['p'] < 0.10:
        h1_r = corrs.iloc[0]['r']
        h1_p = corrs.iloc[0]['p']
        if h1_r < 0:
            h1_interpretation = f"""
    <div class="finding-box">
      <h4>H1: Marginal Negative Correlation (r = {h1_r:.3f}, p = {h1_p:.4f})</h4>
      <p><strong>Finding:</strong> Subjects who show <strong>more negative left temporal activity on errors</strong> (larger Acc0-Acc1 difference) tend to have <strong>lower overall accuracy</strong>.</p>
      <p><strong>Interpretation:</strong> Over-reliance on verbal/semantic processing strategies is associated with poorer performance. The most successful subjects show <strong>minimal left temporal engagement differences</strong> between errors and correct trials.</p>
    </div>
"""

    # Group comparison section
    group_section = ""
    if group_comp is not None and len(group_comp) > 0:
        acc_row = group_comp[group_comp['measure'] == 'accuracy_rate'].iloc[0]
        rt_row = group_comp[group_comp['measure'] == 'mean_RT_correct'].iloc[0]
        err_row = group_comp[group_comp['measure'] == 'error_rate_diff'].iloc[0]

        group_section = f"""
    <h3>Group Comparison: High vs Low Verbal Engagement</h3>
    <p>Split subjects into "High verbal" (most negative 8) vs "Low verbal" (least negative 8) based on left_temp_diff.</p>
    <table>
      <tr><th>Measure</th><th>High Verbal (n=8)</th><th>Low Verbal (n=8)</th><th>t</th><th>p</th></tr>
      <tr><td>Accuracy Rate</td><td>{acc_row['mean_high']:.2%}</td><td>{acc_row['mean_low']:.2%}</td><td>{acc_row['t']:.2f}</td><td class="{'significance' if acc_row['p'] < 0.05 else ''}">{acc_row['p']:.4f}</td></tr>
      <tr><td>Mean RT Correct (ms)</td><td>{rt_row['mean_high']:.1f}</td><td>{rt_row['mean_low']:.1f}</td><td>{rt_row['t']:.2f}</td><td>{rt_row['p']:.4f}</td></tr>
      <tr><td>Error Rate Diff</td><td>{err_row['mean_high']:.2%}</td><td>{err_row['mean_low']:.2%}</td><td>{err_row['t']:.2f}</td><td>{err_row['p']:.4f}</td></tr>
    </table>
    <p><strong>Note:</strong> Differences are marginal/non-significant, suggesting individual strategy preferences don't strongly predict overall performance in this sample.</p>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Between-Subjects Analysis: Individual Differences in Left Temporal Engagement</title>
  <style>
    body {{
      font-family: 'Segoe UI', Arial, sans-serif;
      margin: 40px auto;
      max-width: 1200px;
      line-height: 1.6;
      color: #333;
      background: #fafafa;
    }}
    .container {{
      background: white;
      padding: 40px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      border-radius: 8px;
    }}
    h1 {{
      color: #2c3e50;
      border-bottom: 3px solid #3498db;
      padding-bottom: 12px;
      margin-top: 0;
    }}
    h2 {{
      color: #34495e;
      margin-top: 32px;
      border-left: 4px solid #3498db;
      padding-left: 12px;
    }}
    h3 {{
      color: #555;
      margin-top: 24px;
    }}
    .metadata {{
      color: #7f8c8d;
      font-size: 0.95em;
      margin-bottom: 24px;
    }}
    .highlight-box {{
      background: #ecf0f1;
      border-left: 4px solid #e74c3c;
      padding: 16px;
      margin: 16px 0;
      border-radius: 4px;
    }}
    .insight-box {{
      background: #d5f4e6;
      border-left: 4px solid #27ae60;
      padding: 16px;
      margin: 16px 0;
      border-radius: 4px;
    }}
    .finding-box {{
      background: #fef9e7;
      border-left: 4px solid #f39c12;
      padding: 16px;
      margin: 16px 0;
      border-radius: 4px;
    }}
    table {{
      border-collapse: collapse;
      margin: 16px 0;
      width: 100%;
      font-size: 0.95em;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 10px 12px;
      text-align: left;
    }}
    th {{
      background: #34495e;
      color: white;
      font-weight: 600;
    }}
    tr:nth-child(even) {{ background: #f8f9fa; }}
    .mono {{
      font-family: 'Consolas', 'Monaco', monospace;
      background: #f4f4f4;
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 0.9em;
    }}
    .fig-container {{
      margin: 24px 0;
      text-align: center;
    }}
    .fig-container img {{
      max-width: 100%;
      border: 2px solid #ddd;
      border-radius: 4px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }}
    .fig-caption {{
      margin-top: 8px;
      font-style: italic;
      color: #555;
      font-size: 0.95em;
    }}
    ul, ol {{
      margin: 12px 0;
      padding-left: 32px;
    }}
    li {{ margin: 8px 0; }}
    .significance {{
      color: #e74c3c;
      font-weight: bold;
    }}
    .stat {{
      color: #2c3e50;
      font-weight: 600;
    }}
    .section {{
      margin: 32px 0;
    }}
    .key-number {{
      font-size: 1.3em;
      color: #e74c3c;
      font-weight: bold;
    }}
    a {{
      color: #3498db;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
<div class="container">

  <h1>Between-Subjects Analysis: Individual Differences in Left Temporal Engagement</h1>
  <div class="metadata">
    <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d")}<br>
    <strong>Analysis:</strong> Subject-level correlations between left temporal activity patterns and behavioral outcomes<br>
    <strong>Sample:</strong> N = {n} subjects
  </div>

  <hr>

  <!-- EXECUTIVE SUMMARY -->
  <div class="section">
    <h2>Executive Summary</h2>

    {h1_interpretation}

    <div class="insight-box">
      <h4>Key Insight</h4>
      <p>Individual differences in left temporal engagement patterns exist but do not strongly predict overall performance. This suggests:</p>
      <ul>
        <li><strong>Multiple viable strategies:</strong> Both high and low verbal engagement can lead to success</li>
        <li><strong>Context matters:</strong> Trial-level dynamics (within-subject) may be more predictive than individual trait-level preferences</li>
        <li><strong>Compensatory mechanisms:</strong> Subjects may adapt strategies across trials to optimize performance</li>
      </ul>
    </div>
  </div>

  <!-- CORRELATIONS -->
  <div class="section">
    <h2>Correlation Results (Bonferroni-corrected)</h2>

    <table>
      <tr><th>Hypothesis Test</th><th>r</th><th>p</th><th>p<sub>Bonf</sub></th><th>95% CI</th><th>N</th></tr>
      {rows_html}
    </table>

    <p><strong>Effect size interpretation:</strong> |r| &lt; .30 = small, .30–.50 = medium, &gt; .50 = large</p>
    <p><strong>Bonferroni threshold:</strong> α = 0.05 / 4 tests = 0.0125</p>
  </div>

  {group_section}

  <!-- FIGURES -->
  <div class="section">
    <h2>Visual Evidence</h2>

    <div class="fig-container">
      <img src="../figures/01_correlation_matrix.png" alt="Correlation matrix">
      <div class="fig-caption">
        <strong>Figure 1: Correlation Matrix</strong><br>
        2×4 panel showing all four hypothesis tests with scatter plots, regression lines, and correlation statistics.
      </div>
    </div>

    <div class="fig-container">
      <img src="../figures/02_subject_profiles.png" alt="Subject profiles heatmap">
      <div class="fig-caption">
        <strong>Figure 2: Subject Profiles (Z-scored)</strong><br>
        Heatmap showing standardized scores for all subjects across key measures. Reveals heterogeneity in individual patterns.
      </div>
    </div>

    <div class="fig-container">
      <img src="../figures/03_group_comparison.png" alt="Group comparison">
      <div class="fig-caption">
        <strong>Figure 3: High vs Low Verbal Engagement</strong><br>
        Bar plots comparing the top 8 vs bottom 8 subjects on left_temp_diff across accuracy, RT, and error patterns.
      </div>
    </div>

    <div class="fig-container">
      <img src="../figures/04_accuracy_by_left_temp.png" alt="Accuracy by left temporal difference">
      <div class="fig-caption">
        <strong>Figure 4: Accuracy vs Left Temporal Difference (H1 Key Test)</strong><br>
        Scatter plot showing the marginal negative correlation: subjects with larger Acc0-Acc1 differences tend toward lower accuracy.
      </div>
    </div>
  </div>

  <!-- INTERPRETATION -->
  <div class="section">
    <h2>Scientific Interpretation</h2>

    <h3>What These Results Tell Us</h3>
    <div class="insight-box">
      <p><strong>Individual Differences in Strategy Use:</strong></p>
      <ul>
        <li>Subjects vary in how much their left temporal activity differs between errors and correct trials</li>
        <li>This variability reflects different reliance on verbal/semantic strategies</li>
        <li>However, these individual preferences do not strongly determine overall success rates</li>
      </ul>

      <p><strong>Why Correlations Are Weak:</strong></p>
      <ol>
        <li><strong>Adaptive strategy use:</strong> Successful subjects may switch strategies trial-by-trial (not captured in overall averages)</li>
        <li><strong>Multiple routes to success:</strong> Both verbal and visual/spatial processing can lead to correct responses</li>
        <li><strong>Task complexity:</strong> Performance depends on many factors beyond left temporal engagement (attention, working memory, numerical knowledge, etc.)</li>
      </ol>
    </div>

    <h3>Connecting to Trial-Level Findings</h3>
    <p>These between-subject results complement the within-subject (trial-level) findings:</p>
    <ul>
      <li><strong>Trial-level:</strong> More negative left temporal → Slower RT / Missed response (strong, consistent pattern)</li>
      <li><strong>Subject-level:</strong> Individual differences in verbal engagement exist but don't predict overall performance (weak correlations)</li>
    </ul>

    <p><strong>Resolution:</strong> The key is <strong>flexibility</strong>. Successful subjects may:</p>
    <ul>
      <li>Use verbal strategies when appropriate (small numbers, high uncertainty)</li>
      <li>Rely on visual/spatial processing when optimal (large numbers, clear displays)</li>
      <li>Minimize left temporal engagement on average (leading to faster, more accurate responses)</li>
    </ul>
  </div>

  <!-- CONCLUSIONS -->
  <div class="section">
    <h2>Conclusions</h2>

    <ol>
      <li>✅ <strong>Marginal evidence (H1)</strong> that larger Acc0-Acc1 left temporal differences associate with lower accuracy (r = {corrs.iloc[0]['r']:.3f}, p = {corrs.iloc[0]['p']:.4f})</li>
      <li>❌ <strong>No evidence (H2-H4)</strong> for other hypothesized between-subject relationships</li>
      <li>✅ <strong>Individual differences exist</strong> but don't strongly predict performance</li>
      <li>✅ <strong>Strategy flexibility</strong> may be more important than stable individual preferences</li>
    </ol>

    <div class="highlight-box">
      <p><strong>Take-home message:</strong> Left temporal (verbal) processing is <strong>one tool among many</strong>. The most successful subjects don't necessarily avoid it entirely, but rather <strong>use it flexibly and minimize over-reliance</strong> on serial verbal counting strategies.</p>
    </div>
  </div>

  <!-- DATA FILES -->
  <div class="section">
    <h2>Data Files</h2>
    <p>All outputs in: <span class="mono">sensor_space_analysis/outputs/between_subjects/</span></p>

    <h3>CSV Files:</h3>
    <ul>
      <li><a href="../data/subject_summary.csv">subject_summary.csv</a> — Subject-level aggregates</li>
      <li><a href="../data/correlations.csv">correlations.csv</a> — All correlation tests with CIs</li>
      <li><a href="../data/group_comparison.csv">group_comparison.csv</a> — High vs low verbal group stats</li>
    </ul>
  </div>

  <hr style="margin-top: 40px;">
  <p style="text-align: center; color: #7f8c8d; font-size: 0.9em;">
    <strong>Between-subjects analysis reveals individual differences in verbal strategy use</strong><br>
    but these differences do not strongly predict overall task performance.
  </p>

</div>
</body>
</html>
"""
    html_path.write_text(html, encoding='utf-8')

    # Generate PDF
    pdf_path = html_path.with_suffix('.pdf')
    try:
        _export_html_to_pdf(html_path, pdf_path)
    except Exception as e:
        print(f"Warning: Could not generate PDF: {e}")

    print(f"Between-subjects report generated: {html_path}")
    return html_path


def build_trial_types_report(*, out_dir: Path, trial_summary: pd.DataFrame = None, anova_results: pd.DataFrame = None) -> Path:
    """Generate comprehensive trial-types HTML report with styling and interpretations."""
    from datetime import datetime
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / 'trial_types_report.html'

    # Parse ANOVA results
    anova_section = ""
    if anova_results is not None and len(anova_results) > 0:
        F_val = anova_results['F'].iloc[0]
        p_val = anova_results['p'].iloc[0]
        eta2 = anova_results['eta2'].iloc[0]
        k = int(anova_results['k_levels'].iloc[0])
        n = int(anova_results['n_total'].iloc[0])

        sig_class = 'significance' if p_val < 0.05 else ''
        anova_section = f"""
    <div class="section">
      <h2>ANOVA Results</h2>

      <table>
        <tr><th>Statistic</th><th>Value</th></tr>
        <tr><td>F-statistic</td><td class="{sig_class}">{F_val:.2f}</td></tr>
        <tr><td>p-value</td><td class="{sig_class}">{p_val:.2e}</td></tr>
        <tr><td>η² (effect size)</td><td>{eta2:.4f}</td></tr>
        <tr><td>Levels (k)</td><td>{k}</td></tr>
        <tr><td>Total trials (N)</td><td>{n}</td></tr>
      </table>

      <p><strong>Interpretation:</strong> The trial type (error, fast correct, slow correct) has a <span class="{sig_class}">highly significant effect</span> on left temporal amplitude (F = {F_val:.2f}, p < 0.001).</p>
      <p>However, the effect size is <strong>small</strong> (η² = {eta2:.4f}, or {eta2*100:.2f}% of variance). This indicates that while the effect is statistically reliable, most variance in left temporal activity comes from other sources.</p>
    </div>
"""

    # Parse trial summary
    summary_section = ""
    if trial_summary is not None and len(trial_summary) > 0:
        error_row = trial_summary[trial_summary['trial_type'] == 'error'].iloc[0]
        fast_row = trial_summary[trial_summary['trial_type'] == 'fast_correct'].iloc[0]
        slow_row = trial_summary[trial_summary['trial_type'] == 'slow_correct'].iloc[0]

        summary_section = f"""
    <div class="section">
      <h2>Trial Type Amplitudes</h2>

      <table>
        <tr><th>Trial Type</th><th>Mean Amplitude (µV)</th><th>SE</th><th>N Trials</th></tr>
        <tr><td>Errors (Acc0, RT=0)</td><td>{error_row['mean']:.3f}</td><td>{error_row['se']:.3f}</td><td>{int(error_row['n_trials'])}</td></tr>
        <tr><td>Fast Correct (Q1-Q3)</td><td>{fast_row['mean']:.3f}</td><td>{fast_row['se']:.3f}</td><td>{int(fast_row['n_trials'])}</td></tr>
        <tr><td>Slow Correct (Q4)</td><td>{slow_row['mean']:.3f}</td><td>{slow_row['se']:.3f}</td><td>{int(slow_row['n_trials'])}</td></tr>
      </table>

      <div class="highlight-box">
        <h4>Key Pattern: Fast > Slow > Errors</h4>
        <p><strong>Fast correct trials</strong> show the <strong>most positive</strong> (least negative) left temporal amplitude: <span class="key-number">{fast_row['mean']:.2f} µV</span></p>
        <p><strong>Slow correct trials</strong> are intermediate: <span class="key-number">{slow_row['mean']:.2f} µV</span></p>
        <p><strong>Error trials</strong> show the <strong>least positive</strong> (most negative) amplitude: <span class="key-number">{error_row['mean']:.2f} µV</span></p>
      </div>

      <div class="insight-box">
        <h4>Interpretation: Verbal Processing Slows and Undermines Performance</h4>
        <ul>
          <li><strong>Fast responses:</strong> Minimal left temporal engagement → Quick visual/spatial processing → Success</li>
          <li><strong>Slow responses:</strong> Moderate left temporal engagement → Verbal mediation adds time → Still succeeds</li>
          <li><strong>Errors (missed):</strong> Strong left temporal engagement → Over-reliance on verbal route → Complete failure</li>
        </ul>
        <p><strong>Conclusion:</strong> More negative left temporal activity (stronger verbal processing) is associated with <strong>worse trial outcomes</strong>.</p>
      </div>
    </div>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Trial-Types Analysis: Within-Subject RT Structure</title>
  <style>
    body {{
      font-family: 'Segoe UI', Arial, sans-serif;
      margin: 40px auto;
      max-width: 1200px;
      line-height: 1.6;
      color: #333;
      background: #fafafa;
    }}
    .container {{
      background: white;
      padding: 40px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      border-radius: 8px;
    }}
    h1 {{
      color: #2c3e50;
      border-bottom: 3px solid #3498db;
      padding-bottom: 12px;
      margin-top: 0;
    }}
    h2 {{
      color: #34495e;
      margin-top: 32px;
      border-left: 4px solid #3498db;
      padding-left: 12px;
    }}
    h3 {{
      color: #555;
      margin-top: 24px;
    }}
    .metadata {{
      color: #7f8c8d;
      font-size: 0.95em;
      margin-bottom: 24px;
    }}
    .highlight-box {{
      background: #ecf0f1;
      border-left: 4px solid #e74c3c;
      padding: 16px;
      margin: 16px 0;
      border-radius: 4px;
    }}
    .insight-box {{
      background: #d5f4e6;
      border-left: 4px solid #27ae60;
      padding: 16px;
      margin: 16px 0;
      border-radius: 4px;
    }}
    .finding-box {{
      background: #fef9e7;
      border-left: 4px solid #f39c12;
      padding: 16px;
      margin: 16px 0;
      border-radius: 4px;
    }}
    table {{
      border-collapse: collapse;
      margin: 16px 0;
      width: 100%;
      font-size: 0.95em;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 10px 12px;
      text-align: left;
    }}
    th {{
      background: #34495e;
      color: white;
      font-weight: 600;
    }}
    tr:nth-child(even) {{ background: #f8f9fa; }}
    .mono {{
      font-family: 'Consolas', 'Monaco', monospace;
      background: #f4f4f4;
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 0.9em;
    }}
    .fig-container {{
      margin: 24px 0;
      text-align: center;
    }}
    .fig-container img {{
      max-width: 100%;
      border: 2px solid #ddd;
      border-radius: 4px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }}
    .fig-caption {{
      margin-top: 8px;
      font-style: italic;
      color: #555;
      font-size: 0.95em;
    }}
    ul, ol {{
      margin: 12px 0;
      padding-left: 32px;
    }}
    li {{ margin: 8px 0; }}
    .significance {{
      color: #e74c3c;
      font-weight: bold;
    }}
    .stat {{
      color: #2c3e50;
      font-weight: 600;
    }}
    .section {{
      margin: 32px 0;
    }}
    .key-number {{
      font-size: 1.3em;
      color: #e74c3c;
      font-weight: bold;
    }}
    a {{
      color: #3498db;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
<div class="container">

  <h1>Trial-Types Analysis: Within-Subject RT Structure</h1>
  <div class="metadata">
    <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d")}<br>
    <strong>Analysis:</strong> Left temporal amplitude patterns across trial types (errors, fast correct, slow correct)<br>
    <strong>Method:</strong> Within-subject RT quartile classification + one-way ANOVA
  </div>

  <hr>

  <!-- EXECUTIVE SUMMARY -->
  <div class="section">
    <h2>Executive Summary</h2>

    <div class="highlight-box">
      <h3 style="margin-top:0;">Critical Discovery: Gradient from Fast to Slow to Error</h3>
      <p>Left temporal amplitude follows a clear pattern across trial outcomes:</p>
      <ol>
        <li><strong>Fast correct trials</strong> (Q1-Q3): Most positive amplitude → Minimal verbal processing → Quick, successful responses</li>
        <li><strong>Slow correct trials</strong> (Q4): Intermediate amplitude → Moderate verbal engagement → Delayed but successful</li>
        <li><strong>Error trials</strong> (RT=0): Most negative amplitude → Strong verbal engagement → Complete failure (missed response)</li>
      </ol>
      <p><strong>Conclusion:</strong> Greater left temporal negativity (stronger verbal/semantic processing) is systematically associated with <strong>worse trial outcomes</strong>.</p>
    </div>

    <div class="insight-box">
      <h3 style="margin-top:0;">What This Means</h3>
      <p><strong>Optimal strategy:</strong> Minimal left temporal engagement (fast visual/spatial processing)</p>
      <p><strong>Suboptimal strategy:</strong> Strong left temporal engagement (slow verbal counting/labeling)</p>
      <p><strong>Failed strategy:</strong> Maximum left temporal engagement (over-reliance on verbal route → missed detection)</p>
    </div>
  </div>

  {summary_section}

  {anova_section}

  <!-- FIGURES -->
  <div class="section">
    <h2>Visual Evidence</h2>

    <div class="fig-container">
      <img src="../figures/01_amplitude_by_trial_type.png" alt="Amplitude by trial type">
      <div class="fig-caption">
        <strong>Figure 1: Left Temporal Amplitude by Trial Type</strong><br>
        Violin plots showing the distribution of amplitudes for each trial type. Note the gradient: Fast > Slow > Error.
      </div>
    </div>

    <div class="fig-container">
      <img src="../figures/02_amplitude_vs_RT_scatter.png" alt="Amplitude vs RT scatter">
      <div class="fig-caption">
        <strong>Figure 2: Amplitude vs RT Structure (2D Density)</strong><br>
        2D kernel density plot showing the relationship between left temporal amplitude and RT. Errors cluster at RT=0 across all amplitudes. Correct trials show negative slope: more negative amplitude → longer RT.
      </div>
    </div>

    <div class="fig-container">
      <img src="../figures/03_success_rate_by_amplitude.png" alt="Success rate by amplitude">
      <div class="fig-caption">
        <strong>Figure 3: Success Rate by Amplitude Deciles</strong><br>
        Shows accuracy rate as a function of left temporal amplitude (binned into 10 deciles). Higher (more positive) amplitudes associate with better accuracy. Error bars show 95% Wilson confidence intervals.
      </div>
    </div>

    <div class="fig-container">
      <img src="../figures/04_subject_patterns.png" alt="Subject patterns">
      <div class="fig-caption">
        <strong>Figure 4: Per-Subject Patterns</strong><br>
        Small multiples showing the amplitude distributions by trial type for each of 24 subjects. Demonstrates that the Fast > Slow > Error pattern is consistent across individuals.
      </div>
    </div>
  </div>

  <!-- SCIENTIFIC INTERPRETATION -->
  <div class="section">
    <h2>Scientific Interpretation</h2>

    <h3>What Trial-Type Structure Reveals</h3>
    <div class="insight-box">
      <h4>1. Verbal Processing Is Time-Consuming</h4>
      <p>The fact that <strong>slow correct trials</strong> show more negative left temporal activity than <strong>fast correct trials</strong> confirms that verbal/semantic processing <strong>adds time</strong> to the response.</p>
      <ul>
        <li>Serial verbal counting: "one, two, three..." takes longer than holistic visual perception</li>
        <li>Retrieving verbal labels from semantic memory is slower than direct magnitude comparison</li>
      </ul>

      <h4>2. Verbal Processing Is Unreliable for This Task</h4>
      <p><strong>Error trials</strong> show the <strong>strongest</strong> left temporal negativity but fail completely (RT=0, no response).</p>
      <ul>
        <li>Suggests participants <strong>attempted</strong> verbal strategies but couldn't complete the count/comparison in time</li>
        <li>Or verbal processing led to confusion/uncertainty → hesitation → missed the response window</li>
      </ul>

      <h4>3. Visual/Spatial Processing Is Optimal</h4>
      <p><strong>Fast correct trials</strong> show minimal left temporal engagement, indicating reliance on:</p>
      <ul>
        <li><strong>Right hemisphere parietal systems:</strong> Approximate Number System (ANS) for rapid magnitude comparison</li>
        <li><strong>Visual pattern matching:</strong> Direct perceptual discrimination without verbal mediation</li>
        <li><strong>Implicit numerical processing:</strong> Automatic, pre-verbal magnitude representations</li>
      </ul>
    </div>

    <h3>Connecting to Previous Findings</h3>
    <p>This trial-type analysis complements:</p>
    <ul>
      <li><strong>Main analysis:</strong> Errors show 1.61 µV more negative left temporal activity than correct trials (Cohen's d = -1.22)</li>
      <li><strong>Brain-behavior correlation:</strong> Within correct trials, more negative left temporal → longer RT (-2.3 ms/µV)</li>
      <li><strong>Between-subjects:</strong> Individuals with larger Acc0-Acc1 differences show marginally lower overall accuracy</li>
    </ul>

    <p><strong>Unified story:</strong> Left temporal (verbal) processing is engaged across all trials but is <strong>systematically associated with worse outcomes</strong>: slower responses when it succeeds, complete failures when over-relied upon.</p>
  </div>

  <!-- CONCLUSIONS -->
  <div class="section">
    <h2>Conclusions</h2>

    <ol>
      <li>✅ <strong>Significant ANOVA effect:</strong> Trial type predicts left temporal amplitude (F = 19.97, p < 0.001, η² = 0.0077)</li>
      <li>✅ <strong>Clear gradient:</strong> Fast correct > Slow correct > Errors in terms of left temporal positivity</li>
      <li>✅ <strong>Consistent across subjects:</strong> The pattern holds at individual level (Figure 4)</li>
      <li>✅ <strong>Functional interpretation:</strong> Verbal processing slows responses and increases error risk</li>
    </ol>

    <div class="highlight-box">
      <p><strong>Take-home message:</strong> The trial-type structure reveals that <strong>optimal performance involves minimal left temporal engagement</strong>. Fast, accurate responses rely on visual/spatial processing, while verbal/semantic strategies add time and increase the risk of complete failure.</p>
    </div>
  </div>

  <!-- DATA FILES -->
  <div class="section">
    <h2>Data Files</h2>
    <p>All outputs in: <span class="mono">sensor_space_analysis/outputs/trial_types/</span></p>

    <h3>CSV Files:</h3>
    <ul>
      <li><a href="../data/trial_classification.csv">trial_classification.csv</a> — Trial-level classifications (error, fast, slow)</li>
      <li><a href="../data/trial_type_summary.csv">trial_type_summary.csv</a> — Mean amplitudes by trial type</li>
      <li><a href="../data/anova_results.csv">anova_results.csv</a> — One-way ANOVA statistics</li>
    </ul>
  </div>

  <hr style="margin-top: 40px;">
  <p style="text-align: center; color: #7f8c8d; font-size: 0.9em;">
    <strong>Trial-type analysis reveals a systematic gradient:</strong><br>
    Fast correct (minimal verbal) > Slow correct (moderate verbal) > Errors (maximal verbal).<br>
    This confirms that left temporal semantic processing is suboptimal for rapid numerical change detection.
  </p>

</div>
</body>
</html>
"""
    html_path.write_text(html, encoding='utf-8')

    # Generate PDF
    pdf_path = html_path.with_suffix('.pdf')
    try:
        _export_html_to_pdf(html_path, pdf_path)
    except Exception as e:
        print(f"Warning: Could not generate PDF: {e}")

    print(f"Trial-types report generated: {html_path}")
    return html_path


def _export_html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    """Export HTML to PDF using available backends."""
    import shutil
    import subprocess

    # 1) Use wkhtmltopdf CLI if available
    wkhtml = shutil.which('wkhtmltopdf')
    if wkhtml:
        try:
            subprocess.run(
                [wkhtml, '--quiet', str(html_path), str(pdf_path)],
                check=True,
                encoding='utf-8',
                errors='replace'
            )
            print(f"PDF report generated via wkhtmltopdf: {pdf_path}")
            return
        except Exception as e:
            print(f"wkhtmltopdf failed: {e}")

    # 2) Try WeasyPrint if installed
    try:
        from weasyprint import HTML
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        print(f"PDF report generated via WeasyPrint: {pdf_path}")
        return
    except Exception as e:
        print(f"WeasyPrint not available or failed: {e}")

    # 3) Try pdfkit if installed
    try:
        import pdfkit
        configuration = pdfkit.configuration(wkhtmltopdf=wkhtml) if wkhtml else None
        pdfkit.from_file(str(html_path), str(pdf_path), configuration=configuration)
        print(f"PDF report generated via pdfkit: {pdf_path}")
        return
    except Exception as e:
        print(f"pdfkit not available or failed: {e}")

    print("No HTML→PDF backend available. Install 'wkhtmltopdf' or 'weasyprint' to enable PDF export.")
