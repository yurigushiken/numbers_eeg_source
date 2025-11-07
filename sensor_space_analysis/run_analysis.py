import os
import sys
from pathlib import Path

# Ensure project root (parent of this folder) is importable when executed as a script
PROJECT_PARENT = Path(__file__).resolve().parents[1]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from sensor_space_analysis.src.data_extraction import aggregate_all_subjects
from sensor_space_analysis.src.preprocessing import (
    identify_rt_outliers,
    subject_level_summary,
    validate_trial_counts,
    center_within_subject,
)
from sensor_space_analysis.src.modeling import (
    build_mixed_model,
    fit_mixed_model,
    extract_fixed_effects,
    extract_random_effects,
    model_diagnostics,
    generate_predictions,
)
from sensor_space_analysis.src.visualization import (
    create_scatter_plot,
    create_spaghetti_plot,
    create_interaction_plot,
    create_forest_plot,
    create_subject_heatmap,
    create_distribution_plots,
)
from sensor_space_analysis.src.report import build_analysis_report


def main():
    root = Path.cwd() / 'sensor_space_analysis'
    outputs = root / 'outputs'
    out_data = outputs / 'data'
    out_figs = outputs / 'figures'
    out_reports = outputs / 'reports'
    for p in (out_data, out_figs, out_reports):
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Extracting trial-level data...")
    data_dir = Path.cwd() / 'data' / 'data_preprocessed' / 'hpf_1.5_lpf_35_baseline-on'
    trial_csv = out_data / 'trial_level_data.csv'
    df = aggregate_all_subjects(data_dir, trial_csv)

    print("=" * 60)
    print("STEP 2: Preprocessing and validation...")
    _ = identify_rt_outliers(df)
    subj_sum = subject_level_summary(df)
    subj_sum.to_csv(out_data / 'subject_level_summary.csv', index=False)
    validate_trial_counts(df, min_trials=1)  # do not exclude; just validate
    df = center_within_subject(df, columns=['left_temp_amp'])

    print("=" * 60)
    print("STEP 3: Fitting mixed-effects model...")
    md = build_mixed_model(df)
    res = fit_mixed_model(md)
    fx = extract_fixed_effects(res)
    rx = extract_random_effects(res)
    fx.to_csv(out_data / 'model_fixed_effects.csv', index=False)
    rx.to_csv(out_data / 'model_random_effects.csv', index=False)
    with open(out_data / 'model_diagnostics.txt', 'w', encoding='utf-8') as f:
        f.write(model_diagnostics(res))
    preds = generate_predictions(df, res)

    print("=" * 60)
    print("STEP 4: Generating visualizations...")
    create_scatter_plot(df, out_figs / '01_trial_scatter.png')
    create_spaghetti_plot(df, out_figs / '02_spaghetti_plot.png')
    create_interaction_plot(preds, out_figs / '03_interaction_plot.png')
    create_forest_plot(rx, out_figs / '04_forest_plot.png')
    create_subject_heatmap(subj_sum, out_figs / '05_subject_heatmap.png')
    create_distribution_plots(df, out_figs / '06_distributions.png')

    print("=" * 60)
    print("STEP 5: Building HTML report...")
    report_html = outputs / 'reports' / 'analysis_report.html'
    build_analysis_report(
        df=df,
        subj_summary=subj_sum,
        model_result=res,
        fixed_effects_csv=(out_data / 'model_fixed_effects.csv'),
        random_effects_csv=(out_data / 'model_random_effects.csv'),
        diagnostics_txt=(out_data / 'model_diagnostics.txt'),
        figures_dir=out_figs,
        output_html=report_html,
    )

    print("=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {outputs}")
    print(f"Report: {report_html}")


if __name__ == "__main__":
    main()
