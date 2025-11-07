import sys
from pathlib import Path

PROJECT_PARENT = Path(__file__).resolve().parents[1]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from sensor_space_analysis.src.data_extraction import aggregate_all_subjects
from sensor_space_analysis.src.between_subjects import (
    compute_subject_summary,
    compute_correlations,
    compute_group_comparison,
    generate_between_subjects_figures,
    save_between_subjects_outputs,
)
from sensor_space_analysis.src.report import build_between_subjects_report


def main():
    import pandas as pd
    root = Path.cwd()
    out_root = root / 'outputs' / 'between_subjects'
    out_data = out_root / 'data'
    out_figs = out_root / 'figures'
    out_reports = out_root / 'reports'
    for p in (out_data, out_figs, out_reports):
        p.mkdir(parents=True, exist_ok=True)

    # Load trial-level data from existing CSV
    trial_csv = root / 'outputs' / 'data' / 'trial_level_data.csv'
    print(f"Loading trial data from: {trial_csv}")
    df = pd.read_csv(trial_csv)

    # Compute subject summary and correlations
    summary = compute_subject_summary(df)
    corrs = compute_correlations(summary)
    group_comp = compute_group_comparison(summary)

    # Save data
    save_between_subjects_outputs(summary, corrs, group_comp, out_root)

    # Figures
    generate_between_subjects_figures(summary, out_figs)

    # Report
    build_between_subjects_report(summary=summary, corrs=corrs, out_dir=out_reports, group_comp=group_comp)
    print(f"Between-subject outputs written to: {out_root}")


if __name__ == '__main__':
    main()

