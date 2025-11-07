import sys
from pathlib import Path

PROJECT_PARENT = Path(__file__).resolve().parents[1]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from sensor_space_analysis.src.data_extraction import aggregate_all_subjects
from sensor_space_analysis.src.trial_types import (
    classify_trials_by_rt,
    summarize_by_trial_type,
    anova_trial_type,
    generate_trial_type_figures,
    save_trial_type_outputs,
)
from sensor_space_analysis.src.report import build_trial_types_report


def main():
    import pandas as pd
    root = Path.cwd()
    out_root = root / 'outputs' / 'trial_types'
    out_data = out_root / 'data'
    out_figs = out_root / 'figures'
    out_reports = out_root / 'reports'
    for p in (out_data, out_figs, out_reports):
        p.mkdir(parents=True, exist_ok=True)

    # Load trial-level data from existing CSV
    trial_csv = root / 'outputs' / 'data' / 'trial_level_data.csv'
    print(f"Loading trial data from: {trial_csv}")
    df = pd.read_csv(trial_csv)

    trials = classify_trials_by_rt(df)
    summ = summarize_by_trial_type(trials)
    anova_df = anova_trial_type(trials)

    save_trial_type_outputs(trials, summ, anova_df, out_root)
    generate_trial_type_figures(trials, out_figs)
    build_trial_types_report(out_dir=out_reports, trial_summary=summ, anova_results=anova_df)
    print(f"Trial-type outputs written to: {out_root}")


if __name__ == '__main__':
    main()

