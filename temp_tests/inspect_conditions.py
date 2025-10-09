from pathlib import Path
import sys

project_root = Path(r"d:/numbers_eeg_source")
sys.path.insert(0, str(project_root))
from code.utils import data_loader

root = project_root / "data" / "data_preprocessed" / "hpf_1.0_lpf_35_baseline-on"
all_counts = {}
acc_counts = {}
for fif in sorted(root.glob('sub-*_preprocessed-epo.fif')):
    subj = fif.stem.split('_')[0]
    epochs = data_loader.load_subject_combined_epochs(subj.split('-')[-1], root)
    if epochs is None:
        continue
    md = epochs.metadata
    if md is None:
        continue
    for cond in md['Condition'].astype(str):
        all_counts.setdefault(cond, 0)
        all_counts[cond] += 1
    acc_mask = md['Target.ACC'].astype(float) >= 0.5
    for cond in md.loc[acc_mask, 'Condition'].astype(str):
        acc_counts.setdefault(cond, 0)
        acc_counts[cond] += 1

print('All counts (first 10):', {k: all_counts[k] for k in list(all_counts)[:10]})
print('Acc counts (conditions with hits):', acc_counts)
