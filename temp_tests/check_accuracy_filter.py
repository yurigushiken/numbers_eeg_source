from pathlib import Path
import sys

project_root = Path(r"d:/numbers_eeg_source")
sys.path.insert(0, str(project_root))
from code.utils import data_loader

data_root = project_root / "data" / "data_preprocessed" / "hpf_1.0_lpf_35_baseline-on"
subject_dir = data_root / "sub-02"
cond_info = {"condition_set_name": "CARDINALITY_1"}

all_evoked, all_epochs = data_loader.get_evoked_for_condition(subject_dir, cond_info, accuracy="all")
acc_evoked, acc_epochs = data_loader.get_evoked_for_condition(subject_dir, cond_info, accuracy="acc1")

print(f"All trials epochs count: {[len(epo) for epo in all_epochs] if all_epochs else []}")
print(f"Acc1 trials epochs count: {[len(epo) for epo in acc_epochs] if acc_epochs else []}")
print(f"All trials available: {bool(all_evoked)}; Acc1 available: {bool(acc_evoked)}")

combined = data_loader.load_subject_combined_epochs('02', data_root)
if combined is not None and combined.metadata is not None:
    print("\nAccuracy distribution (Target.ACC):")
    print(combined.metadata['Target.ACC'].value_counts())
