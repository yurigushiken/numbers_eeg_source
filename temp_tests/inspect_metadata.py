from pathlib import Path
import mne
import pandas as pd

project_root = Path(r"d:/numbers_eeg_source")
data_root = project_root / "data" / "data_preprocessed" / "hpf_1.0_lpf_35_baseline-on"
subject_columns = None
unique_values = {}

for fif_path in sorted(data_root.glob("sub-*_preprocessed-epo.fif"))[:3]:
    epochs = mne.read_epochs(fif_path, preload=False, verbose=False)
    metadata = epochs.metadata
    print(f"File: {fif_path.name}, n_epochs={len(epochs)}")
    if metadata is not None:
        if subject_columns is None:
            subject_columns = metadata.columns.tolist()
            print("columns:", subject_columns)
        for col in subject_columns:
            unique_values.setdefault(col, set()).update(metadata[col].dropna().unique().tolist())
    print()

print("Unique values snapshot:")
for col, values in unique_values.items():
    sample = list(values)[:5]
    print(f"{col} -> {sample}")

