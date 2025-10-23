# D:\numbers_eeg_nn_project\code\01_prepare_for_nn.py

import os
import re
import argparse
import pandas as pd
import mne
import numpy as np
from sklearn.preprocessing import LabelEncoder

"""
Convert HAPPE .set outputs to materialized MNE .fif epochs with behavioral metadata.

Supports CLI flags to process one or multiple dataset roots in a single run.
Example:
  python -X utf8 -u scripts/prepare_from_happe.py \
    --dataset-root data_input_from_happe/hpf_1.0_lpf_35_baseline-off \
    --dataset-root data_input_from_happe/hpf_1.0_lpf_35_baseline-on

Inputs:
- HAPPE EEGLAB .set files under a per-dataset "5 - processed" directory
- HAPPE QC CSV under "6 - quality_assessment_outputs" used to keep usable trials
- Behavior CSVs under data_behavior/data_UTF8/SubjectXX.csv

Outputs:
- data_preprocessed/<dataset>/sub-XX_preprocessed-epo.fif (with metadata)
- data_preprocessed/<dataset>/sub-XX_preprocessed-epo_metadata.csv
"""

# --- 1. CONFIGURATION (defaults; can be overridden by CLI) ---
PROCESS_ACC_ONLY = False  # default: keep all trials

# --- CONSTANT PATHS ---
BEHAVIORAL_DATA_DIR = os.path.join("data_behavior", "data_UTF8")


def _discover_happe_dirs(dataset_root: str):
    """Return (set_dir, usable_trials_csv_path) for a given HAPPE dataset root.

    Expects child folders named like '5 - processed*' and '6 - quality_assessment_outputs*'.
    """
    proc_dirs = [d for d in os.listdir(dataset_root) if d.lower().startswith("5 - processed")]
    if not proc_dirs:
        raise FileNotFoundError(f"No '5 - processed*' folder found under {dataset_root}")
    happe_set_dir = os.path.join(dataset_root, proc_dirs[0])

    qc_dirs = [d for d in os.listdir(dataset_root) if d.lower().startswith("6 - quality_assessment_outputs")]
    if not qc_dirs:
        raise FileNotFoundError(f"No '6 - quality_assessment_outputs*' folder found under {dataset_root}")
    qc_dir = os.path.join(dataset_root, qc_dirs[0])

    qc_csvs = [f for f in os.listdir(qc_dir) if f.lower().endswith(".csv") and f.lower().startswith("happe_dataqc")]
    if not qc_csvs:
        raise FileNotFoundError(f"No HAPPE_dataQC*.csv found under {qc_dir}")
    qc_csvs.sort()
    happe_usable_trials_file = os.path.join(qc_dir, qc_csvs[0])

    return happe_set_dir, happe_usable_trials_file


def parse_args():
    parser = argparse.ArgumentParser(description="Convert HAPPE .set to .fif with metadata")
    parser.add_argument(
        "--dataset-root",
        dest="dataset_roots",
        nargs="+",
        default=None,
        help=(
            "One or more HAPPE dataset roots containing '5 - processed' and "
            "'6 - quality_assessment_outputs' subfolders. If omitted, uses the "
            "default example: data_input_from_happe/hpf_1.0_lpf_45_baseline-on-example"
        ),
    )
    parser.add_argument(
        "--output-root",
        dest="output_root",
        default="data_preprocessed",
        help="Root directory where per-dataset outputs will be written",
    )
    parser.add_argument(
        "--acc-only",
        dest="acc_only",
        action="store_true",
        help="If set, keep only accurate trials (Target.ACC == 1)",
    )
    parser.add_argument(
        "--output-dataset-tag",
        dest="output_dataset_tag",
        default=None,
        help="Override the output dataset folder name under output_root (e.g., hpf_1.5_lpf_35_baseline-off-avgref)",
    )
    return parser.parse_args()


# --- 3. PRE-SCAN FOR ALL LABELS TO CREATE A GLOBAL ENCODER ---
# Ensures event codes are consistent across subjects when saving MNE events
print("--- Pass 1: Scanning for all unique labels ---")
all_labels = set()
subject_ids_in_folder = sorted([re.search(r'(?:Subject|Subj)(\d+)', f).group(1).zfill(2) for f in os.listdir(BEHAVIORAL_DATA_DIR) if re.search(r'(?:Subject|Subj)(\d+)', f)])

for subject_id in subject_ids_in_folder:
    try:
        behavioral_file = os.path.join(BEHAVIORAL_DATA_DIR, f"Subject{subject_id}.csv")
        behavioral_df = pd.read_csv(behavioral_file, on_bad_lines='warn', low_memory=False)
        behavioral_df['transition_label'] = behavioral_df['CellNumber'].astype(str)
        all_labels.update(behavioral_df['transition_label'].dropna().unique())
    except Exception as e:
        print(f"Could not process {behavioral_file} for label scanning: {e}")

global_le = LabelEncoder()
global_le.fit(sorted(list(all_labels)))
print(f"Found {len(global_le.classes_)} unique labels across all subjects.")
print("-" * 50)


# --- NEW: LABELING HELPERS (ported from 01_prepare_for_nn.py) ---
SMALL_SET = {1, 2, 3}
LARGE_SET = {4, 5, 6}


def direction_label(cond):
    try:
        s = str(int(cond)).zfill(2)
        prime, target = s[0], s[1]
    except (ValueError, TypeError):
        return pd.NA
    if prime == target:
        return "NC"
    return "I" if prime < target else "D"


def transition_category(cond):
    try:
        s = str(int(cond)).zfill(2)
        a, b = int(s[0]), int(s[1])
    except (ValueError, TypeError):
        return pd.NA
    if a == b:
        return "NC"
    if a in SMALL_SET and b in SMALL_SET:
        return "iSS" if a < b else "dSS"
    if a in LARGE_SET and b in LARGE_SET:
        return "iLL" if a < b else "dLL"
    if a in SMALL_SET and b in LARGE_SET:
        return "iSL"
    if a in LARGE_SET and b in SMALL_SET:
        return "dLS"
    return pd.NA


def size_category(cond):
    try:
        s = str(int(cond)).zfill(2)
        a, b = int(s[0]), int(s[1])
    except (ValueError, TypeError):
        return pd.NA
    if a == b:
        return "NC"
    a_small, b_small = a in SMALL_SET, b in SMALL_SET
    if a_small and b_small:
        return "SS"
    if not a_small and not b_small:
        return "LL"
    return "cross"
# --- END NEW HELPERS ---


def process_dataset(dataset_root: str, output_root: str, output_dataset_tag: str | None = None):
    print(f"\n=== DATASET: {dataset_root} ===")
    happe_set_dir, happe_usable_trials_file = _discover_happe_dirs(dataset_root)

    # Output goes to data_preprocessed/<dataset_tag>
    dataset_tag = output_dataset_tag or os.path.basename(dataset_root)
    output_dir = os.path.join(output_root, dataset_tag)
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Outputting to: {output_dir} ---")

    # Load QC usable trials table
    usable_trials_df = pd.read_csv(happe_usable_trials_file)
    usable_trials_df.rename(columns={usable_trials_df.columns[0]: 'SessionInfo'}, inplace=True)

    # --- MAIN PROCESSING LOOP ---
    print("\n--- Pass 2: Processing and saving individual subjects ---")
    for subject_id in subject_ids_in_folder:
        try:
            print(f"Processing Subject {subject_id}...")

            # Find the .set file for this subject (support a few common naming variants)
            candidate_paths = [
                os.path.join(happe_set_dir, f"Subject{subject_id}.set"),
                os.path.join(happe_set_dir, f"subject{subject_id}_processed.set"),
                os.path.join(happe_set_dir, f"subject{int(subject_id)}_processed.set"),
            ]
            set_file_path = next((p for p in candidate_paths if os.path.exists(p)), None)
            if set_file_path is None:
                raise FileNotFoundError(
                    f"Could not find .set for Subject {subject_id} in {happe_set_dir}. "
                    f"Tried: {', '.join(os.path.basename(p) for p in candidate_paths)}"
                )
            epochs = mne.io.read_epochs_eeglab(set_file_path, verbose=False)

            # Find subject row in QC file
            subject_id_int = int(subject_id)
            pattern = re.compile(fr"subject0*{subject_id_int}(?:\\.mff|_processed\\.set)?", re.IGNORECASE)
            subject_rows = usable_trials_df[usable_trials_df['SessionInfo'].str.match(pattern, na=False)]
            if subject_rows.empty:
                raise ValueError(f"No usable-trials row found for subject {subject_id} in {happe_usable_trials_file}")
            subject_row = subject_rows.iloc[0]
            raw_keep = str(subject_row['Kept_Segs_Indxs']).strip()
            if raw_keep.lower() == 'all':
                kept_indices_1based = None  # expand after Trial_Continuous is created
            else:
                kept_indices_1based = [int(i.strip()) for i in raw_keep.split(',') if i.strip().isdigit()]

            behavioral_file = os.path.join(BEHAVIORAL_DATA_DIR, f"Subject{subject_id}.csv")
            behavioral_df = pd.read_csv(behavioral_file, on_bad_lines='warn', low_memory=False)

            non_practice_mask = behavioral_df['Procedure[Block]'] != "Practiceproc"
            behavioral_df.loc[non_practice_mask, 'Trial_Continuous'] = np.arange(1, non_practice_mask.sum() + 1)

            behavioral_df['transition_label'] = behavioral_df['CellNumber'].astype(str)

            if kept_indices_1based is None:
                kept_indices_1based = behavioral_df.loc[non_practice_mask, 'Trial_Continuous'].astype(int).tolist()
            behavioral_df_filtered = behavioral_df[behavioral_df['Trial_Continuous'].isin(kept_indices_1based)].copy()

            # Remove trials with Condition == 99
            valid_condition_mask = behavioral_df_filtered['CellNumber'].astype(str) != '99'
            epochs = epochs[valid_condition_mask]
            behavioral_df_filtered = behavioral_df_filtered[valid_condition_mask]

            # Create unified accuracy column from all Target*.ACC columns
            # Rationale: spreadsheets may have multiple ACC columns; we coalesce left-to-right
            acc_cols = [col for col in behavioral_df_filtered.columns if 'ACC' in col and 'Target' in col and 'OverallAcc' not in col]
            if acc_cols:
                for col in acc_cols:
                    behavioral_df_filtered[col] = pd.to_numeric(behavioral_df_filtered[col], errors='coerce')
                unified_acc = behavioral_df_filtered[acc_cols[0]]
                for i in range(1, len(acc_cols)):
                    unified_acc = unified_acc.fillna(behavioral_df_filtered[acc_cols[i]])
                behavioral_df_filtered['unified_ACC'] = unified_acc
            else:
                behavioral_df_filtered['unified_ACC'] = behavioral_df_filtered['Target.ACC']

            if len(epochs) != len(behavioral_df_filtered):
                raise ValueError(
                    f"FATAL MISMATCH for Subject {subject_id}: Num epochs ({len(epochs)}) != "
                    f"Num behavioral trials ({len(behavioral_df_filtered)})."
                )

            # Build final metadata expected by downstream trainer (see README Data model)
            behavioral_df_filtered['direction'] = behavioral_df_filtered['CellNumber'].apply(direction_label)
            behavioral_df_filtered['change_group'] = behavioral_df_filtered['CellNumber'].apply(transition_category)
            behavioral_df_filtered['size'] = behavioral_df_filtered['CellNumber'].apply(size_category)

            final_metadata = pd.DataFrame({
                'SubjectID': subject_id,
                'Block': behavioral_df_filtered['Block'],
                'Trial': behavioral_df_filtered['Trial'],
                'Procedure': behavioral_df_filtered['Procedure[Block]'],
                'Condition': behavioral_df_filtered['CellNumber'],
                'Target.ACC': behavioral_df_filtered['unified_ACC'],
                'Target.RT': behavioral_df_filtered['Target.RT'],
                'Trial_Continuous': behavioral_df_filtered['Trial_Continuous'],
                'direction': behavioral_df_filtered['direction'],
                'change_group': behavioral_df_filtered['change_group'],
                'size': behavioral_df_filtered['size']
            })

            encoded_labels = global_le.transform(behavioral_df_filtered['transition_label'])
            events_from_metadata = np.array([
                np.arange(len(behavioral_df_filtered)),
                np.zeros(len(behavioral_df_filtered), int),
                encoded_labels
            ]).T

            _ = {label: i for i, label in enumerate(global_le.classes_)}

            epochs_with_metadata = mne.EpochsArray(
                epochs.get_data(), info=epochs.info, events=events_from_metadata,
                tmin=epochs.tmin, event_id=None, metadata=final_metadata
            )

            if PROCESS_ACC_ONLY:
                final_epochs = epochs_with_metadata[epochs_with_metadata.metadata['Target.ACC'] == 1]
                print(f"  Keeping {len(final_epochs)} accurate trials.")
            else:
                final_epochs = epochs_with_metadata
                print(f"  Keeping all {len(final_epochs)} trials.")

            # Attach 3D montage (REQUIRED for spatial analyses: cz_step, XAI topomaps)
            # FAIL-FAST: Must succeed for scientific validity
            try:
                from mne.channels import read_custom_montage
                mont = read_custom_montage("net/AdultAverageNet128_v1.sfp")
                final_epochs.set_montage(mont, match_case=False, match_alias=True, on_missing="warn")
                
                # CRITICAL: Verify montage was actually attached
                attached_montage = final_epochs.get_montage()
                if attached_montage is None:
                    raise RuntimeError("Montage attachment failed - get_montage() returned None")
                
                # Verify Cz exists (required for cz_step)
                pos = attached_montage.get_positions()
                if 'Cz' not in pos.get('ch_pos', {}):
                    raise RuntimeError("Montage missing Cz channel - cz_step will not work")
                
                print(f"  [montage] OK: {len(attached_montage.ch_names)} channels, Cz present")
                
                # Apply average rereferencing using all EEG channels (no projections)
                try:
                    final_epochs = final_epochs.copy().set_eeg_reference('average', projection=False, verbose=False)
                    print("  [reref] Applied average reference across EEG channels")
                except Exception as e_rr:
                    print(f"  [reref] WARNING: Failed to apply average reference ({e_rr}). Proceeding without reref.")
                
            except Exception as e:
                print(f"  [montage] CRITICAL ERROR: {e}")
                raise RuntimeError(f"Cannot proceed without valid montage for subject {subject_id}") from e

            output_filename = f"sub-{subject_id}_preprocessed-epo.fif"
            output_path = os.path.join(output_dir, output_filename)
            final_epochs.save(output_path, overwrite=True, verbose=False)

            print(f"  -> Saved to {output_path}")

            # Also export per-subject metadata to CSV for manual verification
            try:
                metadata_csv_filename = f"sub-{subject_id}_preprocessed-epo_metadata.csv"
                metadata_csv_path = os.path.join(output_dir, metadata_csv_filename)
                final_epochs.metadata.to_csv(metadata_csv_path, index=False)
                print(f"  -> Wrote metadata CSV to {metadata_csv_path}")
            except Exception as _e_csv:
                print(f"[csv] failed to write metadata for subject {subject_id} ({_e_csv})")

        except Exception as e:
            import traceback
            print(f"!!! FAILED for subject {subject_id}: {e}")
            traceback.print_exc()
            continue

    print("\n--- DATA PREPARATION COMPLETE ---")


if __name__ == "__main__":
    args = parse_args()

    if args.acc_only:
        PROCESS_ACC_ONLY = True

    dataset_roots = args.dataset_roots or [
        os.path.join("data_input_from_happe", "hpf_1.0_lpf_45_baseline-on-example")
    ]

    for root in dataset_roots:
        process_dataset(root, args.output_root, args.output_dataset_tag)


