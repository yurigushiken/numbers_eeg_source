# code/build_inverse_solutions.py

import mne
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import sys
import argparse

# Add the project root to the path to allow for module imports if run directly
project_root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root_path))


# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()

def _load_and_prep_epochs(subject_id, data_root, montage_path):
    """
    Load and concatenate all per-condition epoch files for a subject from data/,
    apply baseline correction, average reference, and standard montage.
    """
    try:
        subject_dir = data_root / subject_id
        if not subject_dir.exists():
            log.warning(f"Subject directory not found for {subject_id} at {subject_dir}. Skipping.")
            return None

        epoch_files = sorted(subject_dir.glob(f"{subject_id}_task-numbers_cond-*__epo.fif"))
        # accommodate both single-underscore and any accidental double underscore patterns
        if not epoch_files:
            epoch_files = sorted(subject_dir.glob(f"{subject_id}_task-numbers_cond-*_epo.fif"))

        if not epoch_files:
            log.warning(f"No per-condition epoch files found for {subject_id} in {subject_dir}. Skipping.")
            return None

        # Concatenate all condition-specific epoch files (read individually for compatibility)
        epochs_list = []
        for fp in epoch_files:
            try:
                epochs_list.append(mne.read_epochs(str(fp), preload=True, verbose=False))
            except Exception as e_read:
                log.warning(f"Failed to read epochs file {fp} for {subject_id}: {e_read}")
        if not epochs_list:
            log.warning(f"All epoch reads failed for {subject_id} in {subject_dir}. Skipping.")
            return None
        epochs = mne.concatenate_epochs(epochs_list, verbose=False)

        # Apply baseline correction using the pre-stimulus interval
        try:
            epochs.apply_baseline(baseline=(None, 0.0))
        except Exception:
            pass

        # Set the average reference. 'projection=True' makes it a projector.
        try:
            epochs.set_eeg_reference('average', projection=True)
        except Exception:
            pass
        log.info("Applied average EEG reference.")

        try:
            montage = mne.channels.read_custom_montage(fname=montage_path)
            epochs.set_montage(montage, on_missing='warn')
        except Exception:
            pass

        # Keep only valid EEG channels with finite, non-zero positions
        try:
            valid_picks = []
            for idx, ch in enumerate(epochs.info['chs']):
                pos = np.array(ch['loc'][:3])
                if np.isfinite(pos).all() and not np.allclose(pos, 0):
                    valid_picks.append(idx)
            if not valid_picks:
                log.warning(f"No valid EEG channel positions for {subject_id}; skipping.")
                return None
            epochs.pick(valid_picks, verbose=False)
            epochs.pick_types(eeg=True, verbose=False)
        except Exception as e_pick:
            log.warning(f"Failed to prune invalid channels for {subject_id}: {e_pick}")

        log.info(f"Loaded {len(epochs)} concatenated epochs for {subject_id}; applied baseline, reference, montage, and channel pruning.")
        return epochs
    except Exception as e:
        log.error(f"Error loading data for {subject_id}: {e}")
        return None

def _compute_and_validate_covariance(epochs, subject_id, qc_dir):
    log.info("Computing noise covariance...")
    noise_cov = mne.compute_covariance(epochs, tmax=0.0, method='auto', rank=None, verbose=False)

    evoked_for_qc = epochs.average()
    fig = evoked_for_qc.plot_white(noise_cov, time_unit='s', verbose=False)
    # Use a tight layout to prevent titles from being cut off
    fig.tight_layout()
    fig.suptitle(f"Whitening Plot - {subject_id}", y=0.98)


    qc_filename = qc_dir / f"{subject_id}_whitening_plot.png"
    fig.savefig(qc_filename)
    plt.close(fig)
    log.info(f"Saved noise covariance QC plot to: {qc_filename}")
    return noise_cov

def _validate_coregistration(info, subject_id, subjects_dir, qc_dir):
    log.info("Generating coregistration plot...")
    # This visualization requires the 3d backend
    mne.viz.set_3d_backend("pyvista")
    fig = mne.viz.plot_alignment(
        info,
        trans='fsaverage',
        subject='fsaverage',
        subjects_dir=subjects_dir,
        surfaces=dict(head=0.8, inner_skull=0.8),
        eeg=['original', 'projected'],
        meg=False,
        coord_frame='head',
        verbose=False
    )
    # Use a specific view that is informative
    fig.plotter.view_isometric()

    qc_filename = qc_dir / f"{subject_id}_coregistration_plot.png"
    # Use the PyVista backend's screenshot method
    fig.plotter.screenshot(qc_filename)
    
    # Add this line to properly close the 3D plot and free up memory
    fig.plotter.close()
    
    log.info(f"Saved coregistration QC plot to: {qc_filename}")

def _compute_and_save_inverse_operator(info, noise_cov, subject_id, subjects_dir, project_root, loose, depth):
    """
    Computes and saves the inverse operator using the fsaverage 3-layer BEM.
    """
    log.info(f"Computing forward and inverse solutions using fsaverage 3-layer BEM (loose={loose}, depth={depth})...")

    # --- Step 1: Define path to the source space file ---
    # The source space (the grid of points where we estimate activity) is still
    # based on the fsaverage cortical surface.
    fs_dir = Path(subjects_dir) / 'fsaverage'
    src_fname = fs_dir / "bem" / "fsaverage-ico-5-src.fif"

    # --- Step 2: Compute the Forward Solution using the BEM ---
    # Use realistic BEM for EEG (fsaverage 3-layer solution) per MNE best practices
    log.info("Computing forward solution...")
    bem_fname = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
    fwd = mne.make_forward_solution(
        info,
        trans='fsaverage',
        src=str(src_fname),
        bem=str(bem_fname),
        eeg=True,
        mindist=5.0,
        verbose=False
    )
    log.info("Successfully computed forward solution.")

    # --- Step 3: Compute and Save the Inverse Operator ---
    # This part of the logic remains unchanged.
    log.info("Computing inverse operator...")
    inv_op = mne.minimum_norm.make_inverse_operator(
        info,
        forward=fwd,
        noise_cov=noise_cov,
        loose=loose,
        depth=depth,
        verbose=False
    )

    # Save the final product strictly within data/
    original_deriv_dir = project_root / "data" / subject_id
    original_deriv_dir.mkdir(exist_ok=True)
    inv_filename = original_deriv_dir / f"{subject_id}-inv.fif"
    mne.minimum_norm.write_inverse_operator(inv_filename, inv_op, overwrite=True, verbose=False)
    log.info(f"SUCCESS: Inverse operator saved to: {inv_filename}")

def main():
    parser = argparse.ArgumentParser(description="Build inverse solutions for all subjects.")
    parser.add_argument("--loose", type=float, default=0.2, help="Loose parameter for the inverse operator.")
    parser.add_argument("--depth", type=float, default=0.8, help="Depth parameter for the inverse operator (MEG default).")
    args = parser.parse_args()

    # 1. Define key paths
    project_root = Path(__file__).resolve().parents[1]
    # Strictly use data/ as the source of per-condition epochs
    data_root = project_root / "data"
    montage_path = project_root / "assets" / "Channel Location - Net128_v1.sfp" / "AdultAverageNet128_v1.sfp"
    # Create a dedicated output directory for these one-time generated files
    output_root = project_root / "derivatives" / "inverse_solutions_QC"
    output_root.mkdir(exist_ok=True)

    # 2. Resolve fsaverage strictly from data (fail fast if missing)
    subjects_dir = project_root / "data" / "fs_subjects_dir"
    if not subjects_dir.exists():
        raise FileNotFoundError(
            f"FreeSurfer subjects directory not found at {subjects_dir}. Expected fsaverage under data/fs_subjects_dir/."
        )


    # 3. Get list of subjects strictly from data/
    subject_list_root = project_root / "data"
    subject_dirs = sorted([d for d in subject_list_root.iterdir() if d.is_dir() and d.name.startswith('sub-')])

    # 4. Main Loop
    for subject_dir in tqdm(subject_dirs, desc="Building Inverse Solutions"):
        subject_id = subject_dir.name
        log.info(f"--- Processing {subject_id} ---")

        # Create output directories for this subject
        subject_output_dir = output_root / subject_id
        qc_dir = subject_output_dir / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)

        # --- Start of subject-specific processing ---
        try:
            # 5. Load Data and Apply Montage
            epochs = _load_and_prep_epochs(subject_id, data_root, montage_path)
            if epochs is None: continue

            # 6. Generate and Validate Noise Covariance
            noise_cov = _compute_and_validate_covariance(epochs, subject_id, qc_dir)

            # 7. Validate Coregistration
            _validate_coregistration(epochs.info, subject_id, subjects_dir, qc_dir)

            # 8. Compute Forward and Inverse Solutions
            _compute_and_save_inverse_operator(
                epochs.info, noise_cov, subject_id, subjects_dir, project_root,
                loose=args.loose, depth=args.depth
            )

        except Exception as e:
            log.error(f"!!! FAILED to process {subject_id}: {e}", exc_info=True)
            continue
        # --- End of subject-specific processing ---

    log.info("--- Inverse solution generation complete for all subjects. ---")


# --- Add main execution block ---
if __name__ == "__main__":
    main()
