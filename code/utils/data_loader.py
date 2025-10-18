"""
Data Loading Utilities
"""
import logging
import os
from pathlib import Path
import yaml
import mne
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, write_inverse_operator
import mne.channels

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Lightweight config helpers (no heavy deps)
try:
    from code.utils.config_utils import (
        load_common_defaults,
        merge_common_into_config,
    )
except Exception:
    # Safe fallback if tests import only parts
    def load_common_defaults(project_root="."):
        return {}
    def merge_common_into_config(common, cfg):
        return cfg

# --- Load Standard Montage on Module Import ---
# As per the lab's finding, individual digitization was not performed.
# Therefore, we MUST load and enforce a standard montage for all subjects to
# ensure a consistent forward model. The 'dig' info from the FIF files is ignored.
try:
    # Use montage under project-root assets
    EGI_128_MONTAGE_PATH = Path(__file__).resolve().parents[2] / 'assets' / 'Channel Location - Net128_v1.sfp' / 'AdultAverageNet128_v1.sfp'
    STANDARD_MONTAGE = mne.channels.read_custom_montage(fname=EGI_128_MONTAGE_PATH)
    log.info(f"Loaded standard EGI 128 montage from: {EGI_128_MONTAGE_PATH}")
except FileNotFoundError:
    log.error(f"CRITICAL: Standard montage file not found at {EGI_128_MONTAGE_PATH}. Pipeline cannot proceed.")
    STANDARD_MONTAGE = None


# --- Load Condition Sets on Module Import ---
# This mimics the structure of the original utils
def _load_condition_sets():
    """Loads the centralized condition set definitions from the YAML file."""
    # Use the condition sets inside this project (code/condition_sets.yaml)
    sets_path = Path(__file__).resolve().parents[1] / 'condition_sets.yaml'
    with open(sets_path, 'r') as f:
        return yaml.safe_load(f)

CONDITION_SETS = _load_condition_sets()
# ---

def load_common_config(project_root: str | Path = "."):
    """Load the common configuration file (configs/common.yaml).

    Parameters
    ----------
    project_root : str | Path
        Project root directory

    Returns
    -------
    dict
        Common configuration dictionary
    """
    common_path = Path(project_root) / "configs" / "common.yaml"
    if common_path.exists():
        with open(common_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

def load_config(config_path, project_root: str | Path = "."):
    """Loads and validates a YAML configuration file and applies common defaults.

    - If configs/common.yaml exists, apply minimal defaults (e.g., baseline),
      without overwriting analysis-specific values.
    """
    log.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    common = load_common_defaults(project_root)
    cfg = merge_common_into_config(common, cfg)
    log.info("Configuration loaded successfully (common defaults applied if present).")
    return cfg


def get_subject_dirs(accuracy, project_root=".", data_source=None):
    """
    Finds all subject directories (e.g., 'sub-02') for a given accuracy dataset.

    Parameters
    ----------
    accuracy : str
        'all' for all trials, 'acc1' for accurate trials only
    project_root : str
        Project root directory (default: ".")
    data_source : str | None
        None or 'new' (combined preprocessed files; default behavior), or a custom path like
        'data/data_preprocessed/hpf_1.5_lpf_40_baseline-on'.

    Returns
    -------
    subject_dirs : list of Path
        List of subject directory paths
    """
    project_root = Path(project_root)

    # Determine data directory based on data_source
    if data_source in (None, "new"):
        # Determine preprocessing variant from common.yaml (if present)
        common = load_common_defaults(project_root)
        preprocessing = ((common.get("data") or {}).get("preprocessing")
                         or "hpf_1.0_lpf_35_baseline-on")
        data_dir = project_root / "data" / "data_preprocessed" / preprocessing
        log.info(f"Using combined preprocessed data: {data_dir}")

    elif data_source == "old":
        raise ValueError("Legacy 'old' data pipeline is no longer supported.")

    else:
        # CUSTOM: User-specified path (e.g., different preprocessing parameters)
        data_dir = project_root / data_source
        log.info(f"Using CUSTOM data path: {data_dir}")

    # For combined files, we need to find .fif files and create pseudo-directories
    if data_source in (None, "new") or "data_preprocessed" in str(data_dir):
        # Look for combined .fif files
        fif_files = sorted(list(data_dir.glob("sub-*_preprocessed-epo.fif")))

        if not fif_files:
            log.warning(f"No combined .fif files found in {data_dir}")
            return []

        # Create pseudo-directories (path with subject name)
        # The loader will extract subject ID and find the .fif file
        subject_dirs = []
        for fif_file in fif_files:
            # Extract 'sub-02' from 'sub-02_preprocessed-epo.fif'
            subject_name = fif_file.stem.split('_')[0]  # 'sub-02'
            subject_path = data_dir / subject_name
            subject_dirs.append(subject_path)

        log.info(f"Found {len(subject_dirs)} subjects with combined epoch files in {data_dir}")
        return subject_dirs

    else:
        # If a custom path does not look like combined preprocessed data, still search for combined files
        fif_files = sorted(list(data_dir.glob("sub-*_preprocessed-epo.fif")))
        subject_dirs = [(data_dir / ff.stem.split('_')[0]) for ff in fif_files]
        log.info(f"Found {len(subject_dirs)} subjects (custom combined path) in {data_dir}.")
        return subject_dirs


def load_subject_combined_epochs(subject_id, data_root):
    """
    Load combined epoch file with metadata for a single subject.

    This function supports the new data structure where all conditions
    are stored in a single .fif file with rich metadata.

    Parameters
    ----------
    subject_id : str
        Subject ID (e.g., '02' for sub-02)
    data_root : Path or str
        Root directory containing combined epoch files
        Expected file: <data_root>/sub-<subject_id>_preprocessed-epo.fif

    Returns
    -------
    epochs : mne.Epochs | None
        Epochs object with metadata, or None if file not found

    Examples
    --------
    >>> epochs = load_subject_combined_epochs(
    ...     subject_id='02',
    ...     data_root=Path('data/data_preprocessed/hpf_1.0_lpf_35_baseline-on')
    ... )
    >>> # Filter by condition using metadata
    >>> epochs_cond11 = epochs[epochs.metadata['Condition'] == '11']
    """
    data_root = Path(data_root)
    fif_path = data_root / f"sub-{subject_id}_preprocessed-epo.fif"

    if not fif_path.exists():
        log.debug(f"Combined epoch file not found: {fif_path}")
        return None

    try:
        log.debug(f"Loading combined epochs from {fif_path}")
        epochs = mne.read_epochs(fif_path, preload=True, verbose=False)

        # Verify metadata exists
        if epochs.metadata is None:
            log.warning(f"Loaded epochs from {fif_path} but metadata is missing!")
            return None

        # Verify 'Condition' column exists
        if 'Condition' not in epochs.metadata.columns:
            log.warning(f"Loaded epochs from {fif_path} but 'Condition' column missing in metadata!")
            return None

        log.debug(f"Successfully loaded {len(epochs)} epochs with metadata")
        return epochs

    except Exception as e:
        log.error(f"Error loading combined epochs from {fif_path}: {e}")
        return None


def load_and_concatenate_subject_epochs(subject_dir):
    """
    Loads all condition-specific epoch files for a subject and concatenates them.
    """
    epoch_files = sorted(list(subject_dir.glob("*-epo.fif")))
    if not epoch_files:
        log.warning(f"No epoch files found for subject {subject_dir.name}")
        return None

    log.debug(f"Loading and concatenating {len(epoch_files)} epoch files for {subject_dir.name}...")
    try:
        # MNE's read_epochs can take a list of files to concatenate them
        epochs = mne.read_epochs(epoch_files, preload=True, verbose=False)
        return epochs
    except Exception as e:
        log.error(f"Error reading epochs for {subject_dir.name}: {e}")
        return None


def create_subject_contrast(subject_dir, config, accuracy: str = "all"):
    """
    Creates a contrast between two conditions for a single subject.

    Parameters
    ----------
    subject_dir : Path
        Location of the subject's data (combined or split structure).
    config : dict
        Parsed analysis configuration.
    accuracy : str
        'all' for all trials, 'acc1' to include only accurate trials (requires metadata).

    Returns
    -------
    tuple
        (contrast_evoked, concatenated_epochs). Returns (None, None) if data cannot be loaded.
    """
    log.debug(f"Creating contrast for subject {subject_dir.name}")
    try:
        # Use unified epoch window configuration (baseline is required there)
        epoch_cfg = config.get('epoch_window', {})
        baseline = epoch_cfg.get('baseline')

        # Allow per-side accuracy override from config
        acc_A = (config.get('contrast', {}).get('condition_A', {}) or {}).get('accuracy') or accuracy
        evoked_A, epochs_A = get_evoked_for_condition(
            subject_dir,
            config['contrast']['condition_A'],
            baseline=baseline,
            accuracy=acc_A,
        )
        acc_B = (config.get('contrast', {}).get('condition_B', {}) or {}).get('accuracy') or accuracy
        evoked_B, epochs_B = get_evoked_for_condition(
            subject_dir,
            config['contrast']['condition_B'],
            baseline=baseline,
            accuracy=acc_B,
        )

        if not evoked_A or not evoked_B:
            return None, None

        # Average the evoked responses for each condition set
        grand_average_A = mne.grand_average(evoked_A)
        grand_average_B = mne.grand_average(evoked_B)

        # Create the contrast
        contrast_evoked = mne.combine_evoked(
            [grand_average_A, grand_average_B],
            weights=config['contrast']['combination_weights']
        )
        log.debug(f"Successfully created contrast for {subject_dir.name}")

        # Concatenate all epochs for covariance calculation
        all_epochs = mne.concatenate_epochs(epochs_A + epochs_B, verbose=False)

        return contrast_evoked, all_epochs

    except Exception as e:
        log.error(f"Error creating contrast for {subject_dir.name}: {e}")
        return None, None


def get_evoked_for_condition(subject_dir, condition_info, baseline=None, use_combined=True, accuracy: str = "all"):
    """
    Loads all epoch files for a given condition set and averages them.

    Operates on combined files (with metadata).

    Parameters
    ----------
    subject_dir : Path
        Subject directory path. For combined files, this can be the data root.
        For split files, this is the subject-specific directory.
    condition_info : dict
        Dictionary with 'condition_set_name' key
    baseline : tuple | None
        Baseline correction period (tmin, tmax) or None
    use_combined : bool
        If True (default), load from combined file with metadata.
    accuracy : str
        'all' for all trials, 'acc1' for accurate-only filtering (requires metadata).

    Returns
    -------
    evoked_list : list of mne.Evoked
        List of Evoked objects for each condition
    epochs_list : list of mne.Epochs
        List of Epochs objects for each condition
    """
    condition_set_name = condition_info['condition_set_name']
    condition_set = CONDITION_SETS.get(condition_set_name)
    if not condition_set:
        log.warning(f"Condition set '{condition_set_name}' not found.")
        return [], []

    # The values of the condition set dict are the lists of condition numbers.
    # We need to flatten them into a single list.
    condition_numbers = [num for sublist in condition_set.values() for num in sublist]

    evoked_list = []
    epochs_list = []

    # Combined file approach (only supported mode)
    if use_combined:
        # Extract subject ID from path
        # Could be: data_preprocessed/hpf_1.0_lpf_35_baseline-on (data_root)
        if 'sub-' in str(subject_dir):
            # Extract from a path that includes a subject directory name like "sub-02"
            subject_id = str(subject_dir).split('sub-')[-1].split('/')[0].split('\\')[0]
        else:
            # Assume subject_dir IS the data_root, will try later
            subject_id = None

        # Try to find combined file
        combined_epochs = None
        if subject_id:
            # Try in the provided directory
            combined_epochs = load_subject_combined_epochs(subject_id, subject_dir)

            # If not found and subject_dir looks like a subject dir, try parent
            if combined_epochs is None and 'sub-' in str(subject_dir):
                parent_dir = subject_dir.parent
                combined_epochs = load_subject_combined_epochs(subject_id, parent_dir)

        if combined_epochs is not None:
            log.debug(f"Using combined file approach for {condition_set_name}")

            accuracy_mask = None
            acc_flag = (accuracy or "all").lower()
            if acc_flag in {"acc1", "acc0"}:
                metadata = combined_epochs.metadata
                if metadata is not None and 'Target.ACC' in metadata.columns:
                    try:
                        acc_series = metadata['Target.ACC'].astype(float)
                        if acc_flag == "acc1":
                            accuracy_mask = acc_series >= 0.5
                        else:
                            accuracy_mask = acc_series < 0.5
                        if accuracy_mask.sum() == 0:
                            log.warning(
                                "Accuracy filtering removed all epochs for %s (%s); using all trials instead.",
                                subject_dir,
                                condition_set_name,
                            )
                            # Fallback: do not filter by accuracy if it removes everything
                            accuracy_mask = None
                    except Exception as exc:
                        log.warning(
                            "Failed to interpret Target.ACC metadata for %s: %s; using all trials.",
                            subject_dir,
                            exc,
                        )
                        accuracy_mask = None
                else:
                    log.warning(
                        "Target.ACC metadata not found for %s; unable to filter accurate trials.",
                        subject_dir,
                    )

            # Filter epochs by condition codes using metadata
            for cond_num in condition_numbers:
                # Filter by metadata - condition codes might be int or str
                try:
                    # Try as string first (most common)
                    cond_mask = combined_epochs.metadata['Condition'].astype(str) == str(cond_num)
                except:
                    # Try as int
                    cond_mask = combined_epochs.metadata['Condition'] == int(cond_num)

                mask = cond_mask
                if accuracy_mask is not None:
                    mask = cond_mask & accuracy_mask

                if not mask.any():
                    log.debug(
                        "No epochs remaining for condition %s after accuracy filtering.",
                        cond_num,
                    )
                    continue

                epochs_cond = combined_epochs[mask]

                # Make a copy to avoid modifying original
                epochs_cond = epochs_cond.copy()

                # Apply baseline correction if provided
                if baseline is not None:
                    try:
                        epochs_cond.apply_baseline(baseline=tuple(baseline), verbose=False)
                    except Exception:
                        epochs_cond.apply_baseline(baseline=(baseline[0], baseline[1]), verbose=False)

                # Ensure average EEG reference is applied
                if not any(p['desc'] == 'Average EEG reference' for p in epochs_cond.info['projs']):
                    epochs_cond.set_eeg_reference('average', projection=True, verbose=False)

                # Enforce the standard montage
                if STANDARD_MONTAGE:
                    epochs_cond.set_montage(STANDARD_MONTAGE, on_missing='warn')

                epochs_list.append(epochs_cond)

                # Create evoked
                evoked = epochs_cond.average()
                if not any(p['desc'] == 'Average EEG reference' for p in evoked.info['projs']):
                    evoked.set_eeg_reference('average', projection=True, verbose=False)
                evoked_list.append(evoked)

            if evoked_list:
                log.debug(f"Successfully loaded {len(evoked_list)} conditions from combined file")
                return evoked_list, epochs_list

            # If accuracy filtering produced no data overall, retry without accuracy filter
            if acc_flag == "acc1":
                log.debug("No epochs after accuracy filtering; retrying without accuracy mask for acc1")
                evoked_list = []
                epochs_list = []
                for cond_num in condition_numbers:
                    try:
                        cond_mask = combined_epochs.metadata['Condition'].astype(str) == str(cond_num)
                    except:
                        cond_mask = combined_epochs.metadata['Condition'] == int(cond_num)

                    if not cond_mask.any():
                        continue

                    epochs_cond = combined_epochs[cond_mask].copy()
                    if baseline is not None:
                        try:
                            epochs_cond.apply_baseline(baseline=tuple(baseline), verbose=False)
                        except Exception:
                            epochs_cond.apply_baseline(baseline=(baseline[0], baseline[1]), verbose=False)
                    if not any(p['desc'] == 'Average EEG reference' for p in epochs_cond.info['projs']):
                        epochs_cond.set_eeg_reference('average', projection=True, verbose=False)
                    if STANDARD_MONTAGE:
                        epochs_cond.set_montage(STANDARD_MONTAGE, on_missing='warn')
                    epochs_list.append(epochs_cond)
                    evoked = epochs_cond.average()
                    if not any(p['desc'] == 'Average EEG reference' for p in evoked.info['projs']):
                        evoked.set_eeg_reference('average', projection=True, verbose=False)
                    evoked_list.append(evoked)

                if evoked_list:
                    return evoked_list, epochs_list

    # If we reach here, combined epochs were not found or yielded no data
    log.warning(f"No epochs loaded for condition set {condition_set_name} using combined files")
    return [], []


def get_inverse_operator(subject_dir):
    """
    Reads the inverse operator for a given subject.
    """
    inv_fname = subject_dir / f"{subject_dir.name}-inv.fif"
    log.info(f"Reading inverse operator from {inv_fname}")
    return mne.minimum_norm.read_inverse_operator(inv_fname, verbose=False)


def get_fsaverage_src(project_root="."):
    """
    Gets the fsaverage source space strictly from data.
    """
    # Prefer MNE's configured SUBJECTS_DIR or fetch fsaverage if needed
    subjects_dir = mne.get_config('SUBJECTS_DIR')
    if subjects_dir is None:
        fetch_fsaverage(verbose=False)
        subjects_dir = mne.get_config('SUBJECTS_DIR')
    fsaverage_src_path = Path(subjects_dir) / "fsaverage" / "bem" / "fsaverage-ico-5-src.fif"
    return mne.read_source_spaces(fsaverage_src_path, verbose=False)


def generate_template_inverse_operator_from_epochs(epochs, subject_dir, config):
    """
    Generates and saves an inverse operator using the fsaverage template from epochs.

    This function is called when a pre-computed inverse operator is not found.
    It uses the provided epochs data to compute a noise covariance matrix, then
    uses the fsaverage template MRI to compute and save the inverse solution.
    """
    log.info(f"Generating fsaverage template inverse operator for {subject_dir.name}")

    # Drop non-scalp channels for source modeling using the modern pick_types method
    epochs.pick_types(eeg=True)

    # 2. Setup fsaverage paths
    subjects_dir = mne.get_config('SUBJECTS_DIR')
    if subjects_dir is None:
        mne.datasets.fetch_fsaverage(verbose=False)
        subjects_dir = mne.get_config('SUBJECTS_DIR')

    fs_dir = Path(subjects_dir) / "fsaverage"
    trans = "fsaverage"
    src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
    bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"

    # 3. Make forward solution
    log.info("Computing forward solution...")
    fwd = mne.make_forward_solution(
        epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, verbose=False
    )

    # 4. Compute noise covariance
    log.info("Computing noise covariance...")
    cov = mne.compute_covariance(epochs, tmax=0, method='auto', rank=None, verbose=False)

    # 5. Get inverse params from config, with EEG-appropriate defaults
    inverse_params = config.get('inverse', {})
    loose = inverse_params.get('loose', 0.2)
    depth = inverse_params.get('depth', 3.0)
    log.info(f"Computing inverse operator with loose={loose}, depth={depth}...")

    # 6. Compute inverse operator
    inv = make_inverse_operator(epochs.info, fwd, cov, loose=loose, depth=depth, verbose=False)

    # 7. Save the inverse operator for future runs
    inv_fname = subject_dir / f"{subject_dir.name}-inv.fif"
    inv_fname.parent.mkdir(exist_ok=True, parents=True)
    write_inverse_operator(inv_fname, inv, overwrite=True, verbose=False)
    log.info(f"Saved inverse operator to {inv_fname}")

    return inv


def compute_subject_source_contrast(evoked, inv_operator, config):
    """
    Computes the source estimate for a contrast of evoked responses.
    """
    method = config['source']['method']
    lambda2 = 1.0 / (config['source']['snr'] ** 2)
    
    # Compute source estimate using signed orientations to preserve polarity
    stc = mne.minimum_norm.apply_inverse(
        evoked,
        inv_operator,
        lambda2,
        method=method,
        pick_ori='normal',
        verbose=False,
    )
    
    # Morph to fsaverage
    # The subject_from is extracted from the inverse operator's source space info
    subject_from = stc.subject
    if subject_from is None:
        # Fallback for older MNE versions if subject info is not in STC
        subject_from = inv_operator['src'][0]['subject_his_id']

    # Use configured SUBJECTS_DIR; fetch fsaverage if missing
    subjects_dir = mne.get_config('SUBJECTS_DIR')
    if subjects_dir is None:
        fetch_fsaverage(verbose=False)
        subjects_dir = mne.get_config('SUBJECTS_DIR')
    subjects_dir = Path(subjects_dir)
    morph = mne.compute_source_morph(
        stc,
        subject_from=subject_from,
        subject_to='fsaverage',
        subjects_dir=subjects_dir,
        verbose=False,
    )
    stc_fsaverage = morph.apply(stc, verbose=False)
    
    return stc_fsaverage
