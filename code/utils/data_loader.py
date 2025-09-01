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

def load_config(config_path):
    """Loads and validates a YAML configuration file."""
    log.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Basic validation can be added here
    log.info("Configuration loaded successfully.")
    return config


def get_subject_dirs(accuracy, project_root="."):
    """
    Finds all subject directories (e.g., 'sub-02') for a given accuracy dataset.
    """
    if accuracy == "all":
        data_dir = Path(project_root) / "data" / "all"
    elif accuracy == "acc1":
        data_dir = Path(project_root) / "data" / "acc=1"
    else:
        raise ValueError("accuracy must be 'all' or 'acc1'")

    subject_dirs = sorted([
        d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')
    ])
    log.info(f"Found {len(subject_dirs)} subject directories in {data_dir}.")
    return subject_dirs


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


def create_subject_contrast(subject_dir, config):
    """
    Creates a contrast between two conditions for a single subject.

    Returns:
        tuple: A tuple containing the contrast Evoked object and the concatenated
               Epochs object used for covariance computation. Returns (None, None)
               if data cannot be loaded.
    """
    log.debug(f"Creating contrast for subject {subject_dir.name}")
    try:
        evoked_A, epochs_A = get_evoked_for_condition(
            subject_dir, config['contrast']['condition_A'], baseline=config.get('baseline')
        )
        evoked_B, epochs_B = get_evoked_for_condition(
            subject_dir, config['contrast']['condition_B'], baseline=config.get('baseline')
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


def get_evoked_for_condition(subject_dir, condition_info, baseline=None):
    """
    Loads all epoch files for a given condition set and averages them.

    Returns:
        tuple: A list of Evoked objects and a list of Epochs objects.
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
    for cond_num in condition_numbers:
        # NOTE: The bug was here. The real files use '_epo.fif'.
        fname = subject_dir / f"{subject_dir.name}_task-numbers_cond-{cond_num}_epo.fif"
        if fname.exists():
            epochs = mne.read_epochs(fname, preload=True, verbose=False)

            # Remove problematic 1 Hz FIR high-pass to avoid distortion on short epochs.
            # Rely on baseline correction below for drift removal.

            # Apply baseline correction if provided by the analysis config
            if baseline is not None:
                try:
                    epochs.apply_baseline(baseline=tuple(baseline), verbose=False)
                except Exception:
                    epochs.apply_baseline(baseline=(baseline[0], baseline[1]), verbose=False)

            # Ensure average EEG reference is applied to match inverse operator assumptions
            epochs.set_eeg_reference('average', projection=True, verbose=False)

            # Enforce the standard montage. This is a critical step to override
            # any incorrect/generic digitization info embedded in the FIF files.
            if STANDARD_MONTAGE:
                epochs.set_montage(STANDARD_MONTAGE, on_missing='warn')

            epochs_list.append(epochs)
            evoked = epochs.average()
            evoked.set_eeg_reference('average', projection=True, verbose=False)
            evoked_list.append(evoked)
        else:
            log.debug(f"Epoch file not found, skipping: {fname}")

    if not evoked_list:
        log.warning(f"No epoch files found for condition set {condition_set_name}")
        return [], []

    return evoked_list, epochs_list


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
    subjects_dir = Path(project_root) / "data" / "all" / "fs_subjects_dir"
    fsaverage_src_path = subjects_dir / "fsaverage" / "bem" / "fsaverage-ico-5-src.fif"
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
    write_inverse_operator(inv_fname, inv, verbose=False)
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

    subjects_dir = Path(__file__).resolve().parents[2] / "data" / "all" / "fs_subjects_dir"
    morph = mne.compute_source_morph(
        stc,
        subject_from=subject_from,
        subject_to='fsaverage',
        subjects_dir=subjects_dir,
        verbose=False,
    )
    stc_fsaverage = morph.apply(stc, verbose=False)
    
    return stc_fsaverage
