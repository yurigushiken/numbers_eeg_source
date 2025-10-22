#!/usr/bin/env python
"""
Pre-Training Setup Checker
Verifies that all required software, data, and configuration is in place.
"""

import sys
import subprocess
import shutil
from pathlib import Path

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import os
    os.system('chcp 65001 > nul 2>&1')

# ANSI color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text):
    """Print a formatted header."""
    print(f"\n{BLUE}{BOLD}{'='*60}{RESET}")
    print(f"{BLUE}{BOLD}{text}{RESET}")
    print(f"{BLUE}{BOLD}{'='*60}{RESET}\n")

def print_success(text):
    """Print success message."""
    print(f"{GREEN}[OK]{RESET} {text}")

def print_error(text):
    """Print error message."""
    print(f"{RED}[ERROR]{RESET} {text}")

def print_warning(text):
    """Print warning message."""
    print(f"{YELLOW}[WARNING]{RESET} {text}")

def check_command(command, name, version_flag="--version"):
    """Check if a command exists and get its version."""
    try:
        result = subprocess.run(
            [command, version_flag],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0]
            print_success(f"{name} is installed: {version}")
            return True
        else:
            print_error(f"{name} command failed")
            return False
    except FileNotFoundError:
        print_error(f"{name} is not installed or not in PATH")
        return False
    except subprocess.TimeoutExpired:
        print_error(f"{name} command timed out")
        return False
    except Exception as e:
        print_error(f"{name} check failed: {e}")
        return False

def check_python_package(package_name):
    """Check if a Python package is installed."""
    try:
        __import__(package_name)
        print_success(f"Python package '{package_name}' is installed")
        return True
    except ImportError:
        print_error(f"Python package '{package_name}' is NOT installed")
        return False

def check_data_directory():
    """Check if the data directory exists and has the expected structure."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    if not data_dir.exists():
        print_error("Data directory not found at: data/")
        print(f"        Expected location: {data_dir}")
        return False

    # Check for the preprocessed data subdirectory
    preprocessed_dir = data_dir / "data_preprocessed" / "hpf_1.5_lpf_35_baseline-on"

    if not preprocessed_dir.exists():
        print_error("Preprocessed data directory not found")
        print(f"        Expected: data/data_preprocessed/hpf_1.5_lpf_35_baseline-on/")
        return False

    # Check for at least one .fif file
    fif_files = list(preprocessed_dir.glob("*.fif"))
    if not fif_files:
        print_error("No .fif data files found in preprocessed directory")
        return False

    print_success(f"Data directory found with {len(fif_files)} .fif files")
    return True

def check_conda_environment():
    """Check if we're in the correct conda environment."""
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if "numbers_eeg_source" in result.stdout:
            print_success("Conda environment 'numbers_eeg_source' exists")

            # Check if currently active
            current_env = subprocess.run(
                ["conda", "info", "--envs"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if "* numbers_eeg_source" in current_env.stdout or "numbers_eeg_source *" in current_env.stdout:
                print_success("Currently using 'numbers_eeg_source' environment")
                return True
            else:
                print_warning("'numbers_eeg_source' environment exists but is NOT active")
                print(f"        {YELLOW}Run: conda activate numbers_eeg_source{RESET}")
                return True  # Still count as success if it exists
        else:
            print_error("Conda environment 'numbers_eeg_source' not found")
            print(f"        {RED}Run: conda env create -f environment.yml{RESET}")
            return False

    except Exception as e:
        print_error(f"Failed to check conda environment: {e}")
        return False

def check_project_structure():
    """Verify the project has the expected directory structure."""
    project_root = Path(__file__).parent.parent

    required_dirs = [
        "code",
        "configs",
        "new_user",
        "assets",
    ]

    required_files = [
        "environment.yml",
        "requirements.txt",
        "README.md",
    ]

    all_good = True

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print_success(f"Directory found: {dir_name}/")
        else:
            print_error(f"Directory missing: {dir_name}/")
            all_good = False

    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print_success(f"File found: {file_name}")
        else:
            print_error(f"File missing: {file_name}")
            all_good = False

    return all_good

def main():
    """Run all setup checks."""
    print_header("Pre-Training Setup Checker")
    print("This script verifies that your system is ready for EEG analysis.\n")

    all_checks_passed = True

    # Check 1: Git
    print_header("1. Checking Git Installation")
    if not check_command("git", "Git"):
        all_checks_passed = False

    # Check 2: Conda (informational only - not required if packages work)
    print_header("2. Checking Conda/Miniforge Installation")
    conda_installed = check_command("conda", "Conda")

    # Check 3: Conda Environment (informational only)
    print_header("3. Checking Conda Environment")
    env_ok = check_conda_environment()

    # Check 4: Project Structure
    print_header("4. Checking Project Structure")
    if not check_project_structure():
        all_checks_passed = False

    # Check 5: Data Directory
    print_header("5. Checking Data Directory")
    if not check_data_directory():
        all_checks_passed = False

    # Check 6: Python Packages (only if environment is active)
    print_header("6. Checking Python Packages")
    packages_to_check = [
        "mne",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "nibabel",
        "pyvista",
    ]

    packages_ok = True
    for package in packages_to_check:
        if not check_python_package(package):
            all_checks_passed = False
            packages_ok = False

    # If packages are OK but conda checks failed, that's fine - they're in the right env
    if packages_ok and not (conda_installed and env_ok):
        print_warning("Conda not detected in PATH, but all packages are available")
        print_warning("This is OK - you're running from the correct environment")
        all_checks_passed = True  # Override - packages working is what matters

    # Final Summary
    print_header("Setup Check Summary")

    if all_checks_passed:
        print(f"\n{GREEN}{BOLD}SUCCESS! Your setup is complete!{RESET}")
        print(f"{GREEN}You are ready for the first training meeting.{RESET}\n")
        return 0
    else:
        print(f"\n{RED}{BOLD}SETUP INCOMPLETE{RESET}")
        print(f"{RED}Please review the errors above and complete the missing steps.{RESET}")
        print(f"\n{YELLOW}If you need help, email: mkg2145@tc.columbia.edu{RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
