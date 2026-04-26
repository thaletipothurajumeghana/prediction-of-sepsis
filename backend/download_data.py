"""
Dataset Download & Preparation Helper
=======================================
Guides you through downloading the sepsis dataset from Kaggle or PhysioNet
and prepares it for training.

Run: python src/download_data.py
"""

import os
import sys
import subprocess

GREEN  = '\033[92m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
RED    = '\033[91m'
RESET  = '\033[0m'
BOLD   = '\033[1m'

BANNER = f"""
{BOLD}{'='*60}
  SEPSIS DATASET — DOWNLOAD GUIDE
{'='*60}{RESET}

Two excellent datasets are available for this project:

{CYAN}{BOLD}Option 1 (RECOMMENDED): Kaggle — Prediction of Sepsis{RESET}
{CYAN}  URL   : https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis{RESET}
  Source : PhysioNet Challenge 2019 data (flattened CSV)
  Size   : ~250 MB
  Format : Single CSV — easy to use
  Label  : SepsisLabel (0/1)
  Samples: ~40,000 patients, 40 features per hour
  Features: HR, O2Sat, Temp, SBP, MAP, Resp, WBC, Lactate,
            Creatinine, Platelets, pH, BUN, Age, Gender, etc.

{CYAN}{BOLD}Option 2: PhysioNet Challenge 2019 (Official){RESET}
{CYAN}  URL   : https://physionet.org/content/challenge-2019/1.0.0/{RESET}
  Source : 3 hospital systems, 60,000+ patients
  Format : One PSV file per patient (needs merging)
  Size   : ~1.5 GB
  Label  : SepsisLabel (0/1), 6-hour early label
  Note   : Requires free PhysioNet account

{CYAN}{BOLD}Option 3: Kaggle — Sepsis Survival (Simple){RESET}
{CYAN}  URL   : https://www.kaggle.com/datasets/joebeachcapital/sepsis-survival-minimal-clinical-records{RESET}
  Source : Clinical ICU records
  Format : Single small CSV — great for quick testing
  Size   : Very small (<5 MB)
  Features: Age, episode_number, hospital_outcome (simpler)
"""


def check_kaggle_api():
    """Check if kaggle CLI is installed and configured."""
    try:
        result = subprocess.run(['kaggle', '--version'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    return False


def download_kaggle(dataset_id, dest_dir='data'):
    """Download a Kaggle dataset using kaggle CLI."""
    os.makedirs(dest_dir, exist_ok=True)
    print(f"\n{GREEN}[→] Downloading via Kaggle API...{RESET}")
    cmd = ['kaggle', 'datasets', 'download', '-d', dataset_id,
           '--unzip', '-p', dest_dir]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0:
        print(f"{GREEN}[✓] Downloaded to '{dest_dir}/' directory{RESET}")
        return True
    else:
        print(f"{RED}[✗] Download failed.{RESET}")
        return False


def manual_instructions():
    print(f"""
{BOLD}─── MANUAL DOWNLOAD STEPS ───────────────────────────────────{RESET}

{BOLD}For Kaggle (Option 1 — recommended):{RESET}

  1. Sign up / log in at https://www.kaggle.com
  2. Visit:
     https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis
  3. Click {BOLD}Download{RESET} (top right) → downloads a ZIP
  4. Extract the ZIP and place the CSV in this project's {CYAN}data/{RESET} folder
  5. Rename the file to {CYAN}data/sepsis_dataset.csv{RESET}
  6. Run: {GREEN}python src/train.py{RESET}

{BOLD}Kaggle API (fast, one command):{RESET}

  pip install kaggle
  # Place your kaggle.json token in ~/.kaggle/kaggle.json
  kaggle datasets download -d salikhussaini49/prediction-of-sepsis --unzip -p data/
  mv data/*.csv data/sepsis_dataset.csv
  python src/train.py

{BOLD}For PhysioNet (Option 2 — advanced):{RESET}

  1. Register at https://physionet.org
  2. Accept the data use agreement for Challenge 2019
  3. Run:
       wget -r -N -c -np --user YOUR_USERNAME --ask-password \\
         https://physionet.org/files/challenge-2019/1.0.0/training/
  4. Set DATA_PATH = 'data/training/p00001.psv'  in src/train.py
     (The loader auto-merges all PSV files in that folder)
  5. Run: {GREEN}python src/train.py{RESET}

{BOLD}Quick test with synthetic data:{RESET}

  python src/download_data.py --synthetic
  python src/train.py

""")


def generate_synthetic(n_samples=5000, dest='data/sepsis_dataset.csv'):
    """
    Generate realistic synthetic sepsis data for testing the pipeline.
    NOT for real medical use — for development only.
    """
    import numpy as np
    import pandas as pd

    print(f"\n{YELLOW}[→] Generating synthetic dataset ({n_samples} samples)...{RESET}")
    np.random.seed(42)
    n_sep = int(n_samples * 0.15)   # 15% sepsis rate (realistic)
    n_non = n_samples - n_sep

    def sample_group(n, sepsis):
        s = float(sepsis)
        return {
            'HR':        np.random.normal(80 + 20*s, 15, n).clip(30, 200),
            'O2Sat':     np.random.normal(97 - 5*s, 3, n).clip(50, 100),
            'Temp':      np.random.normal(36.8 + 1.2*s, 0.8, n).clip(33, 42),
            'SBP':       np.random.normal(120 - 20*s, 20, n).clip(50, 200),
            'MAP':       np.random.normal(80 - 15*s, 15, n).clip(30, 150),
            'DBP':       np.random.normal(75 - 10*s, 12, n).clip(30, 130),
            'Resp':      np.random.normal(16 + 6*s, 4, n).clip(6, 50),
            'WBC':       np.random.normal(7 + 8*s, 4, n).clip(0.5, 40),
            'Lactate':   np.random.normal(1.0 + 2.5*s, 0.8, n).clip(0.1, 15),
            'Creatinine':np.random.normal(0.9 + 1.5*s, 0.5, n).clip(0.3, 15),
            'Platelets': np.random.normal(250 - 100*s, 80, n).clip(10, 600),
            'Hgb':       np.random.normal(13 - 2*s, 2, n).clip(4, 20),
            'pH':        np.random.normal(7.40 - 0.08*s, 0.05, n).clip(6.8, 7.6),
            'Glucose':   np.random.normal(100 + 50*s, 30, n).clip(40, 500),
            'BUN':       np.random.normal(14 + 20*s, 8, n).clip(3, 100),
            'Bilirubin_total': np.random.normal(0.8 + 2*s, 0.6, n).clip(0.1, 20),
            'HCO3':      np.random.normal(24 - 5*s, 3, n).clip(5, 40),
            'PaCO2':     np.random.normal(40 + 5*s, 5, n).clip(15, 80),
            'PTT':       np.random.normal(30 + 15*s, 8, n).clip(10, 150),
            'Fibrinogen':np.random.normal(300 - 100*s, 80, n).clip(50, 700),
            'Calcium':   np.random.normal(9.0 - 0.8*s, 0.8, n).clip(5, 14),
            'Potassium': np.random.normal(4.0 + 0.5*s, 0.5, n).clip(2, 7),
            'Sodium':    np.random.normal(140 + 3*s, 5, n).clip(120, 165),
            'Age':       np.random.normal(55 + 10*s, 18, n).clip(18, 100),
            'Gender':    np.random.binomial(1, 0.5, n),
            'ICULOS':    np.random.exponential(24 + 24*s, n).clip(1, 300),
            'SepsisLabel': np.ones(n, dtype=int) * int(sepsis),
        }

    sep_data = pd.DataFrame(sample_group(n_sep, True))
    non_data = pd.DataFrame(sample_group(n_non, False))

    df = pd.concat([sep_data, non_data], ignore_index=True).sample(frac=1, random_state=42)

    # Introduce realistic missingness
    for col in ['Lactate', 'PTT', 'Fibrinogen', 'PaCO2', 'Bilirubin_total']:
        mask = np.random.rand(len(df)) < 0.35
        df.loc[mask, col] = np.nan

    os.makedirs('data', exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"{GREEN}[✓] Synthetic dataset saved to '{dest}'{RESET}")
    print(f"    {n_sep} sepsis cases + {n_non} non-sepsis = {n_samples} total")
    print(f"\n{YELLOW}⚠ Reminder: Synthetic data is for pipeline testing only.{RESET}")
    print(f"{YELLOW}  Use real datasets for medical applications.{RESET}")


def main():
    print(BANNER)

    if '--synthetic' in sys.argv:
        generate_synthetic()
        print(f"\n{GREEN}Now run:{RESET} python src/train.py")
        return

    print(f"\n{BOLD}Do you have the Kaggle CLI installed?{RESET}")
    has_kaggle = check_kaggle_api()

    if has_kaggle:
        print(f"{GREEN}[✓] Kaggle API found!{RESET}")
        choice = input("\nAuto-download the recommended dataset? (y/n): ").strip().lower()
        if choice == 'y':
            success = download_kaggle('salikhussaini49/prediction-of-sepsis')
            if success:
                # Rename to standard name
                import glob
                csvs = glob.glob('data/*.csv')
                if csvs and not os.path.exists('data/sepsis_dataset.csv'):
                    os.rename(csvs[0], 'data/sepsis_dataset.csv')
                    print(f"{GREEN}[✓] Renamed to data/sepsis_dataset.csv{RESET}")
                print(f"\n{GREEN}Ready! Run: python src/train.py{RESET}")
                return
    else:
        print(f"{YELLOW}[!] Kaggle CLI not found.{RESET}")

    manual_instructions()
    gen = input("Generate synthetic data for pipeline testing? (y/n): ").strip().lower()
    if gen == 'y':
        generate_synthetic()
        print(f"\n{GREEN}Now run:{RESET} python src/train.py")


if __name__ == '__main__':
    main()
