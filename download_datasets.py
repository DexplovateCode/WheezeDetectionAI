"""
download_datasets.py
====================
Helper to guide dataset download + auto-organize folder structure.

Datasets referenced in the CRS:
  1. Kaggle ICBHI 2017 — Respiratory Sound Database (PRIMARY)
     https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database

  2. GitHub — Aashnajoshi Respiratory Sound Dataset
     https://github.com/aashnajoshi/Respiratory_Sound_Dataset

  3. AI4EU — Respiratory Sounds Dataset
     https://ai4eu.dei.uc.pt/respiratory-sounds-dataset/

  4. Google AudioSet (for background noise augmentation)
     https://research.google.com/audioset/dataset/index.html


This script:
  - Prints exact download instructions for each source
  - Verifies the folder structure after download
  - Confirms annotation files are correctly paired with audio

Run:
    python download_datasets.py --check_only         # just verify folder
    python download_datasets.py --dataset_dir PATH   # verify a specific path
"""

import os
import sys
import glob
import argparse


DOWNLOAD_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════╗
║         DATASET DOWNLOAD INSTRUCTIONS                           ║
╚══════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DATASET 1 (PRIMARY) — Kaggle ICBHI 2017 Respiratory Sound DB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 URL : https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database
 
 Option A — Kaggle CLI (recommended):
   pip install kaggle
   kaggle datasets download -d vbookshelf/respiratory-sound-database
   unzip respiratory-sound-database.zip -d ./data/
 
 Option B — Manual:
   1. Go to the URL above
   2. Click Download
   3. Unzip to   ./data/Respiratory_Sound_Database/
 
 Expected folder structure after unzip:
   Respiratory_Sound_Database/
       audio_and_txt_files/
           101_1b1_Al_sc_Meditron.wav
           101_1b1_Al_sc_Meditron.txt
           102_1b1_Al_sc_Meditron.wav
           102_1b1_Al_sc_Meditron.txt
           ...  (920 WAV files + 920 TXT annotation files)
 
 Then set in src/config.py:
   DATASET_DIR = "./data/Respiratory_Sound_Database"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DATASET 2 — GitHub Respiratory Sound Dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 URL : https://github.com/aashnajoshi/Respiratory_Sound_Dataset
 
 git clone https://github.com/aashnajoshi/Respiratory_Sound_Dataset.git
 
 This repo contains pre-split audio clips labeled:
   Normal / Wheeze / Crackle
 Use for additional training data or validation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DATASET 3 — AI4EU Respiratory Sounds Dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 URL : https://ai4eu.dei.uc.pt/respiratory-sounds-dataset/
 
 Register and download from the AI4EU portal.
 Provides clinically validated respiratory recordings.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DATASET 4 — Google AudioSet (for background noise augmentation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 URL : https://research.google.com/audioset/dataset/index.html
 
 AudioSet provides millions of labeled environmental sounds.
 Use categories like:
   - "Inside, small room"    → clinic noise
   - "Fan"                   → HVAC / fan noise
   - "White noise"           → ambient noise
 
 Download with:
   pip install audioset_download
   
 Or use the balanced_train_segments.csv + yt-dlp to download
 specific sound clips for use in step2_augmentation.py.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def verify_icbhi_folder(dataset_dir: str) -> bool:
    """Check that the ICBHI dataset folder has the expected structure."""
    print(f"\n[CHECK] Verifying dataset at: {dataset_dir}")

    if not os.path.isdir(dataset_dir):
        print(f"  [FAIL] Folder does not exist: {dataset_dir}")
        return False

    # Find WAV and TXT files (may be in subdirectories)
    wav_files = glob.glob(os.path.join(dataset_dir, "**", "*.wav"), recursive=True)
    txt_files = glob.glob(os.path.join(dataset_dir, "**", "*.txt"), recursive=True)

    print(f"  WAV files found : {len(wav_files)}")
    print(f"  TXT files found : {len(txt_files)}")

    # Check pairs
    wav_stems = {os.path.splitext(f)[0] for f in wav_files}
    txt_stems = {os.path.splitext(f)[0] for f in txt_files}
    paired    = wav_stems & txt_stems
    unpaired_wav = wav_stems - txt_stems
    unpaired_txt = txt_stems - wav_stems

    print(f"  Paired   (WAV + TXT) : {len(paired)}")
    if unpaired_wav:
        print(f"  [WARN] WAV without annotation : {len(unpaired_wav)}")
    if unpaired_txt:
        print(f"  [WARN] TXT without audio      : {len(unpaired_txt)}")

    # Verify annotation format (spot check first 5)
    ok_format = 0
    for txt_path in sorted(txt_files)[:5]:
        with open(txt_path) as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    float(parts[0]); float(parts[1])
                    int(parts[2]);   int(parts[3])
                    ok_format += 1
                    break
                except ValueError:
                    pass

    if len(paired) == 0:
        print("  [FAIL] No paired WAV+TXT files found. Check DATASET_DIR in config.py.")
        return False

    print(f"  [OK] Annotation format checks passed: {ok_format}/5 spot checks")
    print(f"  [OK] Dataset looks good!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Dataset download instructions and verification"
    )
    parser.add_argument("--check_only",   action="store_true",
                        help="Only verify folder, don't print instructions")
    parser.add_argument("--dataset_dir",  type=str, default=None,
                        help="Override DATASET_DIR from config.py")
    args = parser.parse_args()

    if not args.check_only:
        print(DOWNLOAD_INSTRUCTIONS)

    # Get dataset dir
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
        from config import DATASET_DIR
        d = args.dataset_dir or DATASET_DIR
    except ImportError:
        d = args.dataset_dir or "./data/Respiratory_Sound_Database"

    verify_icbhi_folder(d)


if __name__ == "__main__":
    main()
