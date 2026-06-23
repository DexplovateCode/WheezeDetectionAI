"""
hf_lung_loader.py
==================
Adapter for HF_Lung_V1 / HF_Lung_V2 dataset.

FOLDER STRUCTURE EXPECTED
--------------------------
All WAV and TXT files are placed directly in one root folder:

    HF_Lung_V1/
        steth_20180814_09_37_11.wav
        steth_20180814_09_37_11_label.txt
        ...

ANNOTATION FORMAT (each _label.txt file)
------------------------------------------
Each row = one labeled time-stamped event inside a 15-second recording.
Format: label  start_time  end_time

Example:
    I        00:00:01.500   00:00:02.457
    E        00:00:02.608   00:00:03.637
    Rhonchi  00:00:12.754   00:00:13.485
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal as scipy_signal

from config import (
    SAMPLE_RATE, CLIP_DURATION,
    BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ,
    DATA_DIR, HF_LUNG_DIR, USE_COMBINED_DATASET, HF_LUNG_MAX_FILES,
)

# ─────────────────────────────────────────────────────────────────────────────
#  LABEL MAPPING
# ─────────────────────────────────────────────────────────────────────────────

HF_LABEL_MAP = {
    "W"         : "Wheeze",
    "S"         : "Wheeze",
    "R"         : "Crackle",
    "D"         : "Crackle",
    "I"         : None,
    "E"         : None,
    "WHEEZE"    : "Wheeze",
    "STRIDOR"   : "Wheeze",
    "RHONCHI"   : "Crackle",
    "RHONCHUS"  : "Crackle",
    "CRACKLE"   : "Crackle",
    "CRACKLES"  : "Crackle",
    "DAS"       : "Crackle",
    "INHALATION": None,
    "EXHALATION": None,
    "INSP"      : None,
    "EXP"       : None,
    "1"         : None,
    "2"         : None,
    "3"         : "Wheeze",
    "4"         : "Wheeze",
    "5"         : "Crackle",
    "6"         : "Crackle",
    "-1"        : "Wheeze",
}

HF_BINARY_MAP = {
    "Wheeze"         : 1,
    "Wheeze+Crackle" : 1,
    "Crackle"        : 0,
    "Normal"         : 0,
}


# ─────────────────────────────────────────────────────────────────────────────
#  AUDIO HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _to_float32(raw):
    if raw.dtype == np.int16:
        return raw.astype(np.float32) / 32768.0
    if raw.dtype == np.int32:
        return raw.astype(np.float32) / 2147483648.0
    if raw.dtype == np.uint8:
        return (raw.astype(np.float32) - 128.0) / 128.0
    return raw.astype(np.float32)

def _to_mono(audio):
    return audio.mean(axis=1) if audio.ndim == 2 else audio

def _resample(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    n = int(len(audio) * target_sr / orig_sr)
    return scipy_signal.resample(audio, n).astype(np.float32)

def _bandpass(audio, sr):
    nyq = sr / 2.0
    sos = scipy_signal.butter(
        4, [BANDPASS_LOW_HZ / nyq, BANDPASS_HIGH_HZ / nyq],
        btype="band", output="sos"
    )
    return scipy_signal.sosfiltfilt(sos, audio).astype(np.float32)

def _normalize(audio):
    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio

def _pad_or_trim(audio, target_len):
    if len(audio) >= target_len:
        return audio[:target_len]
    return np.pad(audio, (0, target_len - len(audio)))


# ─────────────────────────────────────────────────────────────────────────────
#  ANNOTATION PARSER
# ─────────────────────────────────────────────────────────────────────────────

def _parse_time(time_str):
    time_str = time_str.strip()
    if ":" in time_str:
        parts = time_str.split(":")
        try:
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
            elif len(parts) == 2:
                m, s = parts
                return int(m) * 60 + float(s)
        except (ValueError, IndexError):
            return 0.0
    return float(time_str)


def parse_hf_lung_annotation(txt_path):
    events = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                try:
                    _parse_time(parts[0])
                    first_is_time = True
                except (ValueError, AttributeError):
                    first_is_time = False

                if first_is_time:
                    start = _parse_time(parts[0])
                    end   = _parse_time(parts[1])
                    raw   = parts[2].strip().upper()
                else:
                    raw   = parts[0].strip().upper()
                    start = _parse_time(parts[1])
                    end   = _parse_time(parts[2])

            except (ValueError, IndexError):
                continue

            mapped_label = HF_LABEL_MAP.get(raw)
            if mapped_label is None:
                continue

            events.append({
                "start": start,
                "end"  : end,
                "code" : raw,
                "label": mapped_label,
            })
    return events


def derive_recording_label(events):
    labels      = {e["label"] for e in events}
    has_wheeze  = "Wheeze"  in labels
    has_crackle = "Crackle" in labels
    if has_wheeze and has_crackle:
        return "Wheeze+Crackle"
    if has_wheeze:
        return "Wheeze"
    if has_crackle:
        return "Crackle"
    return "Normal"


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_hf_lung_dataset(
    dataset_dir  = None,
    sample_rate  = SAMPLE_RATE,
    clip_duration = CLIP_DURATION,
    max_files    = None,
    verbose      = True,
):
    if dataset_dir is None:
        dataset_dir = HF_LUNG_DIR

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    wav_files = sorted(glob.glob(os.path.join(dataset_dir, "*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in: {dataset_dir}")

    if max_files:
        wav_files = wav_files[:max_files]

    if verbose:
        print(f"[HF_LUNG] Dataset folder : {dataset_dir}")
        print(f"[HF_LUNG] WAV files found : {len(wav_files)}")

    target_len = int(sample_rate * clip_duration)
    hop_len    = target_len // 2
    records    = []
    clip_id    = 0
    skipped    = 0

    for idx, wav_path in enumerate(wav_files):
        if verbose and idx % 200 == 0:
            print(f"  ... {idx}/{len(wav_files)} — {clip_id} clips so far")

        stem     = os.path.splitext(wav_path)[0]
        txt_path = stem + "_label.txt"
        if not os.path.isfile(txt_path):
            txt_path = stem + ".txt"
        if not os.path.isfile(txt_path):
            skipped += 1
            continue

        try:
            sr_orig, raw = wavfile.read(wav_path)
        except Exception as e:
            skipped += 1
            continue

        audio = _to_float32(_to_mono(raw))
        audio = _resample(audio, sr_orig, sample_rate)
        audio = _bandpass(audio, sample_rate)
        audio = _normalize(audio)

        events    = parse_hf_lung_annotation(txt_path)
        file_stem = os.path.basename(stem)
        full_len  = len(audio)

        start_idx = 0
        while start_idx + target_len <= full_len:
            clip      = audio[start_idx : start_idx + target_len]
            start_sec = start_idx / sample_rate
            end_sec   = (start_idx + target_len) / sample_rate

            window_events = [
                e for e in events
                if e["end"] > start_sec and e["start"] < end_sec
            ]
            window_label  = derive_recording_label(window_events)
            window_binary = HF_BINARY_MAP[window_label]

            records.append({
                "clip_id"      : clip_id,
                "file_stem"    : f"{file_stem}_w{start_idx}",
                "label"        : window_label,
                "binary_label" : window_binary,
                "start"        : round(start_sec, 3),
                "end"          : round(end_sec, 3),
                "audio"        : clip,
                "sr"           : sample_rate,
            })
            clip_id   += 1
            start_idx += hop_len

    df = pd.DataFrame(records)

    if verbose:
        print(f"\n[HF_LUNG] Total clips : {len(df)}")
        if len(df) > 0:
            for lbl, cnt in df["label"].value_counts().items():
                print(f"  {lbl:<20} {cnt:>5}  ({100*cnt/len(df):.1f}%)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  VERIFY FOLDER
# ─────────────────────────────────────────────────────────────────────────────

def verify_hf_lung_folder(dataset_dir=None):
    if dataset_dir is None:
        dataset_dir = HF_LUNG_DIR

    print(f"\n[VERIFY] Checking: {dataset_dir}")
    if not os.path.isdir(dataset_dir):
        print(f"  [FAIL] Folder does not exist")
        return False

    wav_files = glob.glob(os.path.join(dataset_dir, "*.wav"))
    txt_files = glob.glob(os.path.join(dataset_dir, "*.txt"))
    print(f"  WAV files : {len(wav_files)}")
    print(f"  TXT files : {len(txt_files)}")

    if len(wav_files) == 0:
        print("  [FAIL] No WAV files found")
        return False

    print(f"  [OK] Folder looks good")
    return True


if __name__ == "__main__":
    verify_hf_lung_folder()
    df = load_hf_lung_dataset(max_files=HF_LUNG_MAX_FILES)
    print(f"\n[DONE] {len(df)} clips loaded.")
