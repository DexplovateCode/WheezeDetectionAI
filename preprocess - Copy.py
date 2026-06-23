# ============================================================
#  BreathPattern — Step 1: Preprocessing
#
#  Data format (flat folder):
#    steth_20180814_09_37_11.wav
#    steth_20180814_09_37_11_label.txt   ← paired label file
#
#  Label file format (one event per line):
#    I  00:00:01.500  00:00:02.457      ← Inspiration
#    D  00:00:01.500  00:00:02.457      ← Diaphragm (skipped)
#    E  00:00:02.608  00:00:03.637      ← Expiration
#    Rhonchi  00:00:12.754  00:00:13.485  ← Abnormal sound
#
#  What this script does:
#    1. Reads every WAV + its _label.txt from DATA_DIR
#    2. For each I / E / Rhonchi event → extracts that audio segment
#    3. Pads/trims to SEG_DURATION (2 sec)
#    4. Computes log-mel spectrogram  shape (64, 101)
#    5. Saves as .npy  +  writes manifest.json
#
#  Usage:
#    python preprocess.py
# ============================================================

import os
import json
import wave
import struct
import math
import numpy as np
from tqdm import tqdm
import librosa

from config import (
    DATA_DIR,
    SPEC_DIR,
    SAMPLE_RATE,
    SEG_DURATION,
    SEG_SAMPLES,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    F_MIN,
    F_MAX,
    CLASS_NAMES,
    CLASS_TO_IDX,
    LABEL_MAP,
    SKIP_LABELS,
)


# ── Time string → seconds ─────────────────────────────────────
def ts_to_sec(ts: str) -> float:
    """'00:00:02.608' → 2.608"""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


# ── Read WAV as float32 numpy array ──────────────────────────
def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        n_ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    fmt = {1: "b", 2: "h", 4: "i"}[sw]
    data = np.array(struct.unpack(f"{n_frames * n_ch}{fmt}", raw), dtype=np.float32)

    if n_ch == 2:
        data = data.reshape(-1, 2).mean(axis=1)

    max_val = float(2 ** (8 * sw - 1))
    data /= max_val
    return data


# ── Log-mel spectrogram via librosa ──────────────────────────
def extract_log_mel(audio: np.ndarray) -> np.ndarray:
    """Returns shape (64, 101)"""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=F_MIN,
        fmax=F_MAX,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


# ── Parse one label file ──────────────────────────────────────
def parse_label_file(label_path: str) -> list:
    events = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            tag, t_start, t_end = parts
            if tag in SKIP_LABELS:
                continue
            if tag not in LABEL_MAP:
                continue
            events.append((LABEL_MAP[tag], ts_to_sec(t_start), ts_to_sec(t_end)))

    ie_events = [(c, s, e) for c, s, e in events if c in ("inspiration", "expiration")]
    abn_events = [(c, s, e) for c, s, e in events if c == "abnormal"]

    merged = []
    for cls, s, e in ie_events:
        is_abn = any(not (e <= a_s or s >= a_e) for _, a_s, a_e in abn_events)
        merged.append(("abnormal" if is_abn else cls, s, e))

    for cls, s, e in abn_events:
        overlaps_ie = any(not (e <= i_e or s >= i_s) for _, i_s, i_e in ie_events)
        if not overlaps_ie:
            merged.append((cls, s, e))

    return merged


# ── Main preprocessing function ───────────────────────────────
def preprocess_dataset():
    os.makedirs(SPEC_DIR, exist_ok=True)

    wav_files = sorted(
        [
            f
            for f in os.listdir(DATA_DIR)
            if f.endswith(".wav") and not f.startswith(".")
        ]
    )

    manifest = []
    total_ok = 0
    total_fail = 0
    class_counts = {c: 0 for c in CLASS_NAMES}

    print(f"\nFound {len(wav_files)} WAV files in {DATA_DIR}")
    print(f"Extracting segments (fixed {SEG_DURATION}s, padded)...\n")

    for wav_fname in tqdm(wav_files, unit="file"):
        wav_path = os.path.join(DATA_DIR, wav_fname)
        label_fname = wav_fname.replace(".wav", "_label.txt")
        label_path = os.path.join(DATA_DIR, label_fname)

        if not os.path.exists(label_path):
            print(f"  [WARN] No label file for {wav_fname} — skipping.")
            continue

        try:
            audio = read_wav(wav_path)
            events = parse_label_file(label_path)

            for idx, (cls, t_start, t_end) in enumerate(events):
                s_start = int(t_start * SAMPLE_RATE)
                s_end = int(t_end * SAMPLE_RATE)
                segment = audio[s_start:s_end]

                if len(segment) < SEG_SAMPLES:
                    segment = np.pad(segment, (0, SEG_SAMPLES - len(segment)))
                else:
                    segment = segment[:SEG_SAMPLES]

                log_mel = extract_log_mel(segment)
                base = wav_fname.replace(".wav", "")
                npy_name = f"{base}_seg{idx:02d}_{cls}.npy"
                npy_path = os.path.join(SPEC_DIR, npy_name)
                np.save(npy_path, log_mel)

                manifest.append(
                    {
                        "npy_path": npy_path,
                        "wav_file": wav_fname,
                        "label_name": cls,
                        "label_idx": CLASS_TO_IDX[cls],
                        "t_start": round(t_start, 3),
                        "t_end": round(t_end, 3),
                        "seg_idx": idx,
                    }
                )
                class_counts[cls] += 1
                total_ok += 1

        except Exception as ex:
            print(f"  [ERROR] {wav_fname}: {ex}")
            total_fail += 1

    manifest_path = os.path.join(SPEC_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*52}")
    print(f"  Preprocessing complete")
    print(f"  Segments saved : {total_ok}")
    print(f"  Failed files   : {total_fail}")
    print(f"\n  Class breakdown:")
    for cls, cnt in class_counts.items():
        print(f"    {cls:<14}: {cnt:>4} segments")
    print(f"\n  Spectrogram shape  : ({N_MELS}, {math.ceil(SEG_SAMPLES/HOP_LENGTH)+1})")
    print(f"  Manifest saved     : {manifest_path}")
    print(f"{'='*52}")

    return manifest


if __name__ == "__main__":
    preprocess_dataset()
