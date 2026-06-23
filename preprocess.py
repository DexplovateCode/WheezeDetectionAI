# ============================================================
#  BreathPattern — Step 1: Preprocessing
#
#  Data format (flat folder):
#    steth_20180814_09_37_11.wav
#    steth_20180814_09_37_11_label.txt   ← paired label file
#
#  Label file format (one event per line):
#    I  00:00:01.500  00:00:02.457      ← Inspiration
#    D  00:00:01.500  00:00:02.457      ← Diaphragm marker (skipped)
#    E  00:00:02.608  00:00:03.637      ← Expiration
#    Wheeze   00:00:12.754  00:00:13.485  ← Abnormal sound
#    Rhonchi  00:00:07.836  00:00:08.862  ← Abnormal sound
#
#  What this script does:
#    1. Reads every WAV + its _label.txt from DATA_DIR
#    2. For each I / E / Wheeze / Rhonchi event → extracts that audio segment
#       (I/E events that overlap an abnormal event are saved BOTH as their
#        breath-phase class AND as the abnormal class — no information lost)
#    3. Pads/trims to SEG_DURATION (2 sec)
#    4. Computes log-mel spectrogram  shape (64, 101)
#    5. Saves as .npy  +  writes manifest.json
#    6. Computes a per-file wheeze probability = wheeze_duration / total_event_duration
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
    """
    Reads a WAV file as mono float32 in [-1, 1].
    Falls back to librosa.load if the raw struct-based read fails
    (e.g. 32-bit float WAV, compressed WAV, or unusual sample widths) —
    this prevents the whole file from being skipped on a parsing edge case.
    """
    try:
        with wave.open(path, "rb") as wf:
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sw not in (1, 2, 4):
            raise ValueError(f"Unsupported sample width: {sw} bytes")

        fmt = {1: "b", 2: "h", 4: "i"}[sw]
        data = np.array(
            struct.unpack(f"<{n_frames * n_ch}{fmt}", raw), dtype=np.float32
        )

        if n_ch > 1:
            data = data.reshape(-1, n_ch).mean(axis=1)

        max_val = float(2 ** (8 * sw - 1))
        data = data / max_val

        # Resample if the file's native rate doesn't match config
        with wave.open(path, "rb") as wf:
            native_sr = wf.getframerate()
        if native_sr != SAMPLE_RATE:
            data = librosa.resample(data, orig_sr=native_sr, target_sr=SAMPLE_RATE)

        return data.astype(np.float32)

    except Exception:
        # Fallback: let librosa handle decoding (covers float WAV, odd headers, etc.)
        data, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return data.astype(np.float32)


# ── Log-mel spectrogram via librosa ──────────────────────────
def extract_log_mel(audio: np.ndarray) -> np.ndarray:
    """Returns shape (N_MELS, T) — e.g. (64, 101) for a 2s/16kHz segment."""
    if audio.size == 0:
        audio = np.zeros(SEG_SAMPLES, dtype=np.float32)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=F_MIN,
        fmax=F_MAX,
    )
    # Guard against silent/all-zero segments -> mel can be all zeros -> log(0)
    mel = np.maximum(mel, 1e-10)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


# ── Parse one label file ──────────────────────────────────────
def parse_label_file(label_path: str) -> list:
    """
    Returns a list of (class_name, t_start, t_end) tuples.

    Unlike the original version, abnormal events (wheeze/rhonchi) are
    NOT merged into breath-phase events. Instead:
      - Every I/E event is kept as-is (inspiration/expiration).
      - Every Wheeze/Rhonchi event is ALSO kept as its own entry.
    This means an I event that overlaps a Wheeze event produces TWO
    segments: one labeled "inspiration", one labeled "wheeze" —
    no information is discarded, and wheeze probability stays accurate.
    """
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
                # Unknown tag — skip safely instead of crashing
                continue

            cls = LABEL_MAP[tag]
            s, e = ts_to_sec(t_start), ts_to_sec(t_end)

            if e <= s:
                # Malformed/zero-duration event — skip
                continue

            events.append((cls, s, e))

    return events


# ── Compute wheeze probability for one file ────────────────────
def compute_wheeze_probability(events: list) -> float:
    """
    wheeze_probability = total wheeze duration / total recording span

    The denominator uses the overall time span covered by breath-phase
    (inspiration/expiration) events, NOT a sum of every event's duration.
    This matters because CAS/DAS events (wheeze, stridor, rhonchi, crackle)
    are sub-events that occur INSIDE an I/E window, often sharing the exact
    same timestamps (see HF_Lung_V1 'D' rows). Summing all event durations
    would double-count that overlapping time and artificially deflate the
    wheeze probability.

    Returns 0.0 if there are no breath-phase events at all.
    """
    breath_events = [
        (s, e) for cls, s, e in events if cls in ("inspiration", "expiration")
    ]
    if not breath_events:
        return 0.0

    recording_span = max(e for _, e in breath_events) - min(s for s, _ in breath_events)
    if recording_span <= 0:
        return 0.0

    wheeze_duration = sum(e - s for cls, s, e in events if cls == "wheeze")
    return round(wheeze_duration / recording_span, 4)


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
    file_summaries = []
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

            if not events:
                print(f"  [WARN] No usable events in {label_fname} — skipping.")
                continue

            wheeze_prob = compute_wheeze_probability(events)
            n_audio_samples = len(audio)

            for idx, (cls, t_start, t_end) in enumerate(events):
                s_start = int(t_start * SAMPLE_RATE)
                s_end = int(t_end * SAMPLE_RATE)

                # Clip indices to the actual audio length to avoid empty
                # slices when a label timestamp slightly exceeds file duration
                s_start = max(0, min(s_start, n_audio_samples))
                s_end = max(s_start, min(s_end, n_audio_samples))

                segment = audio[s_start:s_end]

                if len(segment) == 0:
                    print(
                        f"  [WARN] Empty segment in {wav_fname} "
                        f"[{t_start:.3f}-{t_end:.3f}] — using silence."
                    )
                    segment = np.zeros(SEG_SAMPLES, dtype=np.float32)
                elif len(segment) < SEG_SAMPLES:
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
                        "file_wheeze_probability": wheeze_prob,
                    }
                )
                class_counts[cls] += 1
                total_ok += 1

            file_summaries.append(
                {
                    "wav_file": wav_fname,
                    "num_events": len(events),
                    "wheeze_probability": wheeze_prob,
                }
            )

        except Exception as ex:
            print(f"  [ERROR] {wav_fname}: {ex}")
            total_fail += 1

    manifest_path = os.path.join(SPEC_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    summary_path = os.path.join(SPEC_DIR, "wheeze_probability_summary.json")
    with open(summary_path, "w") as f:
        json.dump(file_summaries, f, indent=2)

    print(f"\n{'='*52}")
    print(f"  Preprocessing complete")
    print(f"  Segments saved : {total_ok}")
    print(f"  Failed files   : {total_fail}")
    print(f"\n  Class breakdown:")
    for cls, cnt in class_counts.items():
        print(f"    {cls:<14}: {cnt:>4} segments")
    print(
        f"\n  Spectrogram shape       : ({N_MELS}, {math.ceil(SEG_SAMPLES/HOP_LENGTH)+1})"
    )
    print(f"  Manifest saved          : {manifest_path}")
    print(f"  Wheeze prob. summary    : {summary_path}")
    print(f"{'='*52}")

    return manifest


if __name__ == "__main__":
    preprocess_dataset()
