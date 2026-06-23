# ==============================================================
#  npy_audit.py — Diagnose and fix 0 KB / empty .npy segment files
#
#  Usage:
#    # Step 1: Audit only (no changes)
#    python npy_audit.py --npy_dir data/spectrograms
#
#    # Step 2: Audit + show root causes
#    python npy_audit.py --npy_dir data/spectrograms --wav_dir data/HF_Lung_v1 --verbose
#
#    # Step 3: Regenerate all broken files
#    python npy_audit.py --npy_dir data/spectrograms --wav_dir data/HF_Lung_v1 --fix
#
#    # Step 4: Delete unfixable files
#    python npy_audit.py --npy_dir data/spectrograms --wav_dir data/HF_Lung_v1 --fix --delete_unfixable
# ==============================================================

import os
import re
import sys
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import librosa
import soundfile as sf

# ── Spectrogram config (must match your preprocess settings) ──────
SAMPLE_RATE  = 16_000
N_MELS       = 64
N_FFT        = 512
HOP_LENGTH   = 160
FIXED_FRAMES = 101
MIN_DURATION = 0.2      # seconds — segments shorter than this are invalid

# ── Filename pattern ──────────────────────────────────────────────
# steth_20180814_09_56_09_seg08_abnormal.npy
# Group 1 = recording stem  (steth_20180814_09_56_09)
# Group 2 = segment index   (08)
# Group 3 = label           (abnormal / inspiration / expiration / etc.)
NPY_RE = re.compile(
    r"^(.+?)_seg(\d+)_([a-zA-Z]+)\.npy$"
)


# ─── Helpers ──────────────────────────────────────────────────────

def parse_npy_name(filename: str):
    """
    Returns (recording_stem, seg_idx, label) or None if unrecognised.
    """
    m = NPY_RE.match(filename)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def is_empty(path: Path) -> bool:
    """True when the file is 0 bytes OR the saved array has no elements."""
    if path.stat().st_size == 0:
        return True
    try:
        arr = np.load(str(path), allow_pickle=False)
        return arr.size == 0
    except Exception:
        return True


def find_wav(wav_dir: Path, stem: str):
    """
    Search *wav_dir* recursively for any audio file whose stem
    matches *stem* (case-insensitive, strips _label suffix).
    Returns Path or None.
    """
    AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
    for p in wav_dir.rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS:
            p_stem = p.stem.lower().replace("_label", "")
            if p_stem == stem.lower():
                return p
    return None


def load_wav_duration(wav_path: Path) -> float:
    """Return duration in seconds without fully loading the file."""
    try:
        info = sf.info(str(wav_path))
        return info.duration
    except Exception:
        try:
            y, sr = librosa.load(str(wav_path), sr=None, mono=True)
            return len(y) / sr
        except Exception:
            return 0.0


def rebuild_spectrogram(wav_path: Path, seg_idx: int) -> np.ndarray | None:
    """
    Re-extract segment *seg_idx* from *wav_path* using the annotation file.
    Returns float32 array (N_MELS, FIXED_FRAMES) or None on failure.
    """
    # Find the annotation file
    label_path = wav_path.with_name(wav_path.stem + "_label.txt")
    if not label_path.exists():
        label_path = wav_path.with_suffix(".txt")
    if not label_path.exists():
        return None

    # Parse annotation
    segments = _parse_annotation(label_path)
    if seg_idx >= len(segments):
        return None

    _, t_start, t_end = segments[seg_idx]
    duration = t_end - t_start

    if duration < MIN_DURATION:
        return None

    try:
        audio, _ = librosa.load(
            str(wav_path),
            sr=SAMPLE_RATE,
            offset=t_start,
            duration=duration,
            mono=True,
        )
    except Exception as e:
        warnings.warn(f"librosa.load failed for {wav_path}: {e}")
        return None

    if len(audio) < 64:           # fewer than 4 ms — too short
        return None

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    # Pad / truncate to fixed size
    T = log_mel.shape[1]
    if T < FIXED_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, FIXED_FRAMES - T)), mode="constant")
    else:
        log_mel = log_mel[:, :FIXED_FRAMES]

    return log_mel   # (N_MELS, FIXED_FRAMES)


def _parse_annotation(txt_path: Path):
    """Parse _label.txt → list of (label, t_start, t_end)."""
    TIME_RE = re.compile(r"(\d+):(\d+):(\d+\.\d+)")

    def to_sec(token):
        m = TIME_RE.fullmatch(token.strip())
        if not m:
            raise ValueError(token)
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))

    segs = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            try:
                segs.append((parts[0], to_sec(parts[1]), to_sec(parts[2])))
            except ValueError:
                continue
    return segs


# ─── Root-cause classifier ────────────────────────────────────────

def diagnose(npy_path: Path, wav_dir: Path | None) -> str:
    """Return a short string describing why the file is empty."""
    parsed = parse_npy_name(npy_path.name)
    if parsed is None:
        return "UNRECOGNISED_FILENAME"

    stem, seg_idx, label = parsed

    if wav_dir is None:
        return "WAV_DIR_NOT_PROVIDED"

    wav_path = find_wav(wav_dir, stem)
    if wav_path is None:
        return "WAV_NOT_FOUND"

    wav_dur = load_wav_duration(wav_path)

    label_path = wav_path.with_name(wav_path.stem + "_label.txt")
    if not label_path.exists():
        label_path = wav_path.with_suffix(".txt")
    if not label_path.exists():
        return "ANNOTATION_NOT_FOUND"

    segs = _parse_annotation(label_path)
    if seg_idx >= len(segs):
        return f"SEG_IDX_OUT_OF_RANGE (file has {len(segs)} segs, requested {seg_idx})"

    raw_label, t_start, t_end = segs[seg_idx]
    duration = t_end - t_start

    if duration < MIN_DURATION:
        return f"SEGMENT_TOO_SHORT ({duration:.3f}s < {MIN_DURATION}s)"

    if t_start >= wav_dur:
        return f"OFFSET_BEYOND_EOF (t_start={t_start:.2f}s, wav_dur={wav_dur:.2f}s)"

    if t_end > wav_dur + 0.5:
        return f"SEGMENT_TRUNCATED (t_end={t_end:.2f}s > wav_dur={wav_dur:.2f}s)"

    # Try loading
    try:
        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE,
                                offset=t_start, duration=duration, mono=True)
        if len(audio) < 64:
            return f"AUDIO_TOO_SHORT_AFTER_LOAD ({len(audio)} samples)"
        return "UNKNOWN_LOAD_SUCCEEDED_NOW"   # file was empty but loads fine — race condition
    except Exception as e:
        return f"LIBROSA_ERROR: {e}"


# ─── Main audit ──────────────────────────────────────────────────

def audit(npy_dir: Path, wav_dir: Path | None, verbose: bool, fix: bool, delete_unfixable: bool):

    all_npy    = sorted(npy_dir.rglob("*.npy"))
    empty_npy  = [p for p in all_npy if is_empty(p)]
    ok_npy     = len(all_npy) - len(empty_npy)

    print("\n" + "═" * 65)
    print("  NPY SEGMENT AUDIT REPORT")
    print("═" * 65)
    print(f"  Directory   : {npy_dir.resolve()}")
    print(f"  Total .npy  : {len(all_npy):>7,}")
    print(f"  OK files    : {ok_npy:>7,}  ({100*ok_npy/max(len(all_npy),1):.1f}%)")
    print(f"  Empty/broken: {len(empty_npy):>7,}  ({100*len(empty_npy)/max(len(all_npy),1):.1f}%)")
    print("═" * 65)

    if not empty_npy:
        print("\n  ✓ No empty files found — dataset is clean.\n")
        return

    # ── Diagnose root causes ──────────────────────────────────────
    cause_counts  = defaultdict(int)
    cause_examples = defaultdict(list)

    print(f"\n  Diagnosing {len(empty_npy)} empty files...")
    for p in empty_npy:
        cause = diagnose(p, wav_dir) if wav_dir else "WAV_DIR_NOT_PROVIDED"
        cause_counts[cause] += 1
        if len(cause_examples[cause]) < 3:
            cause_examples[cause].append(p.name)

    print("\n  ROOT CAUSE BREAKDOWN:")
    print(f"  {'Cause':<45} {'Count':>6}  {'Example'}")
    print("  " + "─" * 62)
    for cause, cnt in sorted(cause_counts.items(), key=lambda x: -x[1]):
        ex = cause_examples[cause][0] if cause_examples[cause] else ""
        print(f"  {cause:<45} {cnt:>6}  {ex}")

    if verbose:
        print(f"\n  FULL LIST ({len(empty_npy)} files):")
        for p in empty_npy:
            cause = diagnose(p, wav_dir) if wav_dir else ""
            print(f"    {p.name}  →  {cause}")

    # ── Fix ───────────────────────────────────────────────────────
    if not fix:
        print(f"\n  Run with --fix to attempt regeneration of {len(empty_npy)} files.")
        print(f"  Run with --fix --delete_unfixable to also remove files that cannot be regenerated.\n")
        return

    if wav_dir is None:
        print("\n  ERROR: --fix requires --wav_dir to be set.\n")
        return

    print(f"\n  FIXING {len(empty_npy)} empty files...")
    fixed = 0
    unfixable = []

    for p in empty_npy:
        parsed = parse_npy_name(p.name)
        if parsed is None:
            unfixable.append((p, "unrecognised filename"))
            continue

        stem, seg_idx, label = parsed
        wav_path = find_wav(wav_dir, stem)
        if wav_path is None:
            unfixable.append((p, "WAV not found"))
            continue

        spec = rebuild_spectrogram(wav_path, seg_idx)
        if spec is None:
            unfixable.append((p, "spectrogram generation failed"))
            continue

        np.save(str(p), spec)
        fixed += 1
        if fixed % 100 == 0:
            print(f"    Fixed {fixed}/{len(empty_npy)}...")

    print(f"\n  ✓ Fixed   : {fixed}")
    print(f"  ✗ Unfixable: {len(unfixable)}")

    if unfixable:
        print(f"\n  Unfixable files:")
        for p, reason in unfixable[:20]:
            print(f"    {p.name}  →  {reason}")
        if len(unfixable) > 20:
            print(f"    ... and {len(unfixable)-20} more")

        if delete_unfixable:
            print(f"\n  Deleting {len(unfixable)} unfixable files...")
            for p, _ in unfixable:
                p.unlink()
                print(f"    Deleted: {p.name}")
            print(f"  Done. These segments cannot be recovered and must be excluded from training.")
        else:
            print(f"\n  Run with --delete_unfixable to remove them from the dataset.")
            print(f"  WARNING: leaving 0-byte files will cause DataLoader errors during training.")

    print()


# ─── Re-verify after fix ─────────────────────────────────────────

def verify(npy_dir: Path):
    """Quick post-fix verification — check shapes are consistent."""
    all_npy   = sorted(npy_dir.rglob("*.npy"))
    shapes    = defaultdict(int)
    bad       = []

    print("\n  POST-FIX VERIFICATION")
    print("  " + "─" * 45)

    for p in all_npy:
        try:
            arr = np.load(str(p), allow_pickle=False)
            shapes[arr.shape] += 1
            if arr.size == 0 or arr.shape != (N_MELS, FIXED_FRAMES):
                bad.append((p.name, arr.shape))
        except Exception as e:
            bad.append((p.name, f"load error: {e}"))

    print(f"  Shape distribution:")
    for shape, cnt in sorted(shapes.items(), key=lambda x: -x[1]):
        expected = " ← EXPECTED" if shape == (N_MELS, FIXED_FRAMES) else " ← WRONG SHAPE"
        print(f"    {str(shape):<20}: {cnt:>6} files{expected}")

    if bad:
        print(f"\n  Still broken ({len(bad)} files):")
        for name, info in bad[:10]:
            print(f"    {name}  →  {info}")
    else:
        print(f"\n  ✓ All {len(all_npy)} files have correct shape {(N_MELS, FIXED_FRAMES)}")


# ─── CLI ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Audit and fix 0 KB / empty .npy spectrogram segment files."
    )
    p.add_argument("--npy_dir",          required=True,
                   help="Directory containing .npy segment files")
    p.add_argument("--wav_dir",          default=None,
                   help="Root directory containing source .wav files (required for --fix)")
    p.add_argument("--fix",              action="store_true",
                   help="Regenerate empty files from source WAVs")
    p.add_argument("--delete_unfixable", action="store_true",
                   help="Delete files that cannot be regenerated (use with --fix)")
    p.add_argument("--verify",           action="store_true",
                   help="Run shape-consistency check after fixing")
    p.add_argument("--verbose",          action="store_true",
                   help="Print every broken filename and its root cause")
    args = p.parse_args()

    npy_dir = Path(args.npy_dir)
    wav_dir = Path(args.wav_dir) if args.wav_dir else None

    if not npy_dir.exists():
        print(f"ERROR: --npy_dir not found: {npy_dir}")
        sys.exit(1)

    audit(npy_dir, wav_dir, args.verbose, args.fix, args.delete_unfixable)

    if args.verify or args.fix:
        verify(npy_dir)
