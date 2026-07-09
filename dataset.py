
import os, re, random
from pathlib import Path
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import config

def hhmmss_to_sec(time_str):
    time_str = time_str.strip()
    parts = time_str.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0])*60 + float(parts[1])
        else:
            return float(parts[0])
    except Exception:
        raise ValueError(f"Cannot parse time: {time_str!r}")

def load_hf_lung_annotations(dataset_dir):
    from collections import Counter
    annotations  = {}
    dataset_path = Path(dataset_dir)
    label_files  = sorted(dataset_path.glob("*_label.txt"))
    print(f"Found {len(label_files)} label files in {dataset_dir!r}")
    for f in label_files:
        audio_stem = f.stem.replace("_label", "")
        events = []
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = re.split(r"\s+", line)
                if len(parts) < 3:
                    continue
                try:
                    label = parts[0].strip()
                    start = hhmmss_to_sec(parts[1])
                    end   = hhmmss_to_sec(parts[2])
                    events.append((start, end, label))
                except ValueError:
                    continue
        annotations[audio_stem] = events
    all_labels = Counter()
    for evs in annotations.values():
        for _, _, lbl in evs:
            all_labels[lbl] += 1
    print("\nAll labels found in dataset:")
    for lbl, cnt in all_labels.most_common():
        tag = " <- WHEEZE (positive)" if lbl in config.WHEEZE_LABELS else ""
        print(f"  [{lbl}]  {cnt} occurrences{tag}")
    print()
    return annotations

def is_wheeze_segment(events, start, end, overlap_thresh=0.3):
    win_len = end - start
    for ev_start, ev_end, label in events:
        if label not in config.WHEEZE_LABELS:
            continue
        overlap = max(0.0, min(end, ev_end) - max(start, ev_start))
        if win_len > 0 and (overlap / win_len) >= overlap_thresh:
            return 1
    return 0

def build_clip_list(dataset_dir, annotations, clip_dur, hop_dur):
    clips = []
    dp    = Path(dataset_dir)
    for ext in ("*.wav", "*.WAV", "*.flac", "*.mp3"):
        for audio_file in sorted(dp.glob(ext)):
            stem   = audio_file.stem
            events = annotations.get(stem, [])
            try:
                duration = librosa.get_duration(path=str(audio_file))
            except Exception:
                continue
            t = 0.0
            while t + clip_dur <= duration + 1e-3:
                end_t = min(t + clip_dur, duration)
                label = is_wheeze_segment(events, t, end_t,overlap_thresh=0.1)
                clips.append((str(audio_file), t, label))
                t += hop_dur
    return clips

def augment_waveform(y, sr):
    shift = int(random.uniform(-config.TIME_SHIFT_MAX, config.TIME_SHIFT_MAX) * len(y))
    y = np.roll(y, shift)
    y = y + np.random.randn(len(y)).astype(np.float32) * config.NOISE_LEVEL
    if random.random() < 0.4:
        steps = random.randint(-config.PITCH_SHIFT_STEPS, config.PITCH_SHIFT_STEPS)
        if steps != 0:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    return y.astype(np.float32)

def waveform_to_logmel(y, sr):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS, fmin=config.FMIN, fmax=config.FMAX,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=config.TOP_DB)
    log_mel = (log_mel + config.TOP_DB) / config.TOP_DB - 1.0
    return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)

class HFLungDataset(Dataset):
    def __init__(self, clips, augment=False):
        self.clips   = clips
        self.augment = augment
    def __len__(self):
        return len(self.clips)
    def __getitem__(self, idx):
        path, start, label = self.clips[idx]
        try:
            y, sr = librosa.load(path, sr=config.SAMPLE_RATE,
                                  offset=start, duration=config.CLIP_DURATION, mono=True)
        except Exception:
            y, sr = np.zeros(config.CLIP_SAMPLES, dtype=np.float32), config.SAMPLE_RATE
        if len(y) < config.CLIP_SAMPLES:
            y = np.pad(y, (0, config.CLIP_SAMPLES - len(y)))
        else:
            y = y[:config.CLIP_SAMPLES]
        if self.augment and config.AUGMENT_TRAIN:
            y = augment_waveform(y, sr)
        return waveform_to_logmel(y, sr), torch.tensor(label, dtype=torch.float32)

def safe_split(data, test_size, seed):
    """Always succeeds — stratified when possible, random otherwise."""
    lbls  = [c[2] for c in data]
    n_pos = sum(lbls)
    n_neg = len(lbls) - n_pos
    # Need at least 2 positives AND 2 negatives on each side
    ratio     = test_size if isinstance(test_size, float) else test_size / len(data)
    min_count = int(len(data) * min(ratio, 1 - ratio))
    need      = 2
    can_strat = (n_pos >= need) and (n_neg >= need) and (min_count >= need)
    if can_strat:
        try:
            return train_test_split(data, test_size=test_size,
                                    random_state=seed, stratify=lbls)
        except ValueError:
            pass
    print(f"  [INFO] Using random split (wheeze={n_pos}, normal={n_neg})")
    return train_test_split(data, test_size=test_size,
                            random_state=seed, shuffle=True)

def get_dataloaders(dataset_dir=config.DATASET_ROOT, val_ratio=0.15):
    annotations = load_hf_lung_annotations(dataset_dir)
    all_clips   = build_clip_list(dataset_dir, annotations,
                                   config.CLIP_DURATION, config.HOP_DURATION)
    if not all_clips:
        raise RuntimeError(f"No audio clips found under {dataset_dir!r}")

    labels = [c[2] for c in all_clips]
    n_pos  = sum(labels)
    n_neg  = len(labels) - n_pos
    print(f"Total clips built : {len(all_clips)}")
    print(f"  Wheeze clips : {n_pos}  ({100*n_pos/len(labels):.1f}%)")
    print(f"  Normal clips : {n_neg}  ({100*n_neg/len(labels):.1f}%)")

    if n_pos == 0:
        raise RuntimeError("Wheeze clips = 0! Check WHEEZE_LABELS in config.py.")

    if os.path.exists(config.TRAIN_LIST) and os.path.exists(config.TEST_LIST):
        with open(config.TRAIN_LIST) as f:
            train_stems = set(l.strip() for l in f)
        with open(config.TEST_LIST) as f:
            test_stems  = set(l.strip() for l in f)
        train_pool = [c for c in all_clips if Path(c[0]).stem in train_stems]
        test_clips = [c for c in all_clips if Path(c[0]).stem in test_stems]
        train_clips, val_clips = safe_split(train_pool, val_ratio, config.SEED)
    else:
        train_clips, temp     = safe_split(all_clips, 0.30, config.SEED)
        val_clips, test_clips = safe_split(temp,      0.50, config.SEED)

    print(f"  Train:{len(train_clips)} | Val:{len(val_clips)} | Test:{len(test_clips)}")

    train_labels = [c[2] for c in train_clips]
    n_tr_pos = sum(train_labels)
    n_tr_neg = len(train_labels) - n_tr_pos
    class_weight = [
        1.0/max(n_tr_neg,1) if l==0 else 1.0/max(n_tr_pos,1)
        for l in train_labels
    ]
    sampler = WeightedRandomSampler(class_weight,
                                     num_samples=len(train_clips), replacement=True)
    kw = dict(num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    return (
        DataLoader(HFLungDataset(train_clips, augment=True),
                   batch_size=config.BATCH_SIZE, sampler=sampler, **kw),
        DataLoader(HFLungDataset(val_clips),
                   batch_size=config.BATCH_SIZE, shuffle=False, **kw),
        DataLoader(HFLungDataset(test_clips),
                   batch_size=config.BATCH_SIZE, shuffle=False, **kw),
    )
