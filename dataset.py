#  BreathPattern — Dataset & DataLoader
#  Loads pre-extracted .npy spectrograms from manifest.json
#  Splits train/val/test stratified by CLASS (not by file).
# ============================================================

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import (
    SPEC_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    NUM_CLASSES,
    CLASS_NAMES,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
)


class BreathDataset(Dataset):
    """
    Loads log-mel spectrogram .npy files.
    Each item: spec (1, 64, 101), label (long int)
    """

    def __init__(
        self,
        manifest: list,
        mean: float = None,
        std: float = None,
        augment: bool = False,
    ):
        self.manifest = manifest
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        spec = np.load(item["npy_path"])  # (64, 101)

        # Normalise
        if self.mean is not None:
            spec = (spec - self.mean) / (self.std + 1e-6)
        else:
            spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        if self.augment:
            spec = self._specaugment(spec)

        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1,64,101)
        label = torch.tensor(item["label_idx"], dtype=torch.long)
        return spec, label

    def _specaugment(self, spec: np.ndarray) -> np.ndarray:
        """Frequency + time masking for augmentation."""
        spec = spec.copy()
        n_mels, n_frames = spec.shape
        # Frequency mask (up to 8 bins)
        f = np.random.randint(0, 8)
        f0 = np.random.randint(0, max(1, n_mels - f))
        spec[f0 : f0 + f, :] = 0.0
        # Time mask (up to 15 frames)
        t = np.random.randint(0, 15)
        t0 = np.random.randint(0, max(1, n_frames - t))
        spec[:, t0 : t0 + t] = 0.0
        return spec


def load_manifest() -> list:
    path = os.path.join(SPEC_DIR, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Manifest not found: {path}\nRun  python preprocess.py  first."
        )
    with open(path) as f:
        return json.load(f)


def compute_normalization_stats(manifest: list):
    """Global mean + std over train split (streaming, no full RAM load)."""
    print("Computing normalisation stats...")
    rsum, rsq, n = 0.0, 0.0, 0
    for item in manifest:
        s = np.load(item["npy_path"]).astype(np.float64)
        rsum += s.sum()
        rsq += (s**2).sum()
        n += s.size
    mean = rsum / n
    std = np.sqrt(rsq / n - mean**2)
    print(f"  mean={mean:.4f}  std={std:.4f}")
    return float(mean), float(std)


# def get_dataloaders(augment_train: bool = True):
def get_loaders(augment_train: bool = True):
    """
    # Returns train_loader, val_loader, test_loader, class_weights.
    Returns train_loader, val_loader, test_loader, class_weights.
    Stratified split by class label.
    """
    manifest = load_manifest()
    labels = [item["label_idx"] for item in manifest]
    idx_all = list(range(len(manifest)))

    # Train vs (val+test)
    idx_train, idx_temp, _, labels_temp = train_test_split(
        idx_all,
        labels,
        test_size=VAL_SPLIT + TEST_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    # Val vs test
    val_frac = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=1.0 - val_frac,
        stratify=labels_temp,
        random_state=RANDOM_SEED,
    )

    train_manifest = [manifest[i] for i in idx_train]
    mean, std = compute_normalization_stats(train_manifest)

    train_ds = BreathDataset(train_manifest, mean, std, augment=augment_train)
    val_ds = BreathDataset([manifest[i] for i in idx_val], mean, std, augment=False)
    test_ds = BreathDataset([manifest[i] for i in idx_test], mean, std, augment=False)

    print(f"\nDataset splits:")
    print(f"  Train : {len(train_ds):>5} segments")
    print(f"  Val   : {len(val_ds):>5} segments")
    print(f"  Test  : {len(test_ds):>5} segments")

    # Class weights for imbalanced classes
    train_labels = [manifest[i]["label_idx"] for i in idx_train]
    counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights /= weights.sum()
    class_weights = torch.tensor(weights, dtype=torch.float32)

    print(f"\nClass distribution (train):")
    for i, name in enumerate(CLASS_NAMES):
        if i >= len(counts):
            continue
        print(f"  {name:<14}: {int(counts[i]):>4}  weight={weights[i]:.4f}")

    kw = dict(num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **kw)

    return train_loader, val_loader, test_loader, class_weights
