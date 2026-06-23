import os
import math

BASE_DIR = "D:\Python Projects\WheezeDetectionAI_GPU\WheezeDetectionAI"
DATASET_PATH = "D:\Python Projects\WheezeDetectionAI_GPU\WheezeDetectionAI\data"
DATA_DIR = "D:\Python Projects\WheezeDetectionAI_GPU\WheezeDetectionAI\data"
SPEC_DIR = os.path.join(BASE_DIR, "data", "spectrograms")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

for _dir in [SPEC_DIR, MODEL_DIR, LOG_DIR, OUTPUT_DIR]:
    os.makedirs(_dir, exist_ok=True)

SAMPLE_RATE = 16000
FULL_DURATION = 15
FULL_SAMPLES = SAMPLE_RATE * FULL_DURATION
SEG_DURATION = 2.0
SEG_SAMPLES = int(SAMPLE_RATE * SEG_DURATION)

CLASS_NAMES = [
    "inspiration",
    "expiration",
    "wheeze",
    "stridor",
    "rhonchi",
    "crackle",
    "normal",  # I/E segments with no overlapping CAS/DAS event
]
# CLASS_NAMES  = ["inspiration", "expiration", "abnormal"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

LABEL_MAP = {
    "I": "inspiration",
    "E": "expiration",
    "D": "crackle",
    "Wheeze": "wheeze",
    "Stridor": "stridor",
    "Rhonchi": "rhonchi",
}
# LABEL_MAP = {
#     "I"      : "inspiration",
#     "E"      : "expiration",
#     "Rhonchi": "abnormal",
#     "Wheeze" : "abnormal",
#     "Crackle": "abnormal",
# }
SKIP_LABELS = set()

N_MELS = 64
N_FFT = 256
HOP_LENGTH = 80
F_MIN = 20
F_MAX = 2000
TIME_FRAMES = math.ceil(SEG_SAMPLES / HOP_LENGTH) + 1

BATCH_SIZE = 256
NUM_WORKERS = 4
PIN_MEMORY = True
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# CHECKPOINT_PATH = os.path.join(MODEL_DIR, "breath_cnn_best.pth")
# FINAL_MODEL     = os.path.join(MODEL_DIR, "breath_cnn_final.pth")

# ── HF Lung Loader Settings ──────────────────────────────────
CLIP_DURATION = 5.0
BANDPASS_LOW_HZ = 100
BANDPASS_HIGH_HZ = 2500
HF_LUNG_DIR = os.path.join(BASE_DIR, "data", "HF_Lung_V1")
DATASET_DIR = os.path.join(BASE_DIR, "data")
USE_COMBINED_DATASET = False
HF_LUNG_MAX_FILES = None

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# GPU / Training additions
USE_AMP = True
EPOCHS = 80
PATIENCE = 15
WEIGHT_DECAY = 1e-4
# NUM_CLASSES = 3

# Reset to no class weights — use focal loss instead
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0  # focuses on hard examples automatically
