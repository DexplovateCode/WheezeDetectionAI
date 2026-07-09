# 🫁 PANN Wheeze Detector — HF_Lung_V1

A deep-learning pipeline that listens to a lung-sound recording and outputs the
**probability that it contains wheezing**. Built on a **CNN14 (PANN)** audio
backbone, fine-tuned on the **HF_Lung_V1** dataset.

---

## 1. What This Project Does

Wheeze is a high-pitched sound produced when airways are narrowed (asthma,
COPD, bronchitis, etc.). This project trains a neural network to recognize
that sound automatically from a lung-sound recording, and gives:

- A **wheeze probability** (0–100%) for the whole recording
- A **per-window breakdown** (which few-second segments contain wheeze)
- A simple **breath-phase tag** (Inspiration / Expiration / Abnormal) per window
- Visual **plots** (waveform, spectrogram, probability-over-time) and a JSON report

---

## 2. Project Files

| File | Role |
|---|---|
| `config.py` | Every path, audio setting, model setting, and hyper-parameter lives here |
| `dataset.py` | Reads HF_Lung_V1 label files, cuts audio into fixed-length clips, labels each clip wheeze / non-wheeze, builds PyTorch `DataLoader`s |
| `model.py` | Defines the CNN14 (PANN) backbone + a small classification head that outputs a wheeze probability |
| `train.py` | Trains the model, saves the best checkpoint, evaluates on the held-out test set |
| `evaluate.py` | Loads a trained checkpoint, computes full metrics (AUC, F1, sensitivity, specificity, etc.) and builds an evaluation dashboard image |
| `predict.py` | Runs the trained model on **one new audio file** and reports the wheeze probability, with optional denoising and plots |

---

## 3. How the Pipeline Works (End to End)

```
Raw audio (.wav)
      │
      ▼
1. Load & (optionally) denoise the waveform
      │
      ▼
2. Slice into fixed-length windows (2-second clips, back-to-back)
      │
      ▼
3. Convert each window into a log-mel spectrogram ("sound picture")
      │
      ▼
4. Feed the spectrogram into the CNN14 (PANN) backbone → 2048-number
   summary ("embedding") of that window's sound
      │
      ▼
5. A small classifier head turns the embedding into one number:
   the probability that window contains a wheeze
      │
      ▼
6. Combine all window probabilities → overall wheeze probability
   for the whole recording + a Wheeze / Normal verdict
```

### Step-by-step detail

1. **Loading & Labels** (`dataset.py`)
   Every `.wav` file has a matching `..._label.txt` file with lines like
   `Wheeze  00:01:02.500  00:01:04.200`. These mark exactly when a wheeze
   happens in the recording.

2. **Clip building**
   Since recordings can be long, the audio is chopped into short,
   fixed-length clips (`CLIP_DURATION` seconds, default 2s, sliding by
   `HOP_DURATION`). A clip is labeled **1 (wheeze)** if a wheeze annotation
   overlaps it enough (`overlap_thresh`), otherwise **0 (normal)**.

3. **Feature extraction — log-mel spectrogram**
   Each clip's raw waveform is converted into a **log-mel spectrogram**: a
   2-D image-like representation showing which frequencies are loud at
   which point in time. This is the format the neural network actually
   "sees."

4. **CNN14 (PANN) backbone** (`model.py`)
   A 6-block convolutional neural network (pretrained on the huge AudioSet
   dataset, then fine-tuned here) scans the spectrogram and compresses it
   into a 2048-number "fingerprint" of the sound.
   PANN stands for Pretrained Audio Neural Networks — a family of models introduced in a 2020 paper by Kong et al., trained on AudioSet (Google's dataset of ~2 million 10-second audio clips covering 527 everyday sound classes: dogs barking, engines, speech, music, etc.).
   CNN14 is the specific architecture in that family your project uses. The "14" refers to the number of weighted layers (12 conv layers across 6 conv blocks + 2 fully-connected layers).
   A backbone just means: the feature-extracting part of the network, as opposed to the task-specific "head" that makes the final decision. The backbone's job is to turn raw input into a rich numerical summary; the head's job is to turn that summary into your actual answer (wheeze / no wheeze).

5. **Classification head**
   A small fully-connected network turns that 2048-number fingerprint into
   a single wheeze probability between 0 and 1 (via a sigmoid).

6. **Training** (`train.py`)
   The model is trained with `BCEWithLogitsLoss` (weighted for class
   imbalance, since wheezes are rarer than normal breathing), using AdamW +
   cosine learning-rate schedule, with the backbone frozen for the first
   few "warm-up" epochs. The best checkpoint (highest validation AUC) is
   saved automatically, with early stopping if it stops improving.

7. **Evaluation** (`evaluate.py`)
   Runs the trained model on the untouched test set and reports accuracy,
   sensitivity/specificity, PPV/NPV, F1, ROC-AUC, average precision, plus
   ROC curve, PR curve, confusion matrix, probability histogram, and
   training curves in a single dashboard image.

8. **Prediction on new audio** (`predict.py`)
   For a brand-new recording: optionally denoises it, slides the same
   fixed-length window across it, gets a wheeze probability per window,
   averages them into one overall probability, and (optionally) plots the
   waveform / spectrogram / probability-over-time together with a saved
   JSON report.

---

## 4. Setup

1. Install dependencies:
   ```bash
   pip install torch librosa numpy scikit-learn matplotlib noisereduce
   ```
2. Edit `config.py`:
   - `DATASET_ROOT` → folder containing the HF_Lung_V1 `.wav` + `_label.txt` files
   - `PRETRAINED_PATH` → path to the downloaded `Cnn14_16k.pth` weights
3. (Optional) Provide `train.txt` / `test.txt` split files; otherwise the
   code automatically creates a stratified 70/15/15 train/val/test split.

## 5. Usage

**Train:**
```bash
python train.py
```

**Evaluate on the test set:**
```bash
python evaluate.py [path_to_checkpoint.pth]
```

**Predict on a new recording:**
```bash
python predict.py path/to/recording.wav --threshold 0.4
```
Useful flags: `--no-denoise`, `--prop-decrease 0.5`, `--stationary`, `--no-plot`.

---

## 6. Key Configuration Values (`config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `SAMPLE_RATE` | 4000 Hz | Native rate of HF_Lung_V1 audio |
| `CLIP_DURATION` / `HOP_DURATION` | 2.0s / 2.0s | Length of each analysis window |
| `N_MELS` / `N_FFT` / `HOP_LENGTH` | 64 / 512 / 128 | Spectrogram resolution |
| `FMIN` / `FMAX` | 50 Hz / 2000 Hz | Frequency band analyzed (Nyquist-capped) |
| `THRESHOLD` | 0.4 | Probability cutoff for "wheeze detected" |
| `MODEL_NAME` | CNN14 | PANN backbone architecture |
| `BATCH_SIZE` / `NUM_EPOCHS` / `LR` | 128 / 40 / 3e-4 | Training hyper-parameters |

---

## 7. Outputs

All results are written to `./outputs/`:
- `checkpoints/best_model.pth` — best trained model
- `results/training_history.json` — per-epoch metrics
- `results/evaluation_dashboard.png` — 6-panel evaluation dashboard
- `results/metrics_summary.json` — final metrics at default + optimal threshold
- `results/<filename>_wheeze_analysis.png` and `_prediction.json` — per-recording prediction outputs
