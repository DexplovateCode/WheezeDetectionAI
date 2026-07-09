# predict.py - Wheeze probability prediction with metrics display

import os, sys, json, argparse
import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("  [WARNING] noisereduce not installed. Run: pip install noisereduce")
    print("            Proceeding without denoising.\n")

import config
from model import WheezeDetector
from dataset import waveform_to_logmel
from config import PROP_DECREASE
print(f"from config prop_decrease:{PROP_DECREASE}")
prop_decrease=PROP_DECREASE
# ─── Denoise Audio ────────────────────────────────────────────────────────────
def denoise_audio(y: np.ndarray, sr: int,
                  noise_clip_sec: float = 0.5,
                  prop_decrease: float=prop_decrease,
                  stationary: bool = False) -> np.ndarray:
    """
    Reduce background noise from a waveform using noisereduce.

    Strategy
    --------
    * A short leading segment (``noise_clip_sec`` seconds) is treated as a
      noise profile.  If the audio is too short to carve out a profile the
      whole signal is used as its own reference (stationary mode fallback).
    # * ``prop_decrease`` controls aggressiveness: 1.0 = full suppression,
      0.5 = gentler reduction (preserves more natural breath texture).
    * ``stationary=True``  → good for steady fan/room noise.
      ``stationary=False`` → better for intermittent noise (default).

    Parameters
    ----------
    y               : 1-D float32 numpy array (audio waveform).
    sr              : Sample rate in Hz.
    noise_clip_sec  : Duration (s) of the leading segment used as noise ref.
    # prop_decrease   : Proportion of noise to reduce (0.0–1.0).
    stationary      : Whether to treat noise as stationary.

    Returns
    -------
    Denoised waveform as a float32 numpy array, same length as input.
    """
    if not NOISEREDUCE_AVAILABLE:
        return y  # pass-through if library missing

    # Non-stationary mode in noisereduce requires time_mask_smooth_ms >= 64.
    # We enforce this minimum whenever non-stationary mode is used.
    TIME_MASK_SMOOTH_MS = 64  # ms — minimum required by SpectralGateNonStationary

    noise_samples = int(noise_clip_sec * sr)
    if len(y) > noise_samples * 2:
        # Use the first `noise_clip_sec` seconds as noise profile
        noise_clip = y[:noise_samples]
        y_denoised = nr.reduce_noise(
            y=y,
            sr=sr,
            y_noise=noise_clip,
            prop_decrease=prop_decrease,
            stationary=stationary,
            time_mask_smooth_ms=TIME_MASK_SMOOTH_MS,
        )
    else:
        # Audio too short — fall back to stationary mode (no explicit noise clip).
        # Stationary mode doesn't require time_mask_smooth_ms, so we omit it.
        y_denoised = nr.reduce_noise(
            y=y,
            sr=sr,
            prop_decrease=prop_decrease,
            stationary=True,
        )

    return y_denoised.astype(np.float32)


# ─── Load & Denoise Audio (shared helper) ────────────────────────────────────
def load_audio(audio_path: str,
               apply_denoising: bool = True,
               noise_clip_sec: float = 0.5,
               prop_decrease: float = prop_decrease,
               stationary: bool = False) -> tuple[np.ndarray, int]:
    """
    Load an audio file with librosa and optionally apply noisereduce denoising.

    Parameters
    ----------
    audio_path      : Path to the audio file.
    apply_denoising : Whether to run denoising (default True).
    noise_clip_sec  : Leading seconds used as noise reference.
    prop_decrease   : Noise suppression strength (0.0–1.0).
    stationary      : Stationary vs non-stationary noise model.

    Returns
    -------
    (y, sr) — waveform (float32 ndarray) and sample rate (int).
    """
    y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)

    if apply_denoising:
        if NOISEREDUCE_AVAILABLE:
            print(f"  Denoising audio  : noisereduce "
                  f"(stationary={stationary}, prop_decrease={prop_decrease})")
            y = denoise_audio(y, sr,
                              noise_clip_sec=noise_clip_sec,
                              prop_decrease=prop_decrease,
                              stationary=stationary)
        # else: warning already printed at import time

    return y, sr


# ─── Load Model ───────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device) -> WheezeDetector:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = WheezeDetector(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    print(f"Loaded checkpoint : {checkpoint_path}")
    print(f"  Trained epoch   : {ckpt.get('epoch', '?')}")
    print(f"  Best val AUC    : {ckpt.get('val_auc', 0):.4f}")
    return model


# ─── Classify window into breath phase ────────────────────────────────────────
def classify_phase(y: np.ndarray, sr: int) -> str:
    """
    Simple energy-based breath phase classifier.
    Returns 'Inspiration', 'Expiration', or 'Abnormal'.
    """
    mid    = len(y) // 2
    e1     = np.sqrt(np.mean(y[:mid] ** 2))
    e2     = np.sqrt(np.mean(y[mid:] ** 2))
    total  = e1 + e2 + 1e-10
    ratio  = e1 / total

    if ratio > 0.55:
        return "Inspiration"
    elif ratio < 0.45:
        return "Expiration"
    else:
        return "Abnormal"


# ─── Predict One File ─────────────────────────────────────────────────────────
def predict_file(audio_path: str, model: WheezeDetector,
                 device: torch.device, threshold: float = config.THRESHOLD,
                 apply_denoising: bool = True,
                 noise_clip_sec: float = 0.5,
                 prop_decrease: float = prop_decrease,
                 stationary: bool = False) -> dict:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # ── Load + optional denoise ───────────────────────────────────────────────
    y, sr = load_audio(
        audio_path,
        apply_denoising=apply_denoising,
        noise_clip_sec=noise_clip_sec,
        prop_decrease=prop_decrease,
        stationary=stationary,
    )

    duration = len(y) / sr

    # Sliding window
    windows = []
    t = 0.0
    while t + config.CLIP_DURATION <= duration + 1e-3:
        start_s = int(t * sr)
        end_s   = min(start_s + config.CLIP_SAMPLES, len(y))
        chunk   = y[start_s:end_s]
        if len(chunk) < config.CLIP_SAMPLES:
            chunk = np.pad(chunk, (0, config.CLIP_SAMPLES - len(chunk)))
        windows.append((t, chunk))
        t += config.HOP_DURATION

    if not windows:
        chunk = y
        if len(chunk) < config.CLIP_SAMPLES:
            chunk = np.pad(chunk, (0, config.CLIP_SAMPLES - len(chunk)))
        windows = [(0.0, chunk)]

    all_probs, all_times, all_phases = [], [], []

    model.eval()
    with torch.no_grad():
        for start_t, chunk in windows:
            spec = waveform_to_logmel(chunk.astype(np.float32), sr)
            spec = spec.unsqueeze(0).to(device)
            _, prob = model(spec)
            all_probs.append(prob.item())
            all_times.append(start_t)
            all_phases.append(classify_phase(chunk, sr))

    all_probs  = np.array(all_probs)
    all_phases = np.array(all_phases)

    n_total       = len(windows)
    n_inspiration = int(np.sum(all_phases == "Inspiration"))
    n_expiration  = int(np.sum(all_phases == "Expiration"))
    n_abnormal    = int(np.sum(all_phases == "Abnormal"))

    mean_prob   = float(np.mean(all_probs))
    max_prob    = float(np.max(all_probs))
    wheeze_prob = round(mean_prob * 100, 1)
    normal_prob = round((1 - mean_prob) * 100, 1)
    detected    = bool(mean_prob >= threshold)

    wheeze_windows = [
        {"start_sec": float(t), "end_sec": float(t + config.CLIP_DURATION),
         "probability": float(p), "phase": ph}
        for t, p, ph in zip(all_times, all_probs, all_phases)
        if p >= threshold
    ]

    return {
        "audio_file"           : os.path.basename(audio_path),
        "duration_sec"         : round(duration, 2),
        "denoising_applied"    : apply_denoising and NOISEREDUCE_AVAILABLE,
        "n_windows"            : n_total,
        "n_inspiration"        : n_inspiration,
        "n_expiration"         : n_expiration,
        "n_abnormal"           : n_abnormal,
        "threshold"            : threshold,
        "wheezing_probability" : wheeze_prob,
        "normal_probability"   : normal_prob,
        "mean_probability"     : round(mean_prob, 4),
        "max_probability"      : round(max_prob,  4),
        "wheeze_detected"      : detected,
        "wheeze_windows"       : wheeze_windows,
        "window_probs"         : [
            {"start_sec": float(t), "prob": round(float(p), 4), "phase": ph}
            for t, p, ph in zip(all_times, all_probs, all_phases)
        ],
    }


# ─── Load Metrics from JSON ───────────────────────────────────────────────────
def load_metrics() -> dict:
    """Load saved model metrics from evaluation results."""
    search_paths = [
        os.path.join(config.RESULTS_DIR, "metrics_summary.json"),
        os.path.join(config.RESULTS_DIR, "test_results.json"),
        os.path.join(config.RESULTS_DIR, "pann_eval_results.json"),
        "./metrics_summary.json",
        "./pann_eval_results.json",
    ]
    for path in search_paths:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            print(f"\n  Loaded metrics from {os.path.basename(path)}")
            return data
    return {}


def extract_metric_values(metrics: dict) -> dict:
    """Extract key metric values from various JSON formats."""
    vals = {}
    src = metrics.get("threshold_0.5", metrics.get("optimal_threshold", metrics))

    vals["accuracy"]      = src.get("accuracy",      metrics.get("accuracy",      None))
    vals["roc_auc"]       = src.get("roc_auc",       metrics.get("roc_auc",       metrics.get("AUC_ROC", None)))
    vals["f1"]            = src.get("f1",             metrics.get("f1",            metrics.get("Macro_F1", None)))
    vals["sensitivity"]   = src.get("sensitivity",    metrics.get("sensitivity",   metrics.get("Wheeze_Sensitivity", None)))
    vals["specificity"]   = src.get("specificity",    metrics.get("specificity",   metrics.get("Wheeze_Specificity", None)))
    vals["ppv"]           = src.get("ppv",            metrics.get("ppv",           None))
    vals["npv"]           = src.get("npv",            metrics.get("npv",           None))
    vals["avg_precision"] = src.get("avg_precision",  metrics.get("avg_precision", None))

    for k, v in vals.items():
        if v is not None and v > 1.5:
            vals[k] = v / 100.0

    return vals


# ─── Print Summary ────────────────────────────────────────────────────────────
def print_summary(result: dict):
    fname    = result["audio_file"]
    n_clips  = result["n_windows"]
    n_insp   = result["n_inspiration"]
    n_exp    = result["n_expiration"]
    n_abn    = result["n_abnormal"]
    w_prob   = result["wheezing_probability"]
    n_prob   = result["normal_probability"]
    detected = result["wheeze_detected"]
    verdict  = "✅ NORMAL" if not detected else "⚠️  WHEEZE DETECTED"
    denoise_status = "✅ Applied" if result.get("denoising_applied") else "⚠️  Skipped (noisereduce not available)"

    print()
    print(f"  File                  : {fname}")
    print(f"  Denoising             : {denoise_status}")
    print(f"  Clips analysed        : {n_clips}")
    print()
    print(f"  {'─'*40}")
    print(f"  Inspiration clips     : {n_insp}")
    print(f"  Expiration clips      : {n_exp}")
    print(f"  Abnormal clips        : {n_abn}")
    print(f"  {'─'*40}")
    print()
    print(f"  Wheezing Probability  : {w_prob} %")
    print(f"  Normal Probability    : {n_prob} %")
    print(f"  Verdict               : {verdict}")


# ─── Print Metrics Table ──────────────────────────────────────────────────────
def print_metrics_table(metrics: dict):
    vals = extract_metric_values(metrics)

    rows = [
        ("Accuracy",           vals.get("accuracy"),      0.85, "Overall correct predictions across all classes"),
        ("Macro F1",           vals.get("f1"),            0.78, "Balanced score across inspiration/expiration/abnormal"),
        ("AUC-ROC",            vals.get("roc_auc"),       0.88, "Class separation ability  (1.0 = perfect)"),
        ("Wheeze Sensitivity", vals.get("sensitivity"),   0.82, "% of real wheeze cases correctly caught (recall)"),
        ("Wheeze Specificity", vals.get("specificity"),   0.78, "% of normal cases correctly identified"),
        ("PPV",                vals.get("ppv"),           0.75, "Precision — when model says wheeze, how often correct"),
        ("NPV",                vals.get("npv"),           0.90, "When model says normal, how often correct"),
        ("Avg Precision",      vals.get("avg_precision"), 0.75, "Area under Precision-Recall curve"),
    ]

    sep = "=" * 90
    print()
    print(sep)
    print("  PANN CNN14 — Model Metrics & Interpretation Guide")
    print(sep)
    print(f"  {'Metric':<22} {'Value':>8}  {'Target':>8}  {'Status':>6}  {'Description'}")
    print(f"  {'─'*22} {'─'*8}  {'─'*8}  {'─'*6}  {'─'*40}")

    for name, val, target, desc in rows:
        if val is None:
            val_str    = "  N/A  "
            status_str = "  —  "
        else:
            val_pct    = val * 100
            tgt_pct    = target * 100
            val_str    = f"{val_pct:>6.2f}%"
            status_str = "  ✅" if val >= target else "  ❌"

        tgt_str = f"{target*100:.1f}%"
        print(f"  {name:<22} {val_str:>8}  {tgt_str:>8} {status_str:>6}  {desc}")

    print(sep)
    print()
    print("  Legend:")
    print("    ✅  Metric meets or exceeds target threshold")
    print("    ❌  Metric below target — model may need more training")
    print("    —   Metric not available in saved results")
    print()
    print("  Interpretation:")
    if vals.get("roc_auc") and vals["roc_auc"] >= 0.90:
        print("    ✅ Excellent AUC (≥0.90) — model strongly separates wheeze from normal")
    if vals.get("sensitivity") and vals["sensitivity"] >= 0.80:
        print("    ✅ High sensitivity — model catches most wheeze cases (low false negatives)")
    if vals.get("specificity") and vals["specificity"] >= 0.80:
        print("    ✅ High specificity — low false alarm rate on normal breathing")
    if vals.get("accuracy") and vals["accuracy"] < 0.85:
        print("    ⚠  Accuracy below 85% — consider more training epochs or data balancing")
    print(sep)


# ─── Visualise ────────────────────────────────────────────────────────────────
def visualise(result: dict, audio_path: str,
              save_dir: str = config.RESULTS_DIR,
              apply_denoising: bool = True,
              noise_clip_sec: float = 0.5,
              prop_decrease: float = prop_decrease,
              stationary: bool = False):
    os.makedirs(save_dir, exist_ok=True)

    times = [w["start_sec"] + config.CLIP_DURATION / 2
             for w in result["window_probs"]]
    probs = [w["prob"] for w in result["window_probs"]]

    # Reload the same (denoised) waveform for the plot so it matches inference
    y, sr = load_audio(
        audio_path,
        apply_denoising=apply_denoising,
        noise_clip_sec=noise_clip_sec,
        prop_decrease=prop_decrease,
        stationary=stationary,
    )

    denoise_tag = " [denoised]" if result.get("denoising_applied") else ""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    fig.suptitle(
        f"Wheeze Analysis{denoise_tag} — {result['audio_file']}\n"
        f"Wheezing: {result['wheezing_probability']}%  |  "
        f"Normal: {result['normal_probability']}%  |  "
        f"Verdict: {'⚠ WHEEZE' if result['wheeze_detected'] else '✓ NORMAL'}",
        fontsize=12, y=0.98
    )

    # 1. Waveform
    ax   = axes[0]
    t_ax = np.linspace(0, result["duration_sec"], len(y))
    ax.plot(t_ax, y, color="#4c84b0", lw=0.6, alpha=0.85)
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform{denoise_tag}")
    ax.set_xlim(0, result["duration_sec"])
    ax.grid(True, alpha=0.3)

    # 2. Log-mel spectrogram
    ax  = axes[1]
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS, fmin=config.FMIN, fmax=config.FMAX
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    try:
        import librosa.display
        img = librosa.display.specshow(
            log_mel, x_axis="time", y_axis="mel",
            sr=sr, hop_length=config.HOP_LENGTH,
            fmin=config.FMIN, fmax=config.FMAX,
            ax=ax, cmap="magma"
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
    except Exception:
        ax.imshow(log_mel, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(f"Log-Mel Spectrogram{denoise_tag}")

    # 3. Wheeze probability
    ax = axes[2]
    colors = ["#e05252" if p >= result["threshold"] else "#4c84b0" for p in probs]
    ax.bar(times, probs, width=config.HOP_DURATION * 0.8,
           color=colors, alpha=0.7, label="Window prob")
    ax.plot(times, probs, "o-", color="#c0392b", ms=3, lw=1.2)
    ax.axhline(result["threshold"], ls="--", color="gray", lw=1,
               label=f'Threshold ({result["threshold"]})')
    ax.axhline(result["mean_probability"], ls=":", color="#2e86de", lw=1.5,
               label=f'Mean ({result["wheezing_probability"]}%)')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Wheeze Probability")
    ax.set_xlabel("Time (s)")
    ax.set_title("Wheeze Probability per Window")
    ax.set_xlim(0, result["duration_sec"])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    fname = os.path.join(
        save_dir,
        os.path.splitext(result["audio_file"])[0] + "_wheeze_analysis.png"
    )
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {fname}")
    return fname


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Wheeze Detection — PANN CNN14")
    parser.add_argument("audio",         type=str, help="Path to input audio file")
    parser.add_argument("--model",       type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
                        help="Path to model checkpoint")
    parser.add_argument("--threshold",   type=float, default=config.THRESHOLD,
                        help="Decision threshold (default 0.5)")
    parser.add_argument("--no-plot",     action="store_true",
                        help="Skip generating the plot")
    # ── Denoising flags ──────────────────────────────────────────────────────
    parser.add_argument("--no-denoise",  action="store_true",
                        help="Disable noisereduce denoising")
    parser.add_argument("--noise-clip",  type=float, default=0.5,
                        help="Seconds of leading audio used as noise profile (default 0.5)")
    parser.add_argument("--prop-decrease", type=float, default=config.PROP_DECREASE,
                        help="Noise suppression strength 0.0–1.0 (default 1.0)")
    parser.add_argument("--stationary", action="store_true",
                        help="Use stationary noise model (default: non-stationary)")
    args = parser.parse_args()

    apply_denoising = not args.no_denoise

    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )

    # Load model
    model = load_model(args.model, device)

    # Predict (with optional denoising baked in)
    result = predict_file(
        args.audio, model, device,
        threshold=args.threshold,
        apply_denoising=apply_denoising,
        noise_clip_sec=args.noise_clip,
        prop_decrease=args.prop_decrease,
        stationary=args.stationary,
    )

    # Print prediction summary
    print_summary(result)

    # Load and print metrics table
    metrics = load_metrics()
    if metrics:
        print_metrics_table(metrics)
    else:
        print("\n  (No metrics file found — run evaluate.py first to generate metrics)")

    # Plot
    if not args.no_plot:
        try:
            visualise(result, args.audio,
                      apply_denoising=apply_denoising,
                      noise_clip_sec=args.noise_clip,
                      prop_decrease=args.prop_decrease,
                      stationary=args.stationary)
        except Exception as e:
            print(f"  (Plotting skipped: {e})")

    # Save JSON
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_json = os.path.join(
        config.RESULTS_DIR,
        os.path.splitext(os.path.basename(args.audio))[0] + "_prediction.json"
    )
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  JSON saved → {out_json}")

    return result


if __name__ == "__main__":
    main()