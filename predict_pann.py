# ============================================================
#  WheezeDetectionAI — PANN Inference Script
#  Usage:
#    python predict_pann.py path/to/recording.wav
#    python predict_pann.py path/to/audio.wav --stride 0.5
#    python predict_pann.py path/to/audio.wav --threshold 60
#    python predict_pann.py path/to/audio.wav --no-denoise
#    python predict_pann.py path/to/audio.wav --vad-speech-removal
#
#  New dependencies for the cleanup stage:
#    pip install noisereduce scipy webrtcvad
#  On Windows, "webrtcvad" has no prebuilt wheel and needs MSVC build tools
#  to compile. Skip that hassle with the drop-in, precompiled fork instead:
#    pip install webrtcvad-wheels        (same "import webrtcvad" in code)
# ============================================================

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import noisereduce as nr
from fractions import Fraction
from scipy.signal import butter, sosfiltfilt, resample_poly

from config import (
    CLASS_NAMES,
    NUM_CLASSES,
    SAMPLE_RATE,
    SEG_SAMPLES,
    SEG_DURATION,
)
from model_pann import Cnn14FineTuned
from preprocess import read_wav, extract_log_mel

PANN_MODEL_PATH = os.path.join("models", "pann_best_model.pth")

# ── Cleanup-stage defaults ──────────────────────────────────────
# Lung/wheeze acoustic energy lives almost entirely between these two
# cutoffs. Filtering the recording down to this band is what actually
# does the work of removing heart sounds and muscle/movement rumble
# (both concentrated below ~100 Hz) and high-frequency hiss/sibilants
# (above ~2 kHz), without touching the wheeze signal itself.
LOWCUT_HZ = 50.0
HIGHCUT_HZ = 2000.0


_VAD_SUPPORTED_RATES = (8000, 16000, 32000, 48000)


# ── Load PANN model ───────────────────────────────────────────
def load_pann_model(model_path=None):
    path = model_path or PANN_MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n  ❌ PANN model not found: {path}\n"
            f"  Run  python train_pann.py  first."
        )
    model = Cnn14FineTuned(num_classes=NUM_CLASSES)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state", state))
    model.eval()
    print(f"  ✅ PANN model loaded: {path}")
    return model


# ── Stage 1+2: noisereduce + band-pass cleanup ──────────────────
def bandpass_filter(
    audio: np.ndarray,
    sr: int,
    lowcut: float = LOWCUT_HZ,
    highcut: float = HIGHCUT_HZ,
    order: int = 4,
) -> np.ndarray:
    """Butterworth band-pass, applied zero-phase (sosfiltfilt) so it
    doesn't smear timing across the segment boundaries used downstream."""
    nyquist = 0.5 * sr
    low = max(lowcut / nyquist, 1e-4)
    high = min(highcut / nyquist, 0.999)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, audio)


def denoise_audio(
    audio: np.ndarray,
    sr: int,
    lowcut: float = LOWCUT_HZ,
    highcut: float = HIGHCUT_HZ,
    prop_decrease: float = 0.85,
    apply_bandpass: bool = True,
    time_mask_smooth_ms=None,
) -> np.ndarray:
    """
    Cleans a raw recording before it reaches the model.

    Stage 1 — noisereduce (spectral gating):
        Strips ambient/background noise (room hiss, AC hum, mic self-noise)
        and attenuates continuous background speech/chatter that doesn't
        match the breath-sound spectral envelope.

        time_mask_smooth_ms is left at None (disabled) on purpose: its
        default (50ms) is computed against a fixed STFT hop length, and at
        low sample rates (e.g. 4000 Hz, common for breath-sound models)
        that default works out to less than a single frame, which raises
        "time_mask_smooth_ms needs to be at least Nms". Disabling it makes
        this safe at any sample rate; frequency-axis mask smoothing still
        runs as normal.

    Stage 2 — band-pass filter (default 100 Hz – 2000 Hz):
        Removes heart-sound energy (S1/S2 sit below ~100 Hz) and
        muscle/movement artifacts (low-frequency rumble from the
        stethoscope shifting or the patient tensing), plus high-frequency
        electrical noise and sibilant speech components above ~2 kHz.

    Note on "brain activity": EEG is an electrical signal, not an acoustic
    one — it cannot be present in a microphone/stethoscope WAV recording,
    so there's nothing for an audio pipeline to remove on that front. If
    your data is actually multimodal (audio synced with EEG channels),
    that needs a separate EEG-side artifact-rejection step, not this one.
    """
    cleaned = nr.reduce_noise(
        y=audio.astype(np.float32),
        sr=sr,
        prop_decrease=prop_decrease,
        stationary=False,
        time_mask_smooth_ms=time_mask_smooth_ms,
    )
    if apply_bandpass:
        try:
            cleaned = bandpass_filter(cleaned, sr, lowcut, highcut)
        except ValueError:
            print(
                "  ⚠️  Band-pass skipped (clip too short for the filter order) "
                "— using noisereduce output only"
            )
    return cleaned.astype(np.float32)


# ── Optional stage 3: VAD-based speech suppression ───────────────
def _resample_for_vad(audio: np.ndarray, sr: int, target_sr: int = 16000):
    """webrtcvad only accepts 8k/16k/32k/48k sample rates."""
    if sr in _VAD_SUPPORTED_RATES:
        return audio, sr
    frac = Fraction(target_sr, sr).limit_denominator(1000)
    resampled = resample_poly(audio, frac.numerator, frac.denominator)
    return resampled.astype(np.float32), target_sr


def suppress_speech_vad(
    audio: np.ndarray, sr: int, aggressiveness: int = 2, frame_ms: int = 30
) -> np.ndarray:
    """
    Best-effort speech removal using WebRTC's voice-activity detector:
    frames flagged as voiced are zeroed out of the original signal.

    Caveat: wheezes and speech overlap substantially in frequency
    (roughly 100 Hz – 2 kHz for both), so VAD is a coarse tool here — it
    flags "speech-shaped" energy rather than speech content specifically,
    and can occasionally zero out wheeze-only segments that share that
    shape. Off by default; enable with --vad-speech-removal for recordings
    with heavy talking in the background.
    """
    try:
        import webrtcvad
    except ImportError:
        print(
            "  ⚠️  webrtcvad not installed — skipping VAD speech removal "
            "(pip install webrtcvad, or on Windows: pip install webrtcvad-wheels)"
        )
        return audio

    vad_audio, vad_sr = _resample_for_vad(audio, sr)
    pcm16 = (np.clip(vad_audio, -1.0, 1.0) * 32767).astype(np.int16)

    frame_len_vad = int(vad_sr * frame_ms / 1000)
    vad = webrtcvad.Vad(aggressiveness)
    cleaned = audio.copy()

    for start in range(0, len(pcm16) - frame_len_vad, frame_len_vad):
        frame = pcm16[start : start + frame_len_vad].tobytes()
        if vad.is_speech(frame, vad_sr):
            t0, t1 = start / vad_sr, (start + frame_len_vad) / vad_sr
            o0, o1 = int(t0 * sr), int(t1 * sr)
            cleaned[o0:o1] = 0.0

    return cleaned


# ── Sliding window prediction ─────────────────────────────────
def predict_file(
    wav_path: str,
    model=None,
    model_path=None,
    stride_sec=1.0,
    denoise=True,
    lowcut=LOWCUT_HZ,
    highcut=HIGHCUT_HZ,
    denoise_strength=0.85,
    vad_speech_removal=False,
    vad_aggressiveness=2,
):
    if model is None:
        model = load_pann_model(model_path)

    audio = read_wav(wav_path)

    if denoise:
        audio = denoise_audio(
            audio,
            SAMPLE_RATE,
            lowcut=lowcut,
            highcut=highcut,
            prop_decrease=denoise_strength,
        )
        print(
            f"  🧹 Cleaned: noisereduce + band-pass [{lowcut:.0f}–{highcut:.0f} Hz] "
            f"(removes background noise, heart sounds, muscle/movement artifacts)"
        )

    if vad_speech_removal:
        audio = suppress_speech_vad(
            audio, SAMPLE_RATE, aggressiveness=vad_aggressiveness
        )
        print(
            f"  🧹 Cleaned: VAD speech suppression (aggressiveness={vad_aggressiveness})"
        )

    stride_samp = int(stride_sec * SAMPLE_RATE)
    results = []

    start = 0
    while start + SEG_SAMPLES <= len(audio):
        seg = audio[start : start + SEG_SAMPLES].copy()
        seg = (seg - seg.mean()) / (seg.std() + 1e-6)

        lm = extract_log_mel(seg)
        lm = (lm - lm.mean()) / (lm.std() + 1e-6)
        tensor = torch.tensor(lm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1)[0].numpy()

        predicted = CLASS_NAMES[probs.argmax()]
        results.append(
            {
                "time": round(start / SAMPLE_RATE, 1),
                "predicted_class": predicted,
                "confidence": float(probs.max()),
                "probabilities": {n: float(p) for n, p in zip(CLASS_NAMES, probs)},
            }
        )
        start += stride_samp

    return results


# ── Metrics guide from pann_eval_results.json ─────────────────
def print_metrics_guide(wheeze_pct: float, threshold: float):
    eval_data = {}
    try:
        import json

        with open("outputs/pann_eval_results.json") as f:
            eval_data = json.load(f)
        print(f"  📊 Loaded metrics from pann_eval_results.json")
    except FileNotFoundError:
        print(f"  ⚠️  No eval results found — run python evaluate_pann.py first")

    accuracy = eval_data.get("accuracy", None)
    macro_f1 = eval_data.get("macro_f1", None)
    auc_roc = eval_data.get("auc_roc", None)
    per_class = eval_data.get("per_class", {})
    w_sens = per_class.get("wheeze", {}).get("sensitivity", None)
    w_spec = per_class.get("wheeze", {}).get("specificity", None)

    rows = [
        (
            "Accuracy",
            accuracy,
            "%",
            85.0,
            True,
            "Overall correct predictions across all classes",
        ),
        (
            "Macro F1",
            macro_f1,
            "%",
            78.0,
            True,
            "Balanced score across inspiration/expiration/abnormal",
        ),
        (
            "AUC-ROC",
            auc_roc,
            "",
            0.88,
            True,
            "Class separation ability  (1.0 = perfect)",
        ),
        (
            "Wheeze Sensitivity",
            w_sens,
            "%",
            82.0,
            True,
            "% of real wheeze cases correctly caught (recall)",
        ),
        (
            "Wheeze Specificity",
            w_spec,
            "%",
            78.0,
            True,
            "% of normal cases correctly identified",
        ),
        (
            "Wheeze Probability",
            wheeze_pct,
            "%",
            threshold,
            False,
            "This recording — probability of abnormal breath sounds",
        ),
    ]

    print(f"\n  {'='*88}")
    print(f"  PANN CNN14 — Model Metrics & Interpretation Guide")
    print(f"  {'='*88}")
    print(f"  {'Metric':<22}  {'Value':>8}  {'Target':>8}  {'Status':<4}  Description")
    print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*4}  {'─'*38}")

    for name, val, unit, target, higher_is_better, desc in rows:
        if val is None:
            val_str = "   N/A  "
            status = "──"
        else:
            val_str = f"{val:>6.2f}{unit}"
            ok = (val >= target) if higher_is_better else (val < target)
            status = "✅" if ok else "❌"
        tgt_str = f"{target:>6.1f}{unit}" if unit else f"{target:>7.2f} "
        print(f"  {name:<22}  {val_str:>8}  {tgt_str:>8}  {status:<4}  {desc}")

    print(f"  {'─'*88}")
    print(f"")
    print(f"  How to read these metrics:")
    print(f"  • Accuracy        — simple overall score")
    print(f"  • Macro F1        — better measure for imbalanced classes")
    print(f"  • AUC-ROC         — 0.5=random  0.88+=clinically useful  1.0=perfect")
    print(
        f"  • Sensitivity     — high = few real wheeze cases missed (clinically critical)"
    )
    print(f"  • Specificity     — high = few false alarms on healthy patients")
    print(
        f"  • Wheeze Prob     — <{threshold:.0f}% = NORMAL,  >={threshold:.0f}% = WHEEZING DETECTED"
    )
    print(f"")
    print(f"  Tip: run  python evaluate_pann.py  to regenerate full test-set metrics.")
    print(f"  {'='*88}\n")


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="WheezeDetectionAI — PANN Inference")
    parser.add_argument("audio", type=str, help="Path to WAV file")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model path (default: {PANN_MODEL_PATH})",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=1.0,
        help="Sliding window stride in seconds (default 1.0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Wheeze %% threshold for verdict (default 50.0)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show per-clip predictions"
    )

    # ── Cleanup-stage flags ─────────────────────────────────────
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Skip the noisereduce + band-pass cleanup stage",
    )
    parser.add_argument(
        "--lowcut",
        type=float,
        default=LOWCUT_HZ,
        help=f"Band-pass low cutoff Hz, removes heart sounds/rumble (default {LOWCUT_HZ:.0f})",
    )
    parser.add_argument(
        "--highcut",
        type=float,
        default=HIGHCUT_HZ,
        help=f"Band-pass high cutoff Hz, removes hiss/sibilants (default {HIGHCUT_HZ:.0f})",
    )
    parser.add_argument(
        "--denoise-strength",
        type=float,
        default=0.85,
        help="noisereduce prop_decrease, 0-1 (default 0.85)",
    )
    parser.add_argument(
        "--vad-speech-removal",
        action="store_true",
        help="Additionally zero out detected speech using WebRTC VAD (experimental)",
    )
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="WebRTC VAD aggressiveness 0(least)-3(most) (default 2)",
    )
    args = parser.parse_args()

    print(f"\n  WheezeDetectionAI — PANN CNN14 Inference")
    print(f"  File   : {args.audio}")
    print(f"  Stride : {args.stride}s  |  Threshold: {args.threshold}%\n")

    results = predict_file(
        args.audio,
        model_path=args.model,
        stride_sec=args.stride,
        denoise=not args.no_denoise,
        lowcut=args.lowcut,
        highcut=args.highcut,
        denoise_strength=args.denoise_strength,
        vad_speech_removal=args.vad_speech_removal,
        vad_aggressiveness=args.vad_aggressiveness,
    )
    n_clips = len(results)
    wheeze_pct = (
        float(np.mean([r["probabilities"].get("wheeze", 0.0) for r in results])) * 100
    )
    normal_pct = 100.0 - wheeze_pct
    verdict = "🚨 WHEEZING DETECTED" if wheeze_pct >= args.threshold else "✅ NORMAL"
    fname = os.path.basename(args.audio)

    # ── Per-clip breakdown (verbose) ──────────────────────────
    if args.verbose:
        print(f"  {'Time':>6}  {'Prediction':<14}  {'Conf':>6}  Probabilities")
        print(f"  {'─'*65}")
        for r in results:
            probs_str = "  ".join(
                f"{n[:3]}={p*100:.0f}%" for n, p in r["probabilities"].items()
            )
            print(
                f"  {r['time']:>5.1f}s  {r['predicted_class']:<14}"
                f"  {r['confidence']*100:>5.1f}%  {probs_str}"
            )
        print()

    # ── Summary ───────────────────────────────────────────────
    from collections import Counter

    counts = Counter(r["predicted_class"] for r in results)

    print(f"  File                 : {fname}")
    print(f"  Clips analysed       : {n_clips}")
    print(f"  ──────────────────────────────────────────")
    print(f"  Inspiration clips    :  {counts.get('inspiration', 0):>4}")
    print(f"  Expiration clips     :  {counts.get('expiration',  0):>4}")
    print(f"  Abnormal clips       :  {counts.get('abnormal',    0):>4}")
    print(f"  ──────────────────────────────────────────")
    print(f"  Wheezing Probability :  {wheeze_pct:>5.1f} %")
    print(f"  Normal Probability   :  {normal_pct:>5.1f} %")
    print(f"  Verdict              :  {verdict}")
    print()

    print_metrics_guide(wheeze_pct, args.threshold)


if __name__ == "__main__":
    main()
