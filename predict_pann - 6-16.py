# ============================================================
#  WheezeDetectionAI — PANN Inference Script
#  Usage:
#    python predict_pann.py path/to/recording.wav
#    python predict_pann.py path/to/audio.wav --stride 0.5
#    python predict_pann.py path/to/audio.wav --threshold 60
# ============================================================

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from config     import (
    CLASS_NAMES, NUM_CLASSES,
    SAMPLE_RATE, SEG_SAMPLES, SEG_DURATION,
)
from model_pann import Cnn14FineTuned
from preprocess import read_wav, extract_log_mel

PANN_MODEL_PATH = os.path.join("models", "pann_best_model.pth")


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


# ── Sliding window prediction ─────────────────────────────────
def predict_file(wav_path: str, model=None, model_path=None, stride_sec=1.0):
    if model is None:
        model = load_pann_model(model_path)

    audio       = read_wav(wav_path)
    stride_samp = int(stride_sec * SAMPLE_RATE)
    results     = []

    start = 0
    while start + SEG_SAMPLES <= len(audio):
        seg = audio[start : start + SEG_SAMPLES].copy()
        seg = (seg - seg.mean()) / (seg.std() + 1e-6)

        lm     = extract_log_mel(seg)
        lm     = (lm - lm.mean()) / (lm.std() + 1e-6)
        tensor = torch.tensor(lm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1)[0].numpy()

        predicted = CLASS_NAMES[probs.argmax()]
        results.append({
            "time"           : round(start / SAMPLE_RATE, 1),
            "predicted_class": predicted,
            "confidence"     : float(probs.max()),
            "probabilities"  : {n: float(p) for n, p in zip(CLASS_NAMES, probs)}
        })
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

    accuracy  = eval_data.get("accuracy",  None)
    macro_f1  = eval_data.get("macro_f1",  None)
    auc_roc   = eval_data.get("auc_roc",   None)
    per_class = eval_data.get("per_class", {})
    w_sens    = per_class.get("abnormal", {}).get("sensitivity", None)
    w_spec    = per_class.get("abnormal", {}).get("specificity", None)

    rows = [
        ("Accuracy",           accuracy,   "%",  85.0,  True,  "Overall correct predictions across all classes"),
        ("Macro F1",           macro_f1,   "%",  78.0,  True,  "Balanced score across inspiration/expiration/abnormal"),
        ("AUC-ROC",            auc_roc,    "",   0.88,  True,  "Class separation ability  (1.0 = perfect)"),
        ("Wheeze Sensitivity", w_sens,     "%",  82.0,  True,  "% of real wheeze cases correctly caught (recall)"),
        ("Wheeze Specificity", w_spec,     "%",  78.0,  True,  "% of normal cases correctly identified"),
        ("Wheeze Probability", wheeze_pct, "%",  threshold, False, "This recording — probability of abnormal breath sounds"),
    ]

    print(f"\n  {'='*88}")
    print(f"  PANN CNN14 — Model Metrics & Interpretation Guide")
    print(f"  {'='*88}")
    print(f"  {'Metric':<22}  {'Value':>8}  {'Target':>8}  {'Status':<4}  Description")
    print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*4}  {'─'*38}")

    for name, val, unit, target, higher_is_better, desc in rows:
        if val is None:
            val_str = "   N/A  "
            status  = "──"
        else:
            val_str = f"{val:>6.2f}{unit}"
            ok      = (val >= target) if higher_is_better else (val < target)
            status  = "✅" if ok else "❌"
        tgt_str = f"{target:>6.1f}{unit}" if unit else f"{target:>7.2f} "
        print(f"  {name:<22}  {val_str:>8}  {tgt_str:>8}  {status:<4}  {desc}")

    print(f"  {'─'*88}")
    print(f"")
    print(f"  How to read these metrics:")
    print(f"  • Accuracy        — simple overall score")
    print(f"  • Macro F1        — better measure for imbalanced classes")
    print(f"  • AUC-ROC         — 0.5=random  0.88+=clinically useful  1.0=perfect")
    print(f"  • Sensitivity     — high = few real wheeze cases missed (clinically critical)")
    print(f"  • Specificity     — high = few false alarms on healthy patients")
    print(f"  • Wheeze Prob     — <{threshold:.0f}% = NORMAL,  >={threshold:.0f}% = WHEEZING DETECTED")
    print(f"")
    print(f"  Tip: run  python evaluate_pann.py  to regenerate full test-set metrics.")
    print(f"  {'='*88}\n")


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="WheezeDetectionAI — PANN Inference")
    parser.add_argument("audio",       type=str,   help="Path to WAV file")
    parser.add_argument("--model",     type=str,   default=None,
                        help=f"Model path (default: {PANN_MODEL_PATH})")
    parser.add_argument("--stride",    type=float, default=1.0,
                        help="Sliding window stride in seconds (default 1.0)")
    parser.add_argument("--threshold", type=float, default=50.0,
                        help="Wheeze %% threshold for verdict (default 50.0)")
    parser.add_argument("--verbose",   action="store_true",
                        help="Show per-clip predictions")
    args = parser.parse_args()

    print(f"\n  WheezeDetectionAI — PANN CNN14 Inference")
    print(f"  File   : {args.audio}")
    print(f"  Stride : {args.stride}s  |  Threshold: {args.threshold}%\n")

    results    = predict_file(args.audio, model_path=args.model, stride_sec=args.stride)
    n_clips    = len(results)
    wheeze_pct = float(np.mean([r["probabilities"].get("abnormal", 0.0) for r in results])) * 100
    normal_pct = 100.0 - wheeze_pct
    verdict    = "🚨 WHEEZING DETECTED" if wheeze_pct >= args.threshold else "✅ NORMAL"
    fname      = os.path.basename(args.audio)

    # ── Per-clip breakdown (verbose) ──────────────────────────
    if args.verbose:
        print(f"  {'Time':>6}  {'Prediction':<14}  {'Conf':>6}  Probabilities")
        print(f"  {'─'*65}")
        for r in results:
            probs_str = "  ".join(
                f"{n[:3]}={p*100:.0f}%"
                for n, p in r["probabilities"].items()
            )
            print(f"  {r['time']:>5.1f}s  {r['predicted_class']:<14}"
                  f"  {r['confidence']*100:>5.1f}%  {probs_str}")
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