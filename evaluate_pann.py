# ============================================================
#  WheezeDetectionAI — PANN Model Evaluation Script
#  Metrics: Accuracy, F1, AUC-ROC, Sensitivity, Specificity
#  Model:   pann_best_model.pth
# ============================================================
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
import config
from model_pann import Cnn14FineTuned
from dataset import get_loaders

PANN_MODEL_PATH = os.path.join(config.MODEL_DIR, "pann_best_model.pth")

def evaluate_pann():
    device = torch.device(config.DEVICE)
    print(f"\n{'='*60}")
    print(f"  PANN CNN14 — Evaluation Report")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── Load test data ────────────────────────────────────────
    _, _, test_loader, _ = get_loaders(augment_train=False)

    # ── Load PANN model ───────────────────────────────────────
    if not os.path.exists(PANN_MODEL_PATH):
        raise FileNotFoundError(
            f"\n  ❌ No PANN model found at {PANN_MODEL_PATH}\n"
            f"  Run python3 train_pann.py first."
        )

    model = Cnn14FineTuned(num_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(PANN_MODEL_PATH, map_location=device))
    model.eval()
    print(f"  ✅ Loaded: {PANN_MODEL_PATH}\n")

    # ── Collect predictions ───────────────────────────────────
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for specs, labels in test_loader:
            specs  = specs.to(device)
            logits = model(specs)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    # ── Overall Accuracy ──────────────────────────────────────
    accuracy = (all_preds == all_labels).mean() * 100
    macro_f1 = f1_score(all_labels, all_preds, average="macro") * 100

    # ── AUC-ROC ───────────────────────────────────────────────
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0

    # ── Per-class metrics ─────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    per_class = {}
    for i, cls in enumerate(config.CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
        precision   = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
        f1          = (2 * tp) / (2 * tp + fp + fn) * 100 if (2*tp+fp+fn) > 0 else 0.0
        per_class[cls] = {
            "sensitivity": round(sensitivity, 2),
            "specificity": round(specificity, 2),
            "precision"  : round(precision,   2),
            "f1"         : round(f1,           2),
            "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn)
        }

    # ── Print Report ──────────────────────────────────────────
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*37}")
    print(f"  {'Overall Accuracy':<25} {accuracy:>9.2f}%")
    print(f"  {'Macro F1':<25} {macro_f1:>9.2f}%")
    print(f"  {'AUC-ROC':<25} {auc:>10.4f}")

    print(f"\n  {'Class':<14} {'Sensitivity':>12} {'Specificity':>12} {'Precision':>10} {'F1':>8}")
    print(f"  {'-'*60}")
    for cls, m in per_class.items():
        print(f"  {cls:<14} {m['sensitivity']:>11.1f}%"
              f" {m['specificity']:>11.1f}%"
              f" {m['precision']:>9.1f}%"
              f" {m['f1']:>7.1f}%")

    # ── Confusion Matrix ──────────────────────────────────────
    print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    print(f"  Labels: {config.CLASS_NAMES}")
    for row in cm:
        print(f"  {row.tolist()}")

    # ── Clinical Threshold Check ──────────────────────────────
    # Uses 'abnormal' class as the wheeze/abnormal detector
    abnormal = per_class.get("abnormal", per_class.get("wheeze", {}))
    print(f"\n  Clinical Threshold Check:")
    print(f"  {'Wheeze/Abnormal Sensitivity ≥82%?':<35} "
          f"{'✅' if abnormal.get('sensitivity',0) >= 82 else '❌'}"
          f"  ({abnormal.get('sensitivity',0):.1f}%)")
    print(f"  {'Wheeze/Abnormal Specificity ≥78%?':<35} "
          f"{'✅' if abnormal.get('specificity',0) >= 78 else '❌'}"
          f"  ({abnormal.get('specificity',0):.1f}%)")
    print(f"  {'AUC-ROC ≥0.88?':<35} "
          f"{'✅' if auc >= 0.88 else '❌'}  ({auc:.4f})")
    print(f"  {'Macro F1 ≥78%?':<35} "
          f"{'✅' if macro_f1 >= 78 else '❌'}  ({macro_f1:.2f}%)")
    print(f"  {'Overall Accuracy ≥85%?':<35} "
          f"{'✅' if accuracy >= 85 else '❌'}  ({accuracy:.2f}%)")

    # ── Classification Report ─────────────────────────────────
    print(f"\n  Sklearn Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=config.CLASS_NAMES,
        digits=4
    ))

    # ── Save results ──────────────────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    results = {
        "model"           : "PANN CNN14",
        "accuracy"        : round(accuracy, 4),
        "macro_f1"        : round(macro_f1, 4),
        "auc_roc"         : round(auc, 4),
        "per_class"       : per_class,
        "confusion_matrix": cm.tolist()
    }
    out_path = os.path.join(config.OUTPUT_DIR, "pann_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {out_path}")
    print(f"{'='*60}\n")
    return results

if __name__ == "__main__":
    evaluate_pann()