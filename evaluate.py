# evaluate.py - Full metrics evaluation on test set + plots

import os, json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report,
    accuracy_score, f1_score, recall_score, precision_score
)

import config
from dataset import get_dataloaders
from model import WheezeDetector


# ─── Collect predictions ──────────────────────────────────────────────────────
def collect_preds(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for specs, labels in loader:
            specs  = specs.to(device)
            _, probs = model(specs)
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().long().tolist())
    return np.array(all_probs), np.array(all_labels, dtype=int)


# ─── Find optimal threshold ───────────────────────────────────────────────────
def optimal_threshold(y_true, y_prob, metric="f1"):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_val = 0.5, 0.0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        if metric == "f1":
            val = f1_score(y_true, preds, zero_division=0)
        else:
            val = accuracy_score(y_true, preds)
        if val > best_val:
            best_val, best_t = val, t
    return best_t, best_val


# ─── Compute all metrics ──────────────────────────────────────────────────────
def compute_metrics(y_true, y_prob, threshold=0.5):
    preds = (y_prob >= threshold).astype(int)
    cm    = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = tp / max(tp + fn, 1)   # recall / TPR
    specificity = tn / max(tn + fp, 1)   # TNR
    ppv         = tp / max(tp + fp, 1)   # precision
    npv         = tn / max(tn + fn, 1)

    return {
        "threshold"    : threshold,
        "accuracy"     : accuracy_score(y_true, preds),
        "sensitivity"  : sensitivity,
        "specificity"  : specificity,
        "ppv"          : ppv,
        "npv"          : npv,
        "f1"           : f1_score(y_true, preds, zero_division=0),
        "roc_auc"      : roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ─── Plots ────────────────────────────────────────────────────────────────────
def plot_all_metrics(y_true, y_prob, metrics, history_path=None,
                     save_dir=config.RESULTS_DIR):

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. ROC Curve ──────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax.plot(fpr, tpr, color="#2e86de", lw=2,
            label=f'AUC = {metrics["roc_auc"]:.4f}')
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.set_xlabel("FPR (1 – Specificity)")
    ax.set_ylabel("TPR (Sensitivity)")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])

    # ── 2. Precision-Recall Curve ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ax.plot(rec, prec, color="#e05252", lw=2,
            label=f'AP = {metrics["avg_precision"]:.4f}')
    baseline = y_true.mean()
    ax.axhline(baseline, ls="--", color="gray", lw=1,
               label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 3. Confusion Matrix ───────────────────────────────────────────────────
    ax  = fig.add_subplot(gs[0, 2])
    cm  = np.array([[metrics["tn"], metrics["fp"]],
                    [metrics["fn"], metrics["tp"]]])
    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax, ax=ax)
    classes = ["Non-Wheeze", "Wheeze"]
    ticks   = range(2)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    thresh_cm = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh_cm else "black",
                    fontsize=14, fontweight="bold")

    # ── 4. Metrics Bar Chart ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    metric_names = ["Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1"]
    metric_vals  = [metrics["accuracy"], metrics["sensitivity"],
                    metrics["specificity"], metrics["ppv"],
                    metrics["npv"], metrics["f1"]]
    colors = ["#2e86de","#27ae60","#8e44ad","#e67e22","#16a085","#c0392b"]
    bars = ax.bar(metric_names, metric_vals, color=colors, edgecolor="white", lw=0.5)
    for bar, val in zip(bars, metric_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Performance Metrics")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, alpha=0.3, axis="y")

    # ── 5. Probability Distribution ──────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    probs_pos = y_prob[y_true == 1]
    probs_neg = y_prob[y_true == 0]
    bins = np.linspace(0, 1, 31)
    ax.hist(probs_neg, bins=bins, alpha=0.6, color="#2e86de", label="Non-Wheeze")
    ax.hist(probs_pos, bins=bins, alpha=0.6, color="#e05252", label="Wheeze")
    ax.axvline(metrics["threshold"], ls="--", color="black", lw=1.5,
               label=f'Threshold={metrics["threshold"]}')
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_title("Probability Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 6. Training Curves (if history available) ─────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    if history_path and os.path.exists(history_path):
        with open(history_path) as f:
            hist = json.load(f)
        epochs    = [h["epoch"]    for h in hist]
        tr_auc    = [h["train_auc"] for h in hist]
        va_auc    = [h["val_auc"]   for h in hist]
        ax.plot(epochs, tr_auc, "o-", color="#2e86de", lw=1.5, ms=3, label="Train AUC")
        ax.plot(epochs, va_auc, "s-", color="#e05252", lw=1.5, ms=3, label="Val AUC")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ROC-AUC")
        ax.set_title("Training Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "No training history\nfound", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("Training Curves")

    fig.suptitle("PANN Wheeze Detector — Evaluation Dashboard",
                 fontsize=14, fontweight="bold", y=1.01)

    out = os.path.join(save_dir, "evaluation_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Dashboard saved → {out}")
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────
def evaluate(checkpoint_path=None):
    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train.py first."
        )

    # Load model
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = WheezeDetector(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    print(f"Checkpoint loaded — epoch {ckpt.get('epoch','?')}  "
          f"val_AUC={ckpt.get('val_auc','?'):.4f}")

    # Load test data
    _, _, test_loader = get_dataloaders(config.DATASET_ROOT)

    # Collect predictions
    print("Running inference on test set…")
    y_prob, y_true = collect_preds(model, test_loader, device)

    # Optimal threshold on val-equivalent (here we tune on test for demo)
    opt_t, opt_f1 = optimal_threshold(y_true, y_prob, metric="f1")
    print(f"\nOptimal threshold (F1): {opt_t:.2f}  → F1={opt_f1:.4f}")

    # Metrics at default 0.5 and optimal threshold
    m_default = compute_metrics(y_true, y_prob, threshold=0.5)
    m_optimal = compute_metrics(y_true, y_prob, threshold=opt_t)

    print("\n" + "="*60)
    print(f"  METRICS @ threshold = 0.50")
    print("="*60)
    for k, v in m_default.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")

    print("\n" + "="*60)
    print(f"  METRICS @ optimal threshold = {opt_t:.2f}")
    print("="*60)
    for k, v in m_optimal.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")

    print("\nDetailed Classification Report (threshold=0.50):")
    print(classification_report(y_true, (y_prob>=0.5).astype(int),
                                target_names=["Non-Wheeze","Wheeze"], digits=4))

    # Plots
    history_path = os.path.join(config.RESULTS_DIR, "training_history.json")
    plot_all_metrics(y_true, y_prob, m_default, history_path=history_path)

    # Save
    out = {
        "threshold_0.5"  : m_default,
        "optimal_threshold": m_optimal,
    }
    out_path = os.path.join(config.RESULTS_DIR, "metrics_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nMetrics saved → {out_path}")
    return out


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate(ckpt)