# train.py - Training pipeline for PANN Wheeze Detector on HF_Lung_V1

import os, time, random, json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)

import config
from dataset import get_dataloaders
from model import build_model


# ─── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Mixup ────────────────────────────────────────────────────────────────────
def mixup_batch(x, y, alpha=config.MIXUP_ALPHA):
    if alpha <= 0:
        return x, y
    lam  = np.random.beta(alpha, alpha)
    idx  = torch.randperm(x.size(0), device=x.device)
    x_m  = lam * x + (1 - lam) * x[idx]
    y_m  = lam * y + (1 - lam) * y[idx]
    return x_m, y_m


# ─── One-epoch helpers ────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device,
              is_train: bool, epoch: int, warmup_epochs: int):
    model.train(is_train)
    total_loss, all_probs, all_labels = 0.0, [], []

    with torch.set_grad_enabled(is_train):
        for step, (specs, labels) in enumerate(loader):
            specs  = specs.to(device, non_blocking=True)   # (B,1,Mel,T)
            labels = labels.to(device, non_blocking=True)  # (B,)

            if is_train and config.MIXUP_ALPHA > 0:
                specs, labels = mixup_batch(specs, labels)

            logits, probs = model(specs)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * specs.size(0)
            all_probs.extend(probs.detach().cpu().tolist())
            # Store hard labels (un-mixed) for metrics
            all_labels.extend(labels.detach().cpu().round().long().tolist())

    n       = len(all_labels)
    avg_loss= total_loss / n
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels, dtype=int)

    preds  = (all_probs >= 0.5).astype(int)
    try:
        auc = roc_auc_score(all_labels, all_probs)
        ap  = average_precision_score(all_labels, all_probs)
    except ValueError:
        auc, ap = 0.0, 0.0

    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
    acc = accuracy_score(all_labels, preds)
    f1  = f1_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    pre = precision_score(all_labels, preds, zero_division=0)

    return dict(loss=avg_loss, auc=auc, ap=ap, acc=acc,
                f1=f1, recall=rec, precision=pre)


# ─── Main Training Loop ───────────────────────────────────────────────────────
def train():
    set_seed()
    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )
    print(f"\n{'='*60}")
    print(f" PANN Wheeze Detector — Training")
    print(f" Device : {device}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(config.DATASET_ROOT)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(device)

    # ── Loss: BCEWithLogitsLoss with auto pos_weight ─────────────────────────
    if config.POS_WEIGHT is None:
        # Compute from training labels
        train_labels = [c for _, _, c in train_loader.dataset.clips]
        n_pos = sum(train_labels)
        n_neg = len(train_labels) - n_pos
        pw    = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
        print(f"Auto pos_weight = {pw.item():.2f}")
    else:
        pw = torch.tensor([config.POS_WEIGHT], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    # ── Optimiser & Scheduler ────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=config.LR,
                      weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS,
                                   eta_min=config.LR_MIN)

    # ── Training loop ────────────────────────────────────────────────────────
    best_auc   = 0.0
    patience_c = 0
    history    = []

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()

        # Warmup: freeze backbone for first N epochs
        if epoch <= config.WARMUP_EPOCHS:
            for p in model.backbone.parameters():
                p.requires_grad = False
        else:
            for p in model.backbone.parameters():
                p.requires_grad = True

        tr = run_epoch(model, train_loader, criterion, optimizer,
                       device, is_train=True, epoch=epoch,
                       warmup_epochs=config.WARMUP_EPOCHS)
        va = run_epoch(model, val_loader, criterion, optimizer,
                       device, is_train=False, epoch=epoch,
                       warmup_epochs=config.WARMUP_EPOCHS)

        scheduler.step()
        elapsed = time.time() - t0

        row = dict(epoch=epoch, **{f"train_{k}": v for k, v in tr.items()},
                   **{f"val_{k}": v for k, v in va.items()})
        history.append(row)

        print(
            f"Ep {epoch:03d}/{config.NUM_EPOCHS} | "
            f"T-loss {tr['loss']:.4f} AUC {tr['auc']:.4f} F1 {tr['f1']:.4f} | "
            f"V-loss {va['loss']:.4f} AUC {va['auc']:.4f} F1 {va['f1']:.4f} "
            f"Acc {va['acc']:.4f} | {elapsed:.1f}s"
        )

        # Save best checkpoint
        if va['auc'] > best_auc:
            best_auc   = va['auc']
            patience_c = 0
            ckpt_path  = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch"     : epoch,
                "state_dict": model.state_dict(),
                "val_auc"   : best_auc,
                "val_f1"    : va['f1'],
                "config"    : {
                    "SAMPLE_RATE"  : config.SAMPLE_RATE,
                    "CLIP_DURATION": config.CLIP_DURATION,
                    "N_MELS"       : config.N_MELS,
                    "N_FFT"        : config.N_FFT,
                    "HOP_LENGTH"   : config.HOP_LENGTH,
                    "FMIN"         : config.FMIN,
                    "FMAX"         : config.FMAX,
                }
            }, ckpt_path)
            print(f"  ✓ Best model saved (val AUC={best_auc:.4f})")
        else:
            patience_c += 1
            if patience_c >= config.PATIENCE:
                print(f"\n Early stopping at epoch {epoch} "
                      f"(no improvement for {config.PATIENCE} epochs).")
                break

    # ── Save training history ─────────────────────────────────────────────────
    hist_path = os.path.join(config.RESULTS_DIR, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved → {hist_path}")

    # ── Final evaluation on test set ─────────────────────────────────────────
    print("\n" + "="*60)
    print(" Test-set evaluation")
    print("="*60)
    evaluate_on_test(model, test_loader, device, criterion)


# ─── Test Evaluation ──────────────────────────────────────────────────────────
def evaluate_on_test(model, test_loader, device, criterion):
    model.eval()
    all_probs, all_labels, total_loss = [], [], 0.0

    with torch.no_grad():
        for specs, labels in test_loader:
            specs  = specs.to(device)
            labels = labels.to(device)
            logits, probs = model(specs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * specs.size(0)
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().long().tolist())

    n          = len(all_labels)
    avg_loss   = total_loss / n
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels, dtype=int)
    preds      = (all_probs >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs)
    ap  = average_precision_score(all_labels, all_probs)
    cm  = confusion_matrix(all_labels, preds)
    report = classification_report(
        all_labels, preds,
        target_names=["Non-Wheeze", "Wheeze"], digits=4
    )

    print(f"\nTest Loss : {avg_loss:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"Avg Prec  : {ap:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    # Save results
    results = dict(
        test_loss=avg_loss, roc_auc=auc, avg_precision=ap,
        confusion_matrix=cm.tolist(),
        report=report
    )
    res_path = os.path.join(config.RESULTS_DIR, "test_results.json")
    with open(res_path, "w") as f:
        json.dump({k: v for k, v in results.items() if k != "report"}, f, indent=2)
    print(f"\nTest results saved → {res_path}")
    return results


if __name__ == "__main__":
    train()