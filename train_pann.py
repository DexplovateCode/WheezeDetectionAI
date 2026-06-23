# ============================================================
#  WheezeDetectionAI — PANN Training Script
#  Uses class_weights already computed in dataset.py
# ============================================================
import torch, torch.nn as nn, os, time
import config
from model_pann import Cnn14FineTuned
from dataset import get_loaders

PRETRAINED_PATH = os.path.join(config.BASE_DIR, "pretrained", "Cnn14_16k.pth")
PANN_MODEL_PATH = os.path.join(config.MODEL_DIR, "pann_best_model.pth")

def run_epoch(model, loader, criterion, optimizer, scaler, device, training=True):
    model.train() if training else model.eval()
    total_loss, correct = 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for specs, labels in loader:
            specs, labels = specs.to(device), labels.to(device)
            with torch.amp.autocast("cuda", enabled=config.USE_AMP):
                out  = model(specs)
                loss = criterion(out, labels)
            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item()
            correct    += (out.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset) * 100

def train_pann():
    device = torch.device(config.DEVICE)
    print(f"\n{'='*55}")
    print(f"  PANN CNN14 Fine-Tuning")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"{'='*55}\n")

    # ── Use class_weights from dataset.py directly ────────────
    train_loader, val_loader, _, class_weights = get_loaders(augment_train=True)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)   # already perfectly computed
    )
    print(f"\n  CrossEntropyLoss with dataset class_weights ✅\n")

    model  = Cnn14FineTuned(
        num_classes=config.NUM_CLASSES,
        pretrained_path=PRETRAINED_PATH
    ).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR,   exist_ok=True)

    best_val, patience_count = float("inf"), 0

    # ── Phase 1: Head only (frozen backbone, 10 epochs) ───────
    print("  Phase 1: Head only — backbone frozen")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=config.WEIGHT_DECAY)

    for epoch in range(1, 11):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True)
        vl, va = run_epoch(model, val_loader,   criterion, optimizer, scaler, device, False)
        print(f"  [P1] Epoch {epoch:02d} | Train {tl:.4f}/{ta:.1f}% | Val {vl:.4f}/{va:.1f}% | {time.time()-t0:.1f}s")
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), PANN_MODEL_PATH)
            print(f"    ✅ Best model saved")

    # ── Phase 2: Full fine-tuning ─────────────────────────────
    print("\n  Phase 2: Full fine-tuning — all layers unfrozen")
    model.unfreeze_all()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4,
        weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True)
    patience_count = 0

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True)
        vl, va = run_epoch(model, val_loader,   criterion, optimizer, scaler, device, False)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  [P2] Epoch {epoch:02d} | Train {tl:.4f}/{ta:.1f}% | Val {vl:.4f}/{va:.1f}% | LR {lr_now:.6f} | {time.time()-t0:.1f}s")
        scheduler.step(vl)
        if vl < best_val:
            best_val = vl; patience_count = 0
            torch.save(model.state_dict(), PANN_MODEL_PATH)
            print(f"    ✅ Best model saved (val_loss={vl:.4f})")
        else:
            patience_count += 1
            if patience_count >= config.PATIENCE:
                print("  Early stopping."); break

    print(f"\n  ✅ Done! Model: {PANN_MODEL_PATH}")

if __name__ == "__main__":
    train_pann()
