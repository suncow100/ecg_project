"""
train.py

Train ResNet1D ECG arrhythmia classifier (4-class: N/S/V/F).

Design:
- DS1 internal validation split (record-level) for checkpoint selection
- DS2 untouched until final evaluation
- Class-weighted CrossEntropyLoss
- Optimizer / scheduler / dropout are CLI-configurable so that changes can
  be ablated ONE AT A TIME instead of bundled together. Bundling multiple
  regularization changes (weight decay + dropout + LR schedule) into a
  single run makes any performance delta impossible to attribute -- this
  script exists specifically to avoid that trap.

Usage:
    # baseline: plain Adam, no weight decay, no scheduler, no dropout
    python train.py --optimizer adam --weight-decay 0 --scheduler none --dropout 0.0 --run-name baseline

    # test ONLY AdamW + weight decay, everything else held at baseline
    python train.py --optimizer adamw --weight-decay 1e-4 --scheduler none --dropout 0.0 --run-name adamw_only

    # test ONLY the cosine scheduler
    python train.py --optimizer adam --weight-decay 0 --scheduler cosine --dropout 0.0 --run-name cosine_only

    # test ONLY dropout
    python train.py --optimizer adam --weight-decay 0 --scheduler none --dropout 0.3 --run-name dropout_only

    # combine once you know which individual changes actually help
    python train.py --optimizer adamw --weight-decay 1e-4 --scheduler cosine --dropout 0.3 --run-name combined
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import mlflow

import config
from class_4_mapping import AAMI_CLASSES_4
from model import ResNet1D

NUM_CLASSES = len(AAMI_CLASSES_4)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train ResNet1D 4-class ECG classifier")
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=None, help="overrides config.LEARNING_RATE if set")
    p.add_argument("--epochs", type=int, default=None, help="overrides config.TRAIN_EPOCHS if set")
    p.add_argument("--run-name", type=str, default=None, help="MLflow run name, for telling ablation runs apart")
    return p.parse_args()


# ---------------------------------------------------------------------------
def load_data():
    d = config.DATASET_DIR
    X_train = np.load(d / "X_train.npy")
    y_train = np.load(d / "y_train.npy")
    record_id_train = np.load(d / "record_id_train.npy")
    X_test = np.load(d / "X_test.npy")
    y_test = np.load(d / "y_test.npy")
    return X_train, y_train, record_id_train, X_test, y_test


def make_internal_val_split(
    record_id_train: np.ndarray,
    val_ratio: float,
    force_subtrain_records: set,
    seed: int,
):
    """Record-level split of DS1 into (sub-train, internal-val). Prevents
    intra-patient leakage between the split used for early stopping and the
    split used for training, same principle as the DS1/DS2 split itself."""
    records = sorted(set(record_id_train.tolist()))
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)

    free_records = [r for r in shuffled if r not in force_subtrain_records]
    n_val = int(round(len(records) * val_ratio))
    val_records = set(free_records[:n_val])

    is_val = np.isin(record_id_train, list(val_records))
    is_subtrain = ~is_val
    return is_subtrain, is_val


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train(mode=train)
    total_loss, n_samples = 0.0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
    return total_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


@dataclass
class TrainState:
    best_f1: float = -1.0
    best_epoch: int = -1
    epochs_since_improve: int = 0


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    epochs = args.epochs or config.TRAIN_EPOCHS
    lr = args.lr or config.LEARNING_RATE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    print(f"[INFO] optimizer={args.optimizer} weight_decay={args.weight_decay} "
          f"scheduler={args.scheduler} dropout={args.dropout} lr={lr} epochs={epochs}")

    X_train, y_train, record_id_train, X_test, y_test = load_data()

    is_subtrain, is_val = make_internal_val_split(
        record_id_train, config.INTERNAL_VAL_RATIO, config.FORCE_TRAIN_RECORDS, config.INTERNAL_VAL_SEED,
    )
    print(f"[INFO] subtrain={is_subtrain.sum()} val={is_val.sum()} test={len(y_test)}")

    class_weights = compute_class_weights(y_train[is_subtrain], NUM_CLASSES)
    print(f"[INFO] class weights ({AAMI_CLASSES_4}): {class_weights.tolist()}")

    def make_loader(X, y, shuffle):
        X_t = torch.from_numpy(X).float().unsqueeze(1)
        y_t = torch.from_numpy(y).long()
        return DataLoader(TensorDataset(X_t, y_t), batch_size=config.BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(X_train[is_subtrain], y_train[is_subtrain], True)
    val_loader = make_loader(X_train[is_val], y_train[is_val], False)
    test_loader = make_loader(X_test, y_test, False)

    model = ResNet1D(num_classes=NUM_CLASSES, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    checkpoint_path = config.CHECKPOINT_DIR / f"best_model_{args.run_name or 'run'}.pt"

    state = TrainState()
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "classes": "N/S/V/F",
            "epochs": epochs,
            "batch_size": config.BATCH_SIZE,
            "lr": lr,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "dropout": args.dropout,
        })

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            val_pred, val_true = evaluate(model, val_loader, device)
            val_f1 = macro_f1(val_true, val_pred)

            # capture LR BEFORE stepping the scheduler, so the logged value
            # reflects the LR actually used during this epoch's training
            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()

            improved = val_f1 > state.best_f1
            if improved:
                state.best_f1, state.best_epoch, state.epochs_since_improve = val_f1, epoch, 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                state.epochs_since_improve += 1

            dt = time.time() - t0
            print(f"[{epoch:03d}/{epochs}] loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_f1={val_f1:.4f} lr={current_lr:.2e} {'*' if improved else ''} ({dt:.1f}s)")

            mlflow.log_metrics({
                "train_loss": train_loss, "val_loss": val_loss,
                "val_f1_macro": val_f1, "lr": current_lr,
            }, step=epoch)

            if state.epochs_since_improve >= config.EARLY_STOP_PATIENCE:
                print(f"[INFO] early stopping at epoch {epoch} "
                      f"(best was epoch {state.best_epoch} with val_f1={state.best_f1:.4f})")
                break

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        test_pred, test_true = evaluate(model, test_loader, device)
        test_f1 = macro_f1(test_true, test_pred)

        print(f"\n[RESULT] best epoch: {state.best_epoch}")
        print(f"[RESULT] test macro F1: {test_f1:.4f}")

        report = classification_report(test_true, test_pred, target_names=AAMI_CLASSES_4, zero_division=0)
        print(report)
        cm = confusion_matrix(test_true, test_pred)
        print("Confusion Matrix (rows=true, cols=pred), order N/S/V/F:")
        print(cm)

        mlflow.log_metric("test_macro_f1", test_f1)
        mlflow.log_text(report, "classification_report.txt")
        mlflow.log_artifact(str(checkpoint_path))


if __name__ == "__main__":
    main()