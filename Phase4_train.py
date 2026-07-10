"""
Phase 4 · 1D-CNN 분류 모델 학습
==================================
Phase 3의 ECGBeatDataset / build_dataloaders를 받아
1D-CNN을 학습하고 MLflow에 기록합니다.

모델 설계 원칙:
  - Global Average Pooling (GAP) 사용 이유:
      Max Pooling은 노이즈 spike 하나에 과민 반응
      GAP는 beat 전체의 평균 활성값을 pooling → 노이즈 내성 강화
      R-peak 위치가 annotation 기반(train)과 NeuroKit2(deploy) 간에
      ±50ms 어긋나도 평행이동에 덜 민감 (Phase 5 ±50ms 매칭과 결이 맞음)
  - Residual connection: gradient vanishing 방지 + 깊은 층에서 형태 보존
  - Q class weight cap: 1220 → 50으로 클리핑 (학습 불안정 방지)
"""

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from Phase1_data_preprocessing import AAMI_CLASS_NAMES
from Phase3_dataset import Phase3Config, build_dataloaders


# ============================================================
# 0. Config
# ============================================================
@dataclass
class Phase4Config:
    # ── 경로 ──────────────────────────────────────────────
    dae_checkpoint: str  = './checkpoints/dae_best.pt'
    save_dir: str        = './checkpoints'
    mlflow_experiment: str = 'ecg_classifier'

    # ── 학습 하이퍼파라미터 ───────────────────────────────
    epochs: int          = 80
    batch_size: int      = 64
    lr: float            = 5e-4
    weight_decay: float  = 1e-4
    early_stop_patience: int = 20

    # ── Q class 가중치 cap ────────────────────────────────
    # Phase 3에서 Q weight=1220으로 학습 불안정 경고가 발생했으므로
    # 50으로 클리핑. N/S/V/F 비율은 그대로 유지됨
    weight_cap: float    = 50.0

    # ── 모델 구조 ──────────────────────────────────────────
    in_channels: int     = 1
    base_channels: int   = 32
    num_classes: int     = 5

    # ── 기타 ───────────────────────────────────────────────
    num_workers: int     = 4
    seed: int            = 42
    device: str          = field(
        default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu'
    )


# ============================================================
# 1. 모델: Residual 1D-CNN + Global Average Pooling
# ============================================================
class ResBlock1D(nn.Module):
    """
    1D Residual Block.
    skip connection으로 gradient vanishing 방지.
    채널 수가 바뀌면 1×1 conv로 projection.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, stride: int = 1):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, stride=1, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        # 채널/stride 불일치 시 projection shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ECGClassifier1DCNN(nn.Module):
    """
    Input : (B, 1, 250) — 비대칭 beat (앞 100 / 뒤 150)
    Output: (B, 5)       — AAMI 5-class logit

    구조:
      Stem conv (1→32, k=15)  → 초기 feature extraction, 큰 kernel로 P/QRS/T 전체 커버
      ResBlock ×3 (32→64→128) → 점진적 채널 확장, stride=2로 시간 해상도 축소
      Global Average Pooling   → 시간축 평균 → 위치 불변성 확보
      FC (128→64→5)            → 최종 분류
      Dropout(0.3)             → 과적합 방지

    GAP를 쓰는 이유:
      Max Pooling: 가장 강한 활성값 하나만 통과 → 노이즈 spike에 민감
      GAP: beat 전체 시간축의 평균 활성값 사용 → 노이즈/위치 변화에 강건
           ±50ms R-peak 위치 오차(Phase 5 매칭 허용오차)가 있어도
           GAP 이후 표현이 크게 달라지지 않음
    """

    def __init__(self, base_channels: int = 32, num_classes: int = 5):
        super().__init__()

        # Stem: 큰 kernel(15)로 P파~T파 전체 맥락 포착
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Residual blocks: stride=2마다 시간 해상도 절반
        # 250 → 125 → 63 → 32
        self.res1 = ResBlock1D(base_channels,      base_channels * 2, stride=2)
        self.res2 = ResBlock1D(base_channels * 2,  base_channels * 4, stride=2)
        self.res3 = ResBlock1D(base_channels * 4,  base_channels * 4, stride=2)

        # Global Average Pooling: 시간축 평균 → (B, 128)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.gap(x)
        return self.classifier(x)


# ============================================================
# 2. 학습 / 검증 루프
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
) -> float:
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    반환: (val_loss, all_preds, all_labels)
    sklearn metric은 호출부에서 계산 (재사용성)
    """
    model.eval()
    losses, preds, labels = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(loss_fn(logits, y).item())
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        labels.extend(y.cpu().tolist())
    return float(np.mean(losses)), np.array(preds), np.array(labels)


def print_metrics(preds: np.ndarray, labels: np.ndarray, split: str) -> Dict[str, float]:
    """per-class F1 출력 및 dict 반환 (MLflow 로깅용)."""
    class_names = [AAMI_CLASS_NAMES[i] for i in range(5)]
    print(f"\n[{split}] Classification Report:")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(labels, preds, labels=list(range(5)))
    print(f"Confusion Matrix:\n{cm}")

    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    f1_macro     = f1_score(labels, preds, average='macro', zero_division=0)
    f1_weighted  = f1_score(labels, preds, average='weighted', zero_division=0)

    metrics = {f'f1_class{i}': float(f1_per_class[i]) for i in range(5)}
    metrics['f1_macro']    = float(f1_macro)
    metrics['f1_weighted'] = float(f1_weighted)
    return metrics


# ============================================================
# 3. 전체 학습 루프
# ============================================================
def train_classifier(cfg: Phase4Config) -> str:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # ── Phase 3 DataLoader ──────────────────────────────────
    p3_cfg = Phase3Config(
        dae_checkpoint=cfg.dae_checkpoint,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    train_loader, test_loader, class_weights = build_dataloaders(
        p3_cfg, cfg.dae_checkpoint
    )

    # ── Q class 가중치 cap ────────────────────────────────
    # Phase 3 출력: Q weight=1220 → 학습 불안정
    # cap=50으로 클리핑, N/S/V/F 비율은 그대로 유지
    capped_weights = torch.clamp(class_weights, max=cfg.weight_cap)
    print(f"\n[Weighted CE] cap={cfg.weight_cap} 적용 후:")
    for i, (orig, capped) in enumerate(zip(class_weights, capped_weights)):
        marker = ' ← capped' if orig > cfg.weight_cap else ''
        print(f"  Class {i}: {orig:.2f} → {capped.item():.2f}{marker}")

    # ── 모델 / 옵티마이저 ────────────────────────────────────
    model    = ECGClassifier1DCNN(cfg.base_channels, cfg.num_classes).to(cfg.device)
    loss_fn  = nn.CrossEntropyLoss(weight=capped_weights.to(cfg.device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4
        # mode='max': f1_macro 기준으로 lr 감소
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n모델 파라미터 수: {n_params:,}")
    print(f"device: {cfg.device}")

    best_f1      = 0.0
    patience_ctr = 0
    best_ckpt    = os.path.join(cfg.save_dir, 'classifier_best.pt')

    # ── MLflow ───────────────────────────────────────────────
    mlflow.set_experiment(cfg.mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_params({
            'model':           'ECGClassifier1DCNN',
            'base_channels':   cfg.base_channels,
            'n_params':        n_params,
            'batch_size':      cfg.batch_size,
            'lr':              cfg.lr,
            'weight_decay':    cfg.weight_decay,
            'weight_cap':      cfg.weight_cap,
            'loss':            'weighted_cross_entropy',
            'pooling':         'global_average',
            'beat_len':        250,
            'beat_before':     100,
            'beat_after':      150,
            'early_stop':      cfg.early_stop_patience,
        })

        print(f"\n{'='*60}")
        print(f" Phase 4 학습 시작 (최대 {cfg.epochs}에폭)")
        print(f"{'='*60}")

        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, loss_fn, cfg.device
            )
            val_loss, preds, labels = evaluate(
                model, test_loader, loss_fn, cfg.device
            )

            f1_macro = float(
                f1_score(labels, preds, average='macro', zero_division=0)
            )
            scheduler.step(f1_macro)

            print(
                f"[Epoch {epoch:03d}/{cfg.epochs}] "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"f1_macro={f1_macro:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
            mlflow.log_metrics(
                {
                    'train_loss': train_loss,
                    'val_loss':   val_loss,
                    'f1_macro':   f1_macro,
                },
                step=epoch,
            )

            # Best 체크포인트 (f1_macro 기준)
            if f1_macro > best_f1:
                best_f1      = f1_macro
                patience_ctr = 0

                # per-class 상세 지표
                metrics = print_metrics(preds, labels, f"Epoch {epoch} Val")
                mlflow.log_metrics(metrics, step=epoch)

                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'epoch':            epoch,
                        'f1_macro':         f1_macro,
                        'metrics':          metrics,
                        'config': {
                            'base_channels': cfg.base_channels,
                            'num_classes':   cfg.num_classes,
                        },
                    },
                    best_ckpt,
                )
                print(f"  → 베스트 모델 저장 (f1_macro={f1_macro:.4f})")
            else:
                patience_ctr += 1
                if patience_ctr >= cfg.early_stop_patience:
                    print(
                        f"\nEarly stopping (f1_macro 기준 "
                        f"{cfg.early_stop_patience}에폭 개선 없음, epoch {epoch})"
                    )
                    break

        # 최종 평가
        print(f"\n{'='*60}")
        print(f" 최종 평가 — 베스트 체크포인트 로드")
        print(f"{'='*60}")
        ckpt = torch.load(best_ckpt, map_location=cfg.device)
        model.load_state_dict(ckpt['model_state_dict'])
        _, preds, labels = evaluate(model, test_loader, loss_fn, cfg.device)
        final_metrics = print_metrics(preds, labels, 'Final Test (DS2)')

        mlflow.log_metrics({f'final_{k}': v for k, v in final_metrics.items()})
        mlflow.log_artifact(best_ckpt)
        
        mlflow.pytorch.log_model(model, 'classifier_model', input_example='pickle')

        print(f"\n[Phase 4 완료]")
        print(f"  best f1_macro : {best_f1:.4f}")
        print(f"  체크포인트    : {best_ckpt}")

    return best_ckpt


# ============================================================
# 4. 배포용 로더 (Phase 6에서 호출)
# ============================================================
def load_classifier(
    checkpoint_path: str,
    device: Optional[str] = None,
) -> nn.Module:
    """
    학습된 분류기를 로드.
    Phase 6 FastAPI에서 DAE와 함께 호출됨.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(checkpoint_path, map_location=device)
    cfg_d  = ckpt['config']
    model  = ECGClassifier1DCNN(
        base_channels=cfg_d['base_channels'],
        num_classes=cfg_d['num_classes'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(
        f"[Classifier] 로드 완료 "
        f"(epoch={ckpt['epoch']}, f1_macro={ckpt['f1_macro']:.4f})"
    )
    return model


# ============================================================
# 5. 진입점
# ============================================================
if __name__ == '__main__':
    cfg = Phase4Config()
    train_classifier(cfg)