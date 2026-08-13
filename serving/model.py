from __future__ import annotations

import torch
import torch.nn as nn

import config
from schemas import AAMIClass

NUM_CLASSES = len(AAMIClass)
CLASS_ORDER = list(AAMIClass)  # 학습 시 class_4_mapping.AAMI_CLASSES_4 와 순서 동일해야 함


class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResNet1D(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=1, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(ResBlock1D(32, 32, 1), ResBlock1D(32, 32, 1))
        self.stage2 = nn.Sequential(ResBlock1D(32, 64, 2), ResBlock1D(64, 64, 1))
        self.stage3 = nn.Sequential(ResBlock1D(64, 128, 2), ResBlock1D(128, 128, 1))
        self.stage4 = nn.Sequential(ResBlock1D(128, 256, 2), ResBlock1D(256, 256, 1))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        return self.fc(x)


def load_model(checkpoint_path=None, device: str | None = None) -> tuple[nn.Module, torch.device]:
    checkpoint_path = checkpoint_path or config.MODEL_CHECKPOINT_PATH
    device = torch.device(device or config.DEVICE)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {checkpoint_path}. "
            "MODEL_CHECKPOINT_PATH 환경변수가 컨테이너 내부 경로와 일치하는지, "
            "볼륨 마운트가 되어있는지 확인하세요."
        )

    model = ResNet1D(num_classes=NUM_CLASSES, dropout=0.0)  # 추론 시 dropout 비활성 의미로 0.0
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # BatchNorm/Dropout을 추론 모드로 고정 -- 빠뜨리면 배치 통계가 흔들려 예측이 불안정해짐
    return model, device


@torch.no_grad()
def predict_beats(
    model: nn.Module, device: torch.device, windows: list  # list[np.ndarray], each (250,)
) -> list[dict]:
    """윈도우 리스트 -> [{predicted_class, class_probabilities}, ...]"""
    if not windows:
        return []

    x = torch.stack([torch.from_numpy(w) for w in windows]).unsqueeze(1).to(device)  # (N,1,250)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()  # (N, num_classes)

    results = []
    for row in probs:
        class_probs = {CLASS_ORDER[i]: float(row[i]) for i in range(NUM_CLASSES)}
        pred_idx = int(row.argmax())
        results.append({"predicted_class": CLASS_ORDER[pred_idx], "class_probabilities": class_probs})
    return results
