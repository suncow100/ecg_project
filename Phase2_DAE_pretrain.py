"""
Phase 2 · DAE 사전학습 (1D U-Net Denoising Autoencoder)
=========================================================
Phase 1(phase1_data_prep.py)이 반드시 먼저 실행되어야 합니다.
Phase 1의 상수/함수를 직접 import해서 사용하므로,
DS1 환자 목록·노이즈 split이 두 파일 간에 항상 동일하게 유지됩니다.

역할:
  - 30초 윈도우(pad-then-crop) 단위로 노이즈 합성 → wavelet 1차 처리
  - DAE가 wavelet 잔차(주로 EM)를 복원하도록 학습
  - 학습 완료 후 체크포인트를 고정(freeze) → Phase 3에서 고정 필터로 사용

누수 방지 원칙:
  1. DS1 환자만 사용 (DS2는 이 파일에서 절대 접근하지 않음)
  2. NSTDB train split(앞 70%)만 사용 (test split은 Phase 5 전용)
  3. clean passthrough 15% 포함 → 배포 시 노이즈 없는 윈도우에도 항등함수로 동작
  4. wavelet → DAE 순서는 노이즈 종류와 무관하게 항상 고정
     (배포 시 노이즈 종류를 알 수 없으므로 조건 분기 없음)
"""

import os
import sys
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
import mlflow
from torch.utils.data import DataLoader, Dataset

# Phase 1 상수 및 NoiseBank import
# → DS1 환자 목록·노이즈 split이 두 파일 간에 항상 동일하게 유지됨
from Phase1_data_preprocessing import (
    DS1_RECORDS,
    NOISE_TYPES,
    NOISE_TRAIN_FRACTION,
    NoiseBank,
    PathConfig,
    load_noise_bank,
    validate_noise_bank,
)


# ============================================================
# 0. Config
# ============================================================
@dataclass
class DAEConfig:
    # ── 경로 ──────────────────────────────────────────────
    mitdb_path: str = '/home/qortjsdn/projects/ecg_project/mit-bih-arrhythmia-database-1.0.0/'
    nstdb_path: str = '/home/qortjsdn/projects/ecg_project/nstdb_noise/'
    checkpoint_dir: str = './checkpoints'
    mlflow_experiment: str = 'ecg_dae_pretraining'

    # ── 신호 파라미터 ──────────────────────────────────────
    fs: int = 360
    window_sec: int = 30        # 임상 분석 단위 (10,800 샘플 @ 360Hz)
    context_sec: int = 5        # 양쪽 context: db6/L8 경계 왜곡 흡수용
                                # effective support = (2^8-1)*(12-1)+1 = 2806 샘플
                                # context 5초 = 1,800샘플이 양쪽에 있어서 중심부는 안전

    # ── 웨이블릿 디노이징 (1단계, 고정) ────────────────────
    wavelet: str = 'db6'
    wavelet_level: int = 8
    # db6/L8: approximation → ~<0.7Hz (BW 대역과 일치) → 0으로 제거
    #         detail soft-threshold → MA 일부 제거
    # EM(광대역 비정상)은 여기서 처리 불가 → DAE가 담당

    # ── 노이즈 합성 파라미터 ──────────────────────────────
    snr_levels_db: List[float] = field(
        default_factory=lambda: [0, 6, 12, 18, 24]
    )
    p_include_noise: Dict[str, float] = field(
        default_factory=lambda: {'bw': 0.5, 'ma': 0.5, 'em': 0.5}
    )
    # BW/MA/EM 각각 독립적으로 포함 여부 결정 → 0~3개 동시 합성 가능
    p_clean_passthrough: float = 0.15
    # 15% 확률로 노이즈 없이 통과 → DAE가 깨끗한 입력에도 항등함수 학습
    # 배포 시 항상 DAE를 호출하므로 이 학습이 필수

    # ── DS1 내부 환자 단위 val split ──────────────────────
    # window 단위 랜덤 split은 같은 환자의 다른 30초가 train/val에 공존
    # → 환자 고유 ECG 패턴을 외우는 overfitting을 못 잡음
    # → 환자 ID 기준 분리로 올바른 overfitting 감지
    val_patient_fraction: float = 0.2

    # ── 모델 구조 ──────────────────────────────────────────
    base_channels: int = 16
    depth: int = 4
    # 채널 수 per layer: 16 → 32 → 64 → 128 → 256 (bottleneck)
    # skip connection이 QRS 같은 sharp transient를 bottleneck 우회로 전달
    # → vanilla AE보다 고주파 디테일 보존에 유리

    # ── 학습 하이퍼파라미터 ───────────────────────────────
    batch_size: int = 16
    lr: float = 1e-3
    epochs: int = 60
    early_stop_patience: int = 8
    loss_fn: str = 'l1'         # 'l1' | 'mse' | 'huber'
    # L1: 큰 아웃라이어(EM spike)에 MSE보다 덜 민감 → ECG 복원에 적합
    num_workers: int = 4
    seed: int = 42
    device: str = field(
        default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu'
    )

    @property
    def window_len(self) -> int:
        return self.window_sec * self.fs           # 10,800

    @property
    def padded_len(self) -> int:
        return (self.window_sec + 2 * self.context_sec) * self.fs  # 14,400


# ============================================================
# 1. 웨이블릿 디노이징 (고정, 무조건 적용)
# ============================================================
def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = 'db6',
    level: int = 8,
) -> np.ndarray:
    """
    2단계 처리:
      1) Approximation(가장 저주파 계수)을 0으로 → BW 제거
         level 8에서 approximation은 ~<0.7Hz 대역만 담당
      2) Detail coefficient soft-threshold (MAD 기반 sigma 추정) → MA 일부 제거
         finest detail로 sigma 추정하는 이유:
         가장 높은 주파수 band의 계수는 실제 ECG보다 노이즈 성분이 지배적이라
         노이즈 표준편차 추정에 가장 신뢰도 높음 (Donoho & Johnstone 1994)

    EM(전 대역 비정상 잡음)은 주파수로 신호와 분리 불가 → DAE가 처리
    """
    coeffs = pywt.wavedec(signal, wavelet, mode='symmetric', level=level)
    cA, details = coeffs[0], coeffs[1:]

    # BW 제거: approximation 0으로
    cA_zeroed = np.zeros_like(cA)

    # EM/MA 제거: MAD 기반 universal threshold
    sigma = np.median(np.abs(details[-1])) / 0.6745 + 1e-12
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    details_denoised = [
        pywt.threshold(d, threshold, mode='soft') for d in details
    ]

    denoised = pywt.waverec(
        [cA_zeroed] + details_denoised, wavelet, mode='symmetric'
    )
    # waverec가 홀수 길이 처리 시 1샘플 길이 차이 발생 가능 → 맞춰줌
    n = len(signal)
    if len(denoised) > n:
        denoised = denoised[:n]
    elif len(denoised) < n:
        denoised = np.pad(denoised, (0, n - len(denoised)))
    return denoised.astype(np.float32)


# ============================================================
# 2. 노이즈 합성 (독립 랜덤, 0~3종 동시 가능)
# ============================================================
def _scale_to_snr(
    clean: np.ndarray, noise_seg: np.ndarray, target_snr_db: float
) -> np.ndarray:
    """신호 전력 기준으로 노이즈를 목표 SNR에 맞게 스케일."""
    sig_power   = np.var(clean) + 1e-8
    noise_power = np.var(noise_seg) + 1e-8
    target_noise_power = sig_power / (10 ** (target_snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return (noise_seg * scale).astype(np.float32)


def inject_noise(
    clean_window: np.ndarray,
    train_noises: Dict[str, np.ndarray],
    cfg: DAEConfig,
) -> np.ndarray:
    """
    BW / MA / EM 각각 독립 확률로 포함 여부 결정 후 합산.
    실제 임상 환경에서 노이즈 종류가 겹치는 상황을 반영.

    clean passthrough(p_clean_passthrough):
      일정 비율은 노이즈를 전혀 넣지 않음.
      → DAE가 "이미 깨끗한 입력 → 그대로 출력"을 학습
      → 배포 시 항상 DAE를 호출하므로, 이 케이스를 학습하지 않으면
         노이즈 없는 윈도우가 들어왔을 때 DAE가 오히려 신호를 왜곡할 수 있음
    """
    if random.random() < cfg.p_clean_passthrough:
        return clean_window.copy()

    noisy = clean_window.copy()
    n = len(clean_window)

    for ntype in cfg.p_include_noise:
        if random.random() >= cfg.p_include_noise[ntype]:
            continue
        noise_sig = train_noises[ntype]
        if len(noise_sig) <= n:
            continue   # 안전장치: NSTDB train split이 window보다 항상 길어야 함
        start = random.randint(0, len(noise_sig) - n - 1)
        seg = noise_sig[start : start + n]
        target_snr = random.choice(cfg.snr_levels_db)
        noisy = noisy + _scale_to_snr(clean_window, seg, target_snr)

    return noisy.astype(np.float32)


# ============================================================
# 3. DS1 내부 환자 단위 train/val split
# ============================================================
def split_ds1_patients(
    cfg: DAEConfig, seed: int
) -> Tuple[List[int], List[int]]:
    """
    DS1 환자를 환자 ID 기준으로 train/val 분리.
    window 단위 랜덤 split을 쓰지 않는 이유:
      같은 환자의 다른 30초 구간이 train/val에 공존하면
      환자 고유 ECG 패턴을 외우는 overfitting을 val loss가 감지 못함.
    """
    records = DS1_RECORDS.copy()
    rng = random.Random(seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * cfg.val_patient_fraction))
    return records[n_val:], records[:n_val]   # train, val


# ============================================================
# 4. Dataset
# ============================================================
class ECGDenoiseDataset(Dataset):
    """
    __init__: 각 DS1 레코드에서 30초 윈도우(pad-then-crop)를 메모리에 적재.
              pad = 앞뒤 context_sec × fs 샘플의 실제 이웃 신호 (zero-padding 아님)
              → wavelet 경계 왜곡이 context 구간에 발생하고 중심 30초는 보호됨

    __getitem__:
      input  = wavelet_denoise(inject_noise(padded_window)) 의 중심 30초
      target = 원본 clean padded_window의 중심 30초
      → DAE는 "wavelet이 처리하고 남긴 잔차 → 원본 clean" 매핑을 학습
      → 에폭마다 다른 노이즈 조합이 생성되어 다양성 확보
    """

    def __init__(
        self,
        records: List[int],
        cfg: DAEConfig,
        train_noises: Dict[str, np.ndarray],
    ):
        self.cfg = cfg
        self.train_noises = train_noises
        self.padded_windows: List[np.ndarray] = []

        ctx = cfg.context_sec * cfg.fs    # 1,800 샘플 (5초)
        win = cfg.window_len              # 10,800 샘플 (30초)
        pad = cfg.padded_len              # 14,400 샘플 (40초)

        for rec_id in records:
            rec_path = os.path.join(cfg.mitdb_path, str(rec_id))
            record = wfdb.rdrecord(rec_path)
            ch_idx = (
                record.sig_name.index('MLII')
                if 'MLII' in record.sig_name
                else 0
            )
            signal = record.p_signal[:, ch_idx].astype(np.float32)

            # 비겹침 30초 단위 슬라이딩 (context는 실제 이웃 신호로 채움)
            n_windows = (len(signal) - 2 * ctx) // win
            for w in range(n_windows):
                center_start = ctx + w * win
                s = center_start - ctx
                e = center_start + win + ctx
                if s < 0 or e > len(signal):
                    continue
                self.padded_windows.append(signal[s:e].copy())

        self.ctx = ctx
        self.win = win
        print(
            f"  [ECGDenoiseDataset] {len(records)}명 → "
            f"{len(self.padded_windows)}개 윈도우 적재"
        )

    def __len__(self) -> int:
        return len(self.padded_windows)

    def __getitem__(self, idx: int):
        clean_padded = self.padded_windows[idx]

        # on-the-fly 노이즈 합성 → wavelet 1차 처리
        noisy_padded   = inject_noise(clean_padded, self.train_noises, self.cfg)
        denoised_padded = wavelet_denoise(
            noisy_padded, self.cfg.wavelet, self.cfg.wavelet_level
        )

        # context 제거: 중심 30초만 입출력으로 사용
        c = self.ctx
        x = denoised_padded[c : c + self.win]   # DAE input
        y = clean_padded[c : c + self.win]       # DAE target (원본 clean)

        return (
            torch.from_numpy(x.copy()).float().unsqueeze(0),  # (1, 10800)
            torch.from_numpy(y.copy()).float().unsqueeze(0),  # (1, 10800)
        )


def _worker_init_fn(worker_id: int) -> None:
    """DataLoader worker별 시드 재설정 → fork된 프로세스가 동일 노이즈 반복 방지."""
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# ============================================================
# 5. 모델: 1D U-Net DAE
# ============================================================
class ConvBlock1D(nn.Module):
    """Conv → BN → ReLU 두 번. kernel=9로 ECG 의미 있는 시간 범위 커버."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 9):
        super().__init__()
        p = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=p),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel, padding=p),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class Down1D(nn.Module):
    """MaxPool(2) → ConvBlock: 시간 해상도 절반, 채널 수 두 배."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv = ConvBlock1D(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))


class Up1D(nn.Module):
    """ConvTranspose(2) → skip concat → ConvBlock.
    skip connection: encoder 동일 레벨의 특징을 decoder에 직접 전달
    → QRS 같은 sharp transient가 bottleneck을 거치지 않고 보존됨
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock1D(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # 홀수 길이 처리 시 1샘플 불일치 방지
        diff = skip.shape[-1] - x.shape[-1]
        if diff > 0:
            x = F.pad(x, (0, diff))
        elif diff < 0:
            x = x[..., :skip.shape[-1]]
        return self.conv(torch.cat([x, skip], dim=1))


class ECG_DAE_UNet1D(nn.Module):
    """
    1D U-Net Denoising Autoencoder.

    depth=4 기준 채널 수:
      in_conv  : 1  → 16
      down[0]  : 16 → 32
      down[1]  : 32 → 64
      down[2]  : 64 → 128
      down[3]  : 128 → 256  (bottleneck)
      up[0]    : 256 + 128 → 128
      up[1]    : 128 + 64  → 64
      up[2]    : 64  + 32  → 32
      up[3]    : 32  + 16  → 16
      out_conv : 16 → 1

    입력 길이가 2^depth의 배수가 아니면 자동으로 padding 후 복원.
    """
    def __init__(self, base_channels: int = 16, depth: int = 4):
        super().__init__()
        self.depth    = depth
        self.divisor  = 2 ** depth

        chs = [base_channels * (2 ** i) for i in range(depth + 1)]
        self.in_conv  = ConvBlock1D(1, chs[0])
        self.downs    = nn.ModuleList(
            [Down1D(chs[i], chs[i + 1]) for i in range(depth)]
        )
        self.ups      = nn.ModuleList(
            [Up1D(chs[i + 1], chs[i], chs[i]) for i in reversed(range(depth))]
        )
        self.out_conv = nn.Conv1d(chs[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_len = x.shape[-1]

        # 길이를 divisor의 배수로 맞춤
        pad = (self.divisor - orig_len % self.divisor) % self.divisor
        if pad:
            x = F.pad(x, (0, pad))

        # Encoder: skip 저장
        skips = [self.in_conv(x)]
        h = skips[0]
        for down in self.downs:
            h = down(h)
            skips.append(h)

        # bottleneck = skips[-1] = h → skip 대상 아님, pop으로 제거
        skips.pop()

        # Decoder: skip concat
        for up in self.ups:
            h = up(h, skips.pop())

        out = self.out_conv(h)
        return out[..., :orig_len]   # padding 제거 후 원래 길이 복원


# ============================================================
# 6. 학습 루프
# ============================================================
def get_loss_fn(name: str) -> nn.Module:
    return {'l1': nn.L1Loss(), 'mse': nn.MSELoss(), 'huber': nn.SmoothL1Loss()}[name]


def train_dae(cfg: DAEConfig) -> str:
    """DAE 학습 전체 루프. 체크포인트 경로를 반환."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ── Phase 1 노이즈 뱅크 재사용 ──────────────────────────
    path_cfg   = PathConfig(mitdb_path=cfg.mitdb_path, nstdb_path=cfg.nstdb_path)
    noise_bank = load_noise_bank(cfg.nstdb_path, NOISE_TYPES, NOISE_TRAIN_FRACTION)
    validate_noise_bank(noise_bank)
    train_noises = noise_bank.train_signals
    # noise_bank.test_signals는 Phase 5 전용 → 여기서 절대 사용 안 함

    # ── DS1 내부 환자 단위 train/val split ──────────────────
    train_records, val_records = split_ds1_patients(cfg, cfg.seed)
    print(f"\n[Phase 2] DAE 학습 환자 ({len(train_records)}명): {train_records}")
    print(f"[Phase 2] DAE 검증 환자 ({len(val_records)}명): {val_records}")

    # ── Dataset / DataLoader ────────────────────────────────
    print("\n윈도우 적재 중...")
    train_ds = ECGDenoiseDataset(train_records, cfg, train_noises)
    val_ds   = ECGDenoiseDataset(val_records,   cfg, train_noises)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, worker_init_fn=_worker_init_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, worker_init_fn=_worker_init_fn,
    )

    # ── 모델 / 옵티마이저 ────────────────────────────────────
    model     = ECG_DAE_UNet1D(cfg.base_channels, cfg.depth).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
    )
    loss_fn   = get_loss_fn(cfg.loss_fn)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n모델 파라미터 수: {n_params:,}")
    print(f"학습 device: {cfg.device}")

    best_val_loss = float('inf')
    patience_ctr  = 0
    best_ckpt     = os.path.join(cfg.checkpoint_dir, 'dae_best.pt')

    # ── MLflow 실험 기록 ─────────────────────────────────────
    mlflow.set_experiment(cfg.mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_params({
            'fs':                  cfg.fs,
            'window_sec':          cfg.window_sec,
            'context_sec':         cfg.context_sec,
            'wavelet':             cfg.wavelet,
            'wavelet_level':       cfg.wavelet_level,
            'base_channels':       cfg.base_channels,
            'depth':               cfg.depth,
            'n_params':            n_params,
            'batch_size':          cfg.batch_size,
            'lr':                  cfg.lr,
            'loss_fn':             cfg.loss_fn,
            'p_bw':                cfg.p_include_noise['bw'],
            'p_ma':                cfg.p_include_noise['ma'],
            'p_em':                cfg.p_include_noise['em'],
            'p_clean_passthrough': cfg.p_clean_passthrough,
            'noise_train_fraction':NOISE_TRAIN_FRACTION,
            'n_train_windows':     len(train_ds),
            'n_val_windows':       len(val_ds),
            'train_patients':      str(train_records),
            'val_patients':        str(val_records),
        })

        print(f"\n{'='*60}")
        print(f" DAE 학습 시작  (최대 {cfg.epochs}에폭, "
              f"early stop patience={cfg.early_stop_patience})")
        print(f"{'='*60}")

        for epoch in range(1, cfg.epochs + 1):
            # Train
            model.train()
            train_losses = []
            for x, y in train_loader:
                x, y = x.to(cfg.device), y.to(cfg.device)
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_loss = float(np.mean(train_losses))

            # Val
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(cfg.device), y.to(cfg.device)
                    val_losses.append(loss_fn(model(x), y).item())
            val_loss = float(np.mean(val_losses))

            scheduler.step(val_loss)
            print(f"  현재 lr: {optimizer.param_groups[0]['lr']:.2e}")
            mlflow.log_metrics(
                {'train_loss': train_loss, 'val_loss': val_loss}, step=epoch
            )
            print(
                f"[Epoch {epoch:03d}/{cfg.epochs}] "
                f"train={train_loss:.5f}  val={val_loss:.5f}"
            )

            # Best 체크포인트 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_ctr  = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config':           asdict(cfg),
                    'epoch':            epoch,
                    'val_loss':         val_loss,
                }, best_ckpt)
                print(f"  → 베스트 모델 저장 (val_loss={val_loss:.5f})")
            else:
                patience_ctr += 1
                if patience_ctr >= cfg.early_stop_patience:
                    print(
                        f"\nEarly stopping: {cfg.early_stop_patience}에폭 동안 "
                        f"val_loss 개선 없음 (epoch {epoch})"
                    )
                    break

        mlflow.log_metric('best_val_loss', best_val_loss)
        mlflow.log_artifact(best_ckpt)

    print(f"\n[Phase 2 완료] best_val_loss={best_val_loss:.5f}")
    print(f"체크포인트: {best_ckpt}")
    return best_ckpt


# ============================================================
# 7. Phase 3 / Phase 6에서 호출하는 frozen DAE 로더
# ============================================================
def load_frozen_dae(
    checkpoint_path: str,
    device: Optional[str] = None,
) -> nn.Module:
    """
    학습된 DAE를 로드하고 파라미터를 완전히 고정(freeze).
    Phase 3 분류기 Dataset과 Phase 6 배포 API에서 동일하게 사용.

    호출 원칙:
      - 노이즈 종류와 무관하게 항상 호출 (배포 시 노이즈 종류 불명)
      - wavelet_denoise() 이후에만 호출 (wavelet → DAE 순서 고정)
      - model.eval() + torch.no_grad() 상태 유지
    """
    ckpt   = torch.load(checkpoint_path, map_location='cpu')
    cfg_d  = ckpt['config']
    model  = ECG_DAE_UNet1D(
        base_channels=cfg_d['base_channels'],
        depth=cfg_d['depth'],
    )
    model.load_state_dict(ckpt['model_state_dict'])

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(
        f"[DAE] 체크포인트 로드 완료 "
        f"(epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f}, "
        f"device={device})"
    )
    return model


# ============================================================
# 8. 진입점
# ============================================================
if __name__ == '__main__':
    cfg = DAEConfig()
    train_dae(cfg)