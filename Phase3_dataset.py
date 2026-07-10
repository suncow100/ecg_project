"""
Phase 3 · 분류기 학습용 Beat 데이터셋 구성
============================================
Phase 1, 2가 완료된 후 실행합니다.

전체 파이프라인 (윈도우 단위):
  raw 30s window (DS1)
    → 노이즈 합성 (BW/MA/EM 독립 랜덤, NSTDB train 70%만)
    → Wavelet 디노이징 (db6/L8, BW+MA 처리)
    → DAE frozen inference (EM 잔차 처리)
    → Z-score 정규화 (윈도우 단위 — beat 간 상대 진폭 보존)
      └ 통계량(mean/std)을 beat 메타데이터로 저장 → Phase 5 Track A 재사용
    → annotation 기반 R-peak 위치 확인
    → 비대칭 beat segmentation (앞 100 / 뒤 150 = 250 샘플)
    → beat pool 적재 (AAMI 라벨 + 원본 symbol 보존)

__getitem__ (on-the-fly):
    → Custom Mixup 적용
    → (1, 250) 텐서 반환

설계 원칙:
  - annotation 기반 segmentation: 학습 전용
    (NeuroKit2는 Phase 5 Track B / Phase 6 배포 전용)
  - Z-score를 beat 단위로 하지 않는 이유:
    beat별 독립 정규화 시 beat 간 상대 진폭 차이 소멸
    → V-class의 핵심 특징(정상 대비 비정상적으로 큰 진폭)이 사라짐
  - Mixup을 __getitem__에서 on-the-fly로 하는 이유:
    에폭마다 다른 쌍이 선택되어 다양성 극대화
"""

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import wfdb
from collections import Counter, defaultdict
from torch.utils.data import DataLoader, Dataset

# ── Phase 1 import ──────────────────────────────────────────
from Phase1_data_preprocessing import (
    AAMI_MAPPING,
    AAMI_CLASS_NAMES,
    DS1_RECORDS,
    DS2_RECORDS,
    NOISE_TYPES,
    NOISE_TRAIN_FRACTION,
    S_MIXUP_GROUPS,
    VALID_SYMBOLS,
    load_noise_bank,
    validate_noise_bank,
)

# ── Phase 2 import ──────────────────────────────────────────
from Phase2_DAE_pretrain import (
    DAEConfig,
    inject_noise,
    load_frozen_dae,
    wavelet_denoise,
)


# ============================================================
# 0. Config
# ============================================================
@dataclass
class Phase3Config:
    # ── 경로 ──────────────────────────────────────────────
    mitdb_path: str = '/home/qortjsdn/projects/ecg_project/mit-bih-arrhythmia-database-1.0.0/'
    nstdb_path: str = '/home/qortjsdn/projects/ecg_project/nstdb_noise/'
    dae_checkpoint: str = './checkpoints/dae_best.pt'

    # ── 신호 파라미터 ──────────────────────────────────────
    fs: int = 360
    window_sec: int = 30
    context_sec: int = 5

    # ── Beat 세그멘테이션 ──────────────────────────────────
    beat_before: int = 100   # R-peak 기준 앞 샘플 수
    beat_after: int  = 150   # R-peak 기준 뒤 샘플 수
    # 총 250 샘플 = beat_before + beat_after
    # 비대칭 이유: T파(R 이후)가 P파(R 이전)보다 생리학적으로 더 길게 펼쳐짐

    # ── 웨이블릿 (Phase 2와 동일하게 고정) ────────────────
    wavelet: str = 'db6'
    wavelet_level: int = 8

    # ── 노이즈 합성 ────────────────────────────────────────
    snr_levels_db: List[float] = field(
        default_factory=lambda: [0, 6, 12, 18, 24]
    )
    p_include_noise: Dict[str, float] = field(
        default_factory=lambda: {'bw': 0.5, 'ma': 0.5, 'em': 0.5}
    )
    p_clean_passthrough: float = 0.15

    # ── Mixup ──────────────────────────────────────────────
    mixup_alpha: float = 0.2
    # Beta(0.2, 0.2): lambda가 0 또는 1 쪽으로 쏠림
    # → 한쪽이 주(主)가 되고 다른 쪽이 약하게 섞임
    # → QRS 엣지 뭉개짐 최소화

    # ── 학습 / DataLoader ──────────────────────────────────
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 42
    device: str = field(
        default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu'
    )

    @property
    def beat_len(self) -> int:
        return self.beat_before + self.beat_after   # 250

    @property
    def window_len(self) -> int:
        return self.window_sec * self.fs            # 10,800

    @property
    def padded_len(self) -> int:
        return (self.window_sec + 2 * self.context_sec) * self.fs  # 14,400


# ============================================================
# 1. Beat 메타데이터 (Track A 정규화 통계량 보존용)
# ============================================================
@dataclass
class BeatMeta:
    """
    beat 하나에 붙는 메타데이터.
    win_mean / win_std: 이 beat가 추출된 30초 윈도우의 Z-score 통계량.
    Phase 5 Track A에서 "윈도우 없이 beat만 있는 상황"에도
    동일한 정규화 기준을 재현하기 위해 저장.
    """
    record_id: int
    peak_sample: int         # 원본 신호에서의 R-peak 위치
    aami_label: int          # AAMI 5-class 인덱스
    orig_symbol: str         # 원본 wfdb 심볼 (Mixup 규칙 적용용)
    win_mean: float          # 이 beat가 속한 30초 윈도우의 평균
    win_std: float           # 이 beat가 속한 30초 윈도우의 표준편차


# ============================================================
# 2. 전처리 파이프라인 (윈도우 단위)
# ============================================================
def preprocess_window(
    window: np.ndarray,
    train_noises: Dict[str, np.ndarray],
    dae_model: nn.Module,
    cfg: Phase3Config,
    apply_noise: bool = True,
) -> Tuple[np.ndarray, float, float]:
    """
    단일 30초 윈도우에 전처리 전체 적용.

    반환:
      normalized  : Z-score 정규화된 윈도우 (beat segmentation에 사용)
      win_mean    : 정규화에 쓰인 평균 (BeatMeta 저장용)
      win_std     : 정규화에 쓰인 표준편차 (BeatMeta 저장용)

    순서:
      노이즈 합성 → Wavelet 디노이징 → DAE → Z-score 정규화
      (이 순서는 Phase 6 배포 파이프라인과 동일하게 고정)
    """
    # 1) 노이즈 합성 (BW/MA/EM 독립 랜덤)
    noisy = inject_noise(window, train_noises, cfg) if apply_noise else window.copy()

    # 2) Wavelet 디노이징 (BW approximation 제거 + MA detail threshold)
    wt_denoised = wavelet_denoise(noisy, cfg.wavelet, cfg.wavelet_level)

    # 3) DAE frozen inference (EM 잔차 처리)
    #    노이즈 종류 무관 항상 호출 — 조건 분기 없음
    with torch.no_grad():
        x = torch.from_numpy(wt_denoised).float().unsqueeze(0).unsqueeze(0)
        x = x.to(next(dae_model.parameters()).device)
        dae_out = dae_model(x).squeeze().cpu().numpy()

    # 4) Z-score 정규화 (윈도우 단위)
    #    beat 단위로 각각 정규화하면 beat 간 상대 진폭 차이가 소멸됨
    #    → V-class 핵심 특징(비정상적으로 큰 QRS 진폭)이 사라짐
    #    → 윈도우 전체의 통계량으로 정규화해서 상대 진폭 보존
    win_mean = float(np.mean(dae_out))
    win_std  = float(np.std(dae_out) + 1e-8)
    normalized = (dae_out - win_mean) / win_std

    return normalized.astype(np.float32), win_mean, win_std


# ============================================================
# 3. 단일 레코드 → beat pool 구축
# ============================================================
def extract_beats_from_record(
    record_id: int,
    cfg: Phase3Config,
    train_noises: Dict[str, np.ndarray],
    dae_model: nn.Module,
    apply_noise: bool = True,
) -> Tuple[List[np.ndarray], List[BeatMeta]]:
    """
    단일 MIT-BIH 레코드를 30초 윈도우 단위로 순회하며 beat를 추출.

    윈도우별 처리:
      1. 윈도우를 실제 이웃 신호로 pad (pad-then-crop, zero-pad 아님)
      2. 전처리 파이프라인 적용 (noise → wavelet → DAE → Z-score)
      3. 이 윈도우 안에 속하는 annotation peak를 찾아 beat 세그멘테이션
         - annotation이 윈도우 범위 밖의 beat는 자연스럽게 건너뜀
         - context 구간(앞뒤 5초)에 걸친 beat도 제외
           (context 제거 후 잘리는 beat가 생기기 때문)

    반환:
      beats : (250,) float32 배열의 리스트
      metas : BeatMeta 리스트 (Track A 정규화 통계량 포함)
    """
    rec_path = os.path.join(cfg.mitdb_path, str(record_id))
    record   = wfdb.rdrecord(rec_path)
    ann      = wfdb.rdann(rec_path, 'atr')

    ch_idx = (
        record.sig_name.index('MLII')
        if 'MLII' in record.sig_name
        else 0
    )
    signal = record.p_signal[:, ch_idx].astype(np.float32)

    ctx = cfg.context_sec * cfg.fs   # 1,800
    win = cfg.window_len             # 10,800

    # annotation을 {peak_sample: symbol} 딕셔너리로 변환
    ann_dict: Dict[int, str] = {
        s: sym for s, sym in zip(ann.sample, ann.symbol)
        if sym in VALID_SYMBOLS
    }

    beats: List[np.ndarray] = []
    metas: List[BeatMeta]   = []

    n_windows = (len(signal) - 2 * ctx) // win
    for w in range(n_windows):
        center_start = ctx + w * win
        pad_start    = center_start - ctx
        pad_end      = center_start + win + ctx

        if pad_start < 0 or pad_end > len(signal):
            continue

        # context 포함 패딩 윈도우 (pad-then-crop)
        padded_window = signal[pad_start:pad_end].copy()

        # 전처리: noise → wavelet → DAE → Z-score
        # 반환되는 normalized는 중심 30초 길이 (context 제거됨)
        normalized, win_mean, win_std = preprocess_window(
            padded_window[ctx: ctx + win],  # 중심 30초만 전처리에 넘김
            train_noises, dae_model, cfg,apply_noise=apply_noise,
        )

        # 이 윈도우 안에서 beat 세그멘테이션
        win_start_abs = center_start   # 원본 신호 기준 윈도우 시작
        win_end_abs   = center_start + win

        for peak_abs, sym in ann_dict.items():
            # 윈도우 범위 안에 있는 peak만 처리
            if peak_abs < win_start_abs or peak_abs >= win_end_abs:
                continue

            # 원본 신호 기준 → normalized 배열 기준으로 좌표 변환
            peak_local = peak_abs - win_start_abs

            seg_start = peak_local - cfg.beat_before
            seg_end   = peak_local + cfg.beat_after

            # 윈도우 경계 안전 확인
            if seg_start < 0 or seg_end > win:
                continue

            beat = normalized[seg_start:seg_end].copy()   # (250,)

            beats.append(beat)
            metas.append(BeatMeta(
                record_id    = record_id,
                peak_sample  = peak_abs,
                aami_label   = AAMI_MAPPING[sym],
                orig_symbol  = sym,
                win_mean     = win_mean,
                win_std      = win_std,
            ))

    return beats, metas


# ============================================================
# 4. Mixup 로직
# ============================================================
def _build_mixup_pools(
    beats: List[np.ndarray],
    metas: List[BeatMeta],
) -> Dict[str, Dict[int, List[int]]]:
    """
    Mixup을 위한 beat index pool 구성.

    구조:
      pools['V'][patient_id]   = [beat_idx, ...]
      pools['F'][patient_id]   = [beat_idx, ...]
      pools['A_group'][pid]    = [beat_idx, ...]   (A, a 심볼)
      pools['J_group'][pid]    = [beat_idx, ...]   (J 심볼)

    patient_id를 key로 쓰는 이유:
      동일 환자 내 Mixup 금지 → 환자 고유 ECG 패턴을 외우는 효과 방지
      다른 환자의 beat를 섞어야 진정한 형태 다양성 확보
    """
    pools: Dict[str, Dict[int, List[int]]] = {
        'V':       defaultdict(list),
        'F':       defaultdict(list),
        'A_group': defaultdict(list),
        'J_group': defaultdict(list),
    }

    for idx, meta in enumerate(metas):
        pid = meta.record_id
        sym = meta.orig_symbol

        if meta.aami_label == 2:          # V class
            pools['V'][pid].append(idx)
        elif meta.aami_label == 3:        # F class
            pools['F'][pid].append(idx)
        elif sym in S_MIXUP_GROUPS['A_group']:   # A, a
            pools['A_group'][pid].append(idx)
        elif sym in S_MIXUP_GROUPS['J_group']:   # J
            pools['J_group'][pid].append(idx)
        # N(0), Q(4): Mixup 제외

    return pools


def _sample_mixup_partner(
    pool_for_class: Dict[int, List[int]],
    exclude_patient: int,
) -> Optional[int]:
    """
    동일 환자를 제외한 pool에서 랜덤 beat index를 선택.
    조건을 만족하는 환자가 없으면 None 반환 (Mixup 건너뜀).
    """
    candidates = {
        pid: idxs
        for pid, idxs in pool_for_class.items()
        if pid != exclude_patient and idxs
    }
    if not candidates:
        return None
    partner_pid  = random.choice(list(candidates.keys()))
    return random.choice(candidates[partner_pid])


def apply_mixup(
    beat: np.ndarray,
    meta: BeatMeta,
    beats: List[np.ndarray],
    pools: Dict[str, Dict[int, List[int]]],
    alpha: float = 0.2,
) -> np.ndarray:
    """
    클래스별 Mixup 규칙 적용.

    V: V끼리, 다른 환자
    F: F끼리, 다른 환자
    S: 원본 symbol 단위 (A_group ↔ A_group, J_group ↔ J_group)
       A와 J는 기원 부위(심방 vs 접합부)가 달라 교차 금지
    Q: 제외 (형태 이질성 과다, 공통 매니폴드 없음)
    N: 제외 (다수 클래스, 추가 증강 불필요)

    lambda ~ Beta(alpha, alpha):
      alpha=0.2이면 lambda가 0 또는 1 근방에 집중
      → 한쪽이 주(主)가 되고 다른 쪽이 약하게 섞임
      → QRS 엣지 뭉개짐(고주파 상쇄) 최소화
    """
    sym   = meta.orig_symbol
    label = meta.aami_label
    pid   = meta.record_id

    # Mixup 대상 pool 선택
    if label == 2:                              # V
        pool = pools['V']
    elif label == 3:                            # F
        pool = pools['F']
    elif sym in S_MIXUP_GROUPS['A_group']:      # S/A계열
        pool = pools['A_group']
    elif sym in S_MIXUP_GROUPS['J_group']:      # S/J계열
        pool = pools['J_group']
    else:                                       # N, Q → 그대로 반환
        return beat

    partner_idx = _sample_mixup_partner(pool, exclude_patient=pid)
    if partner_idx is None:                     # 조건 만족 파트너 없음
        return beat

    lam = np.random.beta(alpha, alpha)
    return (lam * beat + (1 - lam) * beats[partner_idx]).astype(np.float32)


# ============================================================
# 5. Dataset
# ============================================================
class ECGBeatDataset(Dataset):
    """
    __init__:
      - DS1 또는 DS2 레코드 전체를 순회
      - 레코드별로 30초 윈도우 단위 전처리 후 beat 추출
      - beat 배열과 BeatMeta를 메모리에 적재
      - Mixup pool을 미리 구성

    __getitem__:
      - on-the-fly Mixup 적용 (에폭마다 다른 쌍 → 다양성 극대화)
      - (1, 250) float32 텐서와 라벨 텐서 반환

    apply_augmentation=False로 설정하면 Mixup 없이 원본 beat 반환
    → Phase 5 Track A/B 평가용 DS2 Dataset에서 사용
    """

    def __init__(
        self,
        record_list: List[int],
        cfg: Phase3Config,
        dae_model: nn.Module,
        train_noises: Dict[str, np.ndarray],
        apply_augmentation: bool = True,
        split_name: str = 'Train',
    ):
        self.cfg               = cfg
        self.apply_augmentation = apply_augmentation

        self.beats: List[np.ndarray] = []
        self.metas: List[BeatMeta]   = []

        print(f"\n[Phase 3 Dataset — {split_name}] beat 추출 중...")

        for rec_id in record_list:
            b, m = extract_beats_from_record(
                rec_id, cfg, train_noises, dae_model,
                apply_noise=apply_augmentation,
            )
            self.beats.extend(b)
            self.metas.extend(m)
            print(f"  레코드 {rec_id:>4}: {len(b):>5}개 beat 추출")

        # Mixup pool 구성 (train split에서만 의미 있음)
        self.pools = _build_mixup_pools(self.beats, self.metas)

        # 클래스 분포 출력
        label_counter = Counter(m.aami_label for m in self.metas)
        print(f"\n[{split_name}] AAMI class 분포 (총 {len(self.beats)}개):")
        for cls in range(5):
            cnt = label_counter.get(cls, 0)
            bar = '█' * (cnt * 30 // max(label_counter.values()))
            print(f"  Class {cls} {AAMI_CLASS_NAMES[cls]:<28}: {cnt:>6}  {bar}")

        # Track A 재사용을 위한 윈도우 통계량 저장 확인
        print(f"\n  win_mean/std 저장 완료 → Phase 5 Track A에서 재사용 가능")

    def __len__(self) -> int:
        return len(self.beats)

    def __getitem__(self, idx: int):
        beat  = self.beats[idx].copy()
        meta  = self.metas[idx]
        label = meta.aami_label

        # on-the-fly Mixup (train split만)
        if self.apply_augmentation:
            beat = apply_mixup(
                beat, meta, self.beats, self.pools, self.cfg.mixup_alpha
            )

        x = torch.from_numpy(beat).float().unsqueeze(0)  # (1, 250)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

    def get_class_weights(self) -> torch.Tensor:
        """
        Weighted Cross-Entropy Loss용 클래스 가중치 계산.
        weight[i] = total / (n_classes * count[i])

        Phase 4 학습 루프에서 이 메서드로 가중치를 받아
        nn.CrossEntropyLoss(weight=...) 에 전달.
        """
        label_counter = Counter(m.aami_label for m in self.metas)
        total         = len(self.metas)
        n_classes     = 5

        weights = []
        print("\n[Weighted Cross-Entropy] 클래스 가중치:")
        for cls in range(n_classes):
            cnt = label_counter.get(cls, 1)   # 0 나누기 방지
            w   = total / (n_classes * cnt)
            weights.append(w)
            print(f"  Class {cls} {AAMI_CLASS_NAMES[cls]:<28}: "
                  f"count={cnt:>6}  weight={w:.4f}")

        # Q(class 4) 가중치 폭발 경고
        if weights[4] > 100:
            print(f"\n  ⚠ Class 4 (Q) 가중치={weights[4]:.1f} — 학습 불안정 위험")
            print(f"    Phase 4에서 cap_value로 클리핑 권장 (예: min(w, 50))")

        return torch.FloatTensor(weights)

    def get_beat_meta(self, idx: int) -> BeatMeta:
        """
        Phase 5 Track A에서 윈도우 통계량(win_mean, win_std)을 재사용할 때 호출.
        beat를 denoise 없이 평가하더라도 동일한 Z-score 기준 적용 가능.
        """
        return self.metas[idx]


# ============================================================
# 6. DataLoader 생성 헬퍼
# ============================================================
def build_dataloaders(
    cfg: Phase3Config,
    dae_checkpoint: str,
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Phase 4 학습 루프에 바로 넘길 수 있는 DataLoader 쌍과
    Weighted CE Loss 가중치를 반환.

    반환:
      train_loader : DS1 기반, Mixup on
      test_loader  : DS2 기반, Mixup off (순수 평가용)
      class_weights: Phase 4 nn.CrossEntropyLoss(weight=...) 전달용
    """
    # Phase 1 노이즈 뱅크 (train split만)
    noise_bank   = load_noise_bank(cfg.nstdb_path, NOISE_TYPES, NOISE_TRAIN_FRACTION)
    validate_noise_bank(noise_bank)
    train_noises = noise_bank.train_signals

    # Phase 2 frozen DAE
    dae_model = load_frozen_dae(dae_checkpoint, device=cfg.device)

    # DS1 Train Dataset
    train_ds = ECGBeatDataset(
        record_list=DS1_RECORDS,
        cfg=cfg,
        dae_model=dae_model,
        train_noises=train_noises,
        apply_augmentation=True,
        split_name='DS1 Train',
    )

    # DS2 Test Dataset (Mixup 없음, noise도 없음 — clean 평가 기준)
    # Phase 5 Track A/B에서 별도 노이즈를 합성하므로 여기선 clean으로만
    test_ds = ECGBeatDataset(
        record_list=DS2_RECORDS,
        cfg=cfg,
        dae_model=dae_model,
        train_noises=train_noises,
        apply_augmentation=False,
        split_name='DS2 Test (clean baseline)',
    )

    class_weights = train_ds.get_class_weights()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    return train_loader, test_loader, class_weights


# ============================================================
# 7. 진입점 (단독 실행 시 데이터 검증)
# ============================================================
if __name__ == '__main__':
    cfg = Phase3Config()

    print("Phase 3 데이터셋 구성 검증 시작...")
    train_loader, test_loader, class_weights = build_dataloaders(
        cfg, cfg.dae_checkpoint
    )

    # 첫 배치 shape 확인
    x_batch, y_batch = next(iter(train_loader))
    print(f"\n[배치 확인]")
    print(f"  X shape : {x_batch.shape}")   # (64, 1, 250) 기대
    print(f"  Y shape : {y_batch.shape}")   # (64,)       기대
    print(f"  X dtype : {x_batch.dtype}")
    print(f"  Y 분포  : {Counter(y_batch.tolist())}")

    print(f"\n[Weighted CE 가중치]")
    print(f"  {class_weights}")
    print(f"\n[Phase 3 완료] → Phase 4 1D-CNN 학습으로 진행")