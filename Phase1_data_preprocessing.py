"""
Phase 1 · 데이터 베이스캠프
============================
담당하는 것:
  1. MIT-BIH Arrhythmia DB 로드 및 DS1/DS2 inter-patient split (de Chazal 표준)
  2. Paced beat 레코드 제외 (102, 104, 107, 217)
  3. AAMI 5-class 심볼 매핑 테이블 정의
  4. NSTDB 노이즈 레코드 로드 및 disjoint 70/30 train/test split
  5. 무결성 검증 (환자 overlap / 노이즈 overlap 방지)

담당하지 않는 것:
  - 신호 전처리 (wavelet, DAE) → Phase 2~3
  - beat segmentation → Phase 3
  - 모델 학습 → Phase 4
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import wfdb
from collections import Counter


# ============================================================
# 0. 경로 설정
# ============================================================
@dataclass
class PathConfig:
    mitdb_path: str = '/home/qortjsdn/projects/ecg_project/mit-bih-arrhythmia-database-1.0.0/'
    nstdb_path: str = '/home/qortjsdn/projects/ecg_project/nstdb_noise/'


# ============================================================
# 1. MIT-BIH 환자 분리 (de Chazal DS1/DS2)
# ============================================================

# Paced beat 레코드: 인공심박동기(pacemaker) 신호가 포함되어
# 심장 고유 리듬 학습을 오염시키므로 전체 파이프라인에서 제외
PACED_RECORDS = {102, 104, 107, 217}

# de Chazal (2004) inter-patient split 표준
# DS1: 학습용 / DS2: 테스트용 — 두 집합 간 환자가 절대 겹치지 않음
DS1_RECORDS: List[int] = [
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
    122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230
]
DS2_RECORDS: List[int] = [
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
    210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234
]

def validate_patient_split() -> None:
    """DS1/DS2 환자 overlap 및 paced 레코드 혼입 검증."""
    ds1_set = set(DS1_RECORDS)
    ds2_set = set(DS2_RECORDS)

    overlap = ds1_set & ds2_set
    assert not overlap, f"DS1/DS2 환자 overlap 발견: {overlap}"

    paced_in_ds1 = ds1_set & PACED_RECORDS
    paced_in_ds2 = ds2_set & PACED_RECORDS
    assert not paced_in_ds1, f"DS1에 paced 레코드 혼입: {paced_in_ds1}"
    assert not paced_in_ds2, f"DS2에 paced 레코드 혼입: {paced_in_ds2}"

    print(f"[OK] 환자 분리 검증 완료")
    print(f"     DS1: {len(DS1_RECORDS)}명  DS2: {len(DS2_RECORDS)}명  "
          f"Paced 제외: {sorted(PACED_RECORDS)}")


# ============================================================
# 2. AAMI 5-class 심볼 매핑
# ============================================================

# wfdb .atr에서 읽히는 원본 심볼 → AAMI EC57 표준 5-class 인덱스 매핑
# 제외 심볼('|', '~', '+', 's', 'T', 'u', '`', '!' 등)은 VALID_SYMBOLS에 없으므로
# 이후 segmentation 단계에서 자동으로 필터링됨
VALID_SYMBOLS = {
    'N', 'L', 'R', 'e', 'j',   # N (Normal)
    'A', 'a', 'J', 'S',         # S (Supraventricular)
    'V', 'E',                   # V (Ventricular)
    'F',                        # F (Fusion)
    '/', 'f', 'Q',              # Q (Unknown/Unclassifiable)
}

AAMI_MAPPING: Dict[str, int] = {
    # Class 0 · N (Normal & bundle branch blocks)
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    # Class 1 · S (Supraventricular ectopic)
    # Phase 3 Mixup에서 원본 심볼 단위(A↔A, J↔J)로 교차하므로 여기서도 보존
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    # Class 2 · V (Ventricular ectopic)
    'V': 2, 'E': 2,
    # Class 3 · F (Fusion)
    'F': 3,
    # Class 4 · Q (Unknown — Mixup 제외 대상)
    '/': 4, 'f': 4, 'Q': 4,
}

AAMI_CLASS_NAMES: Dict[int, str] = {
    0: 'N (Normal)',
    1: 'S (Supraventricular)',
    2: 'V (Ventricular)',
    3: 'F (Fusion)',
    4: 'Q (Unknown)',
}

# Phase 3 Mixup 규칙을 위한 S-class 원본 심볼 그룹
# A ↔ A, J ↔ J 교차만 허용. a, S는 샘플 수가 적어 단독 처리
S_MIXUP_GROUPS: Dict[str, List[str]] = {
    'A_group': ['A', 'a'],   # 심방 조기박동 계열 (기원 동일)
    'J_group': ['J'],        # 접합부 조기박동 (기원 다름 → A와 교차 금지)
}


# ============================================================
# 3. MIT-BIH 레코드 메타데이터 로드
# ============================================================
@dataclass
class RecordMeta:
    """단일 MIT-BIH 레코드의 로드 결과."""
    record_id: int
    fs: int
    n_samples: int
    duration_min: float
    ch_used: str             # 사용한 채널명 (MLII 우선)
    symbol_counts: Dict[str, int]   # 원본 심볼별 beat 수
    aami_counts: Dict[int, int]     # AAMI class별 beat 수
    n_valid: int             # VALID_SYMBOLS에 해당하는 beat 수


def load_record_meta(record_id: int, mitdb_path: str) -> RecordMeta:
    """
    레코드 하나를 로드해서 메타데이터만 추출.
    실제 신호 배열은 Phase 3 Dataset에서 필요할 때 로드.
    MLII 채널 우선 탐색 → 없으면 채널 0 사용.
    """
    rec_path = os.path.join(mitdb_path, str(record_id))
    record = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, 'atr')

    ch_idx = record.sig_name.index('MLII') if 'MLII' in record.sig_name else 0
    ch_name = record.sig_name[ch_idx]

    symbol_counts: Dict[str, int] = Counter(ann.symbol)

    valid_symbols = [s for s in ann.symbol if s in VALID_SYMBOLS]
    aami_counts: Dict[int, int] = Counter(AAMI_MAPPING[s] for s in valid_symbols)

    duration_min = record.sig_len / record.fs / 60

    return RecordMeta(
        record_id=record_id,
        fs=record.fs,
        n_samples=record.sig_len,
        duration_min=round(duration_min, 2),
        ch_used=ch_name,
        symbol_counts=dict(symbol_counts),
        aami_counts=dict(aami_counts),
        n_valid=len(valid_symbols),
    )


def load_all_records_meta(
    record_ids: List[int],
    mitdb_path: str,
    split_name: str,
) -> List[RecordMeta]:
    """레코드 목록 전체 메타데이터 로드 및 요약 출력."""
    metas = []
    print(f"\n{'='*60}")
    print(f"[MIT-BIH] {split_name} 레코드 메타데이터 로드 ({len(record_ids)}명)")
    print(f"{'='*60}")

    total_aami: Dict[int, int] = Counter()
    for rid in record_ids:
        meta = load_record_meta(rid, mitdb_path)
        metas.append(meta)
        total_aami.update(meta.aami_counts)
        print(f"  {rid:>4}  {meta.ch_used:<6}  {meta.duration_min:>6.1f}min  "
              f"valid beats: {meta.n_valid:>5}")

    print(f"\n  [{split_name}] AAMI class 분포:")
    total_valid = sum(total_aami.values())
    for cls in range(5):
        cnt = total_aami.get(cls, 0)
        bar = '█' * (cnt * 30 // max(total_aami.values()) if total_aami else 0)
        print(f"    Class {cls} {AAMI_CLASS_NAMES[cls]:<28}: {cnt:>6}  {bar}")
    print(f"    Total valid beats: {total_valid}")

    return metas


# ============================================================
# 4. NSTDB 노이즈 로드 및 disjoint 70/30 split
# ============================================================

# NSTDB 노이즈 종류별 주파수 특성 (전처리 전략 참고용 주석)
# BW  (baseline wander)  : < 0.5~1Hz  저주파 → Wavelet approximation 제거
# MA  (muscle artifact)  : 20~500Hz   광대역 → Wavelet detail threshold 일부 처리
# EM  (electrode motion) : 전 대역 비정상 → DAE가 담당 (주파수로 분리 불가)
NOISE_TYPES: List[str] = ['bw', 'ma', 'em']
NOISE_TRAIN_FRACTION: float = 0.70   # 앞 70% → Phase 2(DAE) + Phase 3(분류기) 학습
                                      # 뒤 30% → Phase 5 Track A/B 평가 전용 (이 파일에서 분리만 함)


@dataclass
class NoiseSplit:
    """단일 노이즈 레코드의 disjoint train/test 분리 결과."""
    noise_type: str
    total_samples: int
    fs: int
    train_end_idx: int       # signal[:train_end_idx] → train
    test_start_idx: int      # signal[test_start_idx:] → test (= train_end_idx)
    train_duration_sec: float
    test_duration_sec: float


@dataclass
class NoiseBank:
    """
    모든 NSTDB 노이즈 레코드를 로드하고 split 정보를 보관.

    실제 numpy 배열은 속성으로 함께 저장.
    - train_signals: Phase 2(DAE 학습) + Phase 3(분류기 학습) 전용
    - test_signals:  Phase 5 Track A/B 평가 전용
                     이 파일 이외의 코드에서 train/test를 절대 혼용하지 말 것.
    """
    splits: Dict[str, NoiseSplit]
    train_signals: Dict[str, np.ndarray]
    test_signals: Dict[str, np.ndarray]


def load_noise_bank(nstdb_path: str, noise_types: List[str],
                    train_fraction: float = NOISE_TRAIN_FRACTION) -> NoiseBank:
    """
    NSTDB 노이즈 레코드 로드 → disjoint split.

    핵심 원칙:
      - split은 시간축 앞/뒤 기준 (무작위 셔플 금지).
        ECG 노이즈는 시간적 자기상관(self-correlation)이 있어서
        랜덤 split이면 train/test가 같은 노이즈 패턴을 공유하게 됨.
      - train_signals만 Phase 2~3에서 사용.
        test_signals는 Phase 5 코드에서 import해서 사용.
    """
    splits: Dict[str, NoiseSplit] = {}
    train_signals: Dict[str, np.ndarray] = {}
    test_signals: Dict[str, np.ndarray] = {}

    print(f"\n{'='*60}")
    print(f"[NSTDB] 노이즈 로드 및 {int(train_fraction*100)}/{int((1-train_fraction)*100)} disjoint split")
    print(f"{'='*60}")

    for ntype in noise_types:
        rec_path = os.path.join(nstdb_path, ntype)
        record = wfdb.rdrecord(rec_path)
        signal = record.p_signal[:, 0].astype(np.float32)
        fs = record.fs
        n = len(signal)

        split_idx = int(n * train_fraction)
        train_sig = signal[:split_idx]
        test_sig = signal[split_idx:]

        split = NoiseSplit(
            noise_type=ntype,
            total_samples=n,
            fs=fs,
            train_end_idx=split_idx,
            test_start_idx=split_idx,
            train_duration_sec=round(split_idx / fs, 1),
            test_duration_sec=round((n - split_idx) / fs, 1),
        )
        splits[ntype] = split
        train_signals[ntype] = train_sig
        test_signals[ntype] = test_sig

        print(f"  {ntype.upper():<4}  총 {n:>8}샘플 ({n/fs/60:.1f}min)  "
              f"train: {split.train_duration_sec:.0f}s  "
              f"test:  {split.test_duration_sec:.0f}s")

    return NoiseBank(splits=splits, train_signals=train_signals, test_signals=test_signals)


def validate_noise_bank(bank: NoiseBank) -> None:
    """train/test overlap이 없는지, 각 split이 충분한 길이인지 검증."""
    # 30초 윈도우(10,800샘플) + context(각 5초 × 2 = 3,600샘플) = 14,400샘플 최소 필요
    MIN_SAMPLES_NEEDED = 14_400

    for ntype, split in bank.splits.items():
        train_sig = bank.train_signals[ntype]
        test_sig  = bank.test_signals[ntype]

        # overlap 검증: split_idx 기준으로 완전히 분리되어 있음을 확인
        assert split.train_end_idx == split.test_start_idx, \
            f"{ntype}: train/test 경계 불일치"

        assert len(train_sig) >= MIN_SAMPLES_NEEDED, \
            f"{ntype} train split이 너무 짧음: {len(train_sig)}샘플 < {MIN_SAMPLES_NEEDED}"
        assert len(test_sig) >= MIN_SAMPLES_NEEDED, \
            f"{ntype} test split이 너무 짧음: {len(test_sig)}샘플 < {MIN_SAMPLES_NEEDED}"

    print(f"\n[OK] 노이즈 split 검증 완료 (train/test disjoint, 최소 길이 충족)")


# ============================================================
# 5. Phase 1 전체 실행 진입점
# ============================================================
def run_phase1(cfg: PathConfig) -> Tuple[List[RecordMeta], List[RecordMeta], NoiseBank]:
    """
    Phase 1 전체 실행.

    반환값:
      ds1_metas  : DS1 레코드 메타데이터 리스트 (Phase 2~3에서 학습 대상 확인용)
      ds2_metas  : DS2 레코드 메타데이터 리스트 (Phase 5 평가 대상 확인용)
      noise_bank : NSTDB 노이즈 배열 + split 정보
                   noise_bank.train_signals → Phase 2~3 전용
                   noise_bank.test_signals  → Phase 5 전용

    이 함수 이후 Phase 2 스크립트(dae_pretrain.py)가
    ds1_metas의 record_id 목록과 noise_bank.train_signals를 받아서 실행됨.
    """
    # 1. 환자 분리 검증
    validate_patient_split()

    # 2. MIT-BIH 메타데이터 로드
    ds1_metas = load_all_records_meta(DS1_RECORDS, cfg.mitdb_path, 'DS1 (Train)')
    ds2_metas = load_all_records_meta(DS2_RECORDS, cfg.mitdb_path, 'DS2 (Test)')

    # 3. NSTDB 노이즈 로드 + split
    noise_bank = load_noise_bank(cfg.nstdb_path, NOISE_TYPES)
    validate_noise_bank(noise_bank)

    # 4. 최종 요약
    print(f"\n{'='*60}")
    print(f"[Phase 1 완료] 다음 단계로 넘길 데이터")
    print(f"{'='*60}")
    print(f"  DS1 학습 환자  : {len(ds1_metas)}명 → Phase 2 DAE 학습, Phase 3 분류기 학습")
    print(f"  DS2 테스트 환자: {len(ds2_metas)}명 → Phase 5 Track A/B 평가")
    print(f"  노이즈 train split : {[k.upper() for k in noise_bank.train_signals]} "
          f"→ Phase 2~3")
    print(f"  노이즈 test split  : {[k.upper() for k in noise_bank.test_signals]}  "
          f"→ Phase 5")

    return ds1_metas, ds2_metas, noise_bank


if __name__ == '__main__':
    cfg = PathConfig()
    ds1_metas, ds2_metas, noise_bank = run_phase1(cfg)