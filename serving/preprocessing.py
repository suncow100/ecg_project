"""
serving/preprocessing.py

Pipeline stage: raw signal (임의 sampling_rate) --> 360Hz 리샘플링(필요시)
             --> nk.ecg_clean() --> nk.ecg_peaks() (Track B 방식 런타임 R-peak)
             --> 250-sample 윈도우 추출 (100 pre-R / 150 post-R)
             --> per-beat Z-score --> 모델 입력 텐서

Design rationale (인터뷰용):
- 학습 시(preprocess.py)는 annotation 기반 R-peak(ground truth)를 썼지만,
  서빙 시점엔 정답 annotation이 없다. 따라서 여기서는 NeuroKit2의 R-peak
  검출기를 쓴다 -- 이게 바로 프로젝트에서 별도로 설계한 "Track B(end-to-end)"
  평가와 정확히 같은 조건이다. 즉 이 함수의 실제 정확도는 Track B 평가
  결과로 이미 특성화되어 있어야 하고, 그 결과가 SNR 기반 reject threshold의
  근거가 된다 (TODO: threshold 미도출 상태).
- 리샘플링은 nk.signal_resample()을 사용해 프로젝트 전반의 NeuroKit2
  의존성과 일관성을 유지한다 (scipy를 별도로 섞지 않음).
- 청크 경계에 걸친 R-peak(윈도우를 다 못 채우는 beat)는 조용히 버린다.
  이는 데이터 손실이 아니라 "다음 청크에서 다시 검출될 가능성이 높은
  beat"이므로 -- 청크가 충분히 겹치거나 연속적으로 전송된다면 실질적
  누락은 발생하지 않는다. (TODO: 청크 간 겹침(overlap) 전략 미구현 --
  현재는 완전 분리된 청크를 가정하므로 경계 beat는 통계적으로 소량 누락됨)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import neurokit2 as nk

import config

EPS = 1e-8


@dataclass
class ExtractedBeat:
    window: np.ndarray       # (250,) float32, z-score normalized
    r_peak_sample: int       # 원본(리샘플링 후) 청크 내 인덱스
    r_peak_offset_sec: float  # 청크 시작 기준 초


def resample_if_needed(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    if sampling_rate == config.EXPECTED_FS:
        return signal
    return nk.signal_resample(
        signal, sampling_rate=sampling_rate, desired_sampling_rate=config.EXPECTED_FS
    )


def clean_and_detect_peaks(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """청크 전체를 한 번에 필터링 -- preprocess.py(학습)와 동일하게 beat 단위가
    아니라 신호 전체 단위로 nk.ecg_clean()을 적용해 edge transient를 청크
    시작/끝 근처로만 국한시킨다."""
    cleaned = nk.ecg_clean(signal, sampling_rate=config.EXPECTED_FS, method="neurokit")

    _, info = nk.ecg_peaks(cleaned, sampling_rate=config.EXPECTED_FS, method="neurokit")
    r_peaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)

    return cleaned, r_peaks


def extract_beats(cleaned: np.ndarray, r_peaks: np.ndarray) -> list[ExtractedBeat]:
    beats: list[ExtractedBeat] = []
    sig_len = len(cleaned)

    for peak in r_peaks:
        start = int(peak) - config.BEAT_PRE_SAMPLES
        end = int(peak) + config.BEAT_POST_SAMPLES
        if start < 0 or end > sig_len:
            continue  # 청크 경계 beat -- docstring 참조

        window = cleaned[start:end].astype(np.float32)
        mu, sigma = window.mean(), window.std()
        window = (window - mu) / (sigma + EPS)

        beats.append(
            ExtractedBeat(
                window=window,
                r_peak_sample=int(peak),
                r_peak_offset_sec=float(peak) / config.EXPECTED_FS,
            )
        )

    return beats


def chunk_to_beats(signal: list[float], sampling_rate: int) -> list[ExtractedBeat]:
    """엔드투엔드 진입점: main.py는 이 함수 하나만 호출하면 된다."""
    arr = np.asarray(signal, dtype=np.float32)
    arr = resample_if_needed(arr, sampling_rate)
    cleaned, r_peaks = clean_and_detect_peaks(arr)
    return extract_beats(cleaned, r_peaks)
