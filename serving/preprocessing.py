"""
serving/preprocessing.py

Pipeline stage: raw signal (임의 sampling_rate) --> 360Hz 리샘플링(필요시)
             --> nk.ecg_clean() --> nk.ecg_peaks() (Track B 방식 런타임 R-peak)
             --> 250-sample 윈도우 추출 (100 pre-R / 150 post-R)
             --> per-beat Z-score --> 모델 입력 텐서
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
    arr = np.asarray(signal, dtype=np.float32)
    arr = resample_if_needed(arr, sampling_rate)
    cleaned, r_peaks = clean_and_detect_peaks(arr)
    return extract_beats(cleaned, r_peaks)
