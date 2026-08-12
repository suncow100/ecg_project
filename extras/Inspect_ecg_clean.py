"""
inspect_ecg_clean.py

목적:
    build_noisy_dataset.py 로 생성한 *_noisy.npy 에 대해
    nk.ecg_clean(method="neurokit") 을 레코드 단위(전체 길이)로 적용하고,
    원본(clean) / 노이즈 삽입(noisy) / 필터링 결과(cleaned) 를 함께 시각화한다.

    이 스크립트는 preprocess.py 에 정식으로 편입하기 전,
    필터링 결과를 눈으로 먼저 확인하기 위한 용도다.
    -> 세그멘테이션(비트 단위 자르기)은 하지 않음. 레코드 전체 신호 레벨에서만 확인.

사용 예:
    python inspect_ecg_clean.py --record 100
    python inspect_ecg_clean.py --record 100 --start 10 --duration 8
    python inspect_ecg_clean.py --record 100 --channel 0 --no-show --save
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

# ---------------------------------------------------------------------------
# 프로젝트 경로/설정 - 실제 config.py 값과 다르면 이 블록만 수정하면 됩니다.
# ---------------------------------------------------------------------------
DATA_DIR = Path("dataset")          # *_noisy.npy, *_noise_manifest.json 위치
OUTPUT_DIR = Path("outputs/ecg_clean_inspect")
FS = 360                            # MIT-BIH 샘플링 레이트
CLEAN_METHOD = "neurokit"


def load_noisy(record: str) -> np.ndarray:
    """build_noisy_dataset.py 가 생성한 노이즈 삽입 신호를 로드."""
    path = DATA_DIR / f"{record}_noisy.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} 를 찾을 수 없습니다. build_noisy_dataset.py 실행 순서를 확인하세요.")
    arr = np.load(path)
    return arr


def load_clean_reference(record: str) -> np.ndarray | None:
    """
    노이즈 삽입 전 원본(clean) 신호가 있다면 로드해서 3-way 비교에 사용.
    파일명이 프로젝트마다 다를 수 있어 몇 가지 후보를 시도한다.
    없으면 None을 반환하고 noisy vs cleaned 2-way 비교만 수행.
    """
    candidates = [
        DATA_DIR / f"{record}.npy",
        DATA_DIR / f"{record}_clean.npy",
        DATA_DIR / f"{record}_orig.npy",
    ]
    for path in candidates:
        if path.exists():
            return np.load(path)
    return None


def load_manifest(record: str) -> dict | None:
    path = DATA_DIR / f"{record}_noise_manifest.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def apply_ecg_clean(signal: np.ndarray, fs: int = FS, method: str = CLEAN_METHOD) -> np.ndarray:
    """
    레코드 전체 길이에 대해 1회 필터링.
    signal: (N,) 단일 채널 또는 (N, C) 다채널
    비트 단위로 자른 뒤 호출하면 edge artifact가 각 윈도우를 오염시키므로
    반드시 세그멘테이션 이전, 전체 신호에 대해 적용해야 함.
    """
    if signal.ndim == 1:
        cleaned = nk.ecg_clean(signal, sampling_rate=fs, method=method)
        _check_finite(cleaned, "channel 0")
        return cleaned

    # 다채널: 채널별로 독립 적용
    cleaned = np.zeros_like(signal, dtype=float)
    for ch in range(signal.shape[1]):
        cleaned[:, ch] = nk.ecg_clean(signal[:, ch], sampling_rate=fs, method=method)
        _check_finite(cleaned[:, ch], f"channel {ch}")
    return cleaned


def _check_finite(arr: np.ndarray, label: str) -> None:
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()
    if n_nan or n_inf:
        print(f"[WARNING] {label}: NaN={n_nan}, Inf={n_inf} 발견 (필터 발산 가능성, 낮은 SNR 구간 확인 필요)")


def plot_comparison(
    record: str,
    noisy: np.ndarray,
    cleaned: np.ndarray,
    clean_ref: np.ndarray | None,
    fs: int,
    start_sec: float,
    duration_sec: float,
    channel: int,
    save: bool,
    show: bool,
) -> None:
    start = int(start_sec * fs)
    end = int((start_sec + duration_sec) * fs)

    def get_ch(arr):
        return arr[start:end, channel] if arr.ndim > 1 else arr[start:end]

    t = np.arange(start, end) / fs

    n_rows = 3 if clean_ref is not None else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.2 * n_rows), sharex=True)
    if n_rows == 2:
        axes = list(axes)

    row = 0
    if clean_ref is not None:
        axes[row].plot(t, get_ch(clean_ref), color="tab:green", linewidth=0.9)
        axes[row].set_title(f"Record {record} - Original (pre-noise) reference")
        row += 1

    axes[row].plot(t, get_ch(noisy), color="tab:red", linewidth=0.9)
    axes[row].set_title(f"Record {record} - Noisy (noise-injected) input")
    row += 1

    axes[row].plot(t, get_ch(cleaned), color="tab:blue", linewidth=0.9)
    axes[row].set_title(f"Record {record} - After nk.ecg_clean(method='{CLEAN_METHOD}')")
    axes[row].set_xlabel("Time (s)")

    for ax in axes:
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)

    fig.tight_layout()

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{record}_ecg_clean_ch{channel}_{start_sec}-{start_sec+duration_sec}s.png"
        fig.savefig(out_path, dpi=150)
        print(f"[INFO] 그래프 저장: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def print_signal_stats(noisy: np.ndarray, cleaned: np.ndarray, channel: int) -> None:
    n = noisy[:, channel] if noisy.ndim > 1 else noisy
    c = cleaned[:, channel] if cleaned.ndim > 1 else cleaned
    print(f"--- channel {channel} 통계 ---")
    print(f"  noisy   : std={np.std(n):.4f}, min={np.min(n):.4f}, max={np.max(n):.4f}")
    print(f"  cleaned : std={np.std(c):.4f}, min={np.min(c):.4f}, max={np.max(c):.4f}")


def main():
    parser = argparse.ArgumentParser(description="nk.ecg_clean() 적용 결과 시각화")
    parser.add_argument("--record", type=str, default="100", help="레코드 번호 (예: 100)")
    parser.add_argument("--start", type=float, default=0.0, help="표시 시작 시각(초)")
    parser.add_argument("--duration", type=float, default=10.0, help="표시 구간 길이(초)")
    parser.add_argument("--channel", type=int, default=0, help="다채널일 경우 표시할 채널 인덱스")
    parser.add_argument("--fs", type=int, default=FS, help="샘플링 레이트")
    parser.add_argument("--save", action="store_true", help="그래프를 PNG로 저장")
    parser.add_argument("--no-show", dest="show", action="store_false", help="화면에 표시하지 않음(저장만)")
    args = parser.parse_args()

    noisy = load_noisy(args.record)
    clean_ref = load_clean_reference(args.record)
    manifest = load_manifest(args.record)

    if manifest is not None:
        print(f"[INFO] noise manifest 발견: {DATA_DIR / (args.record + '_noise_manifest.json')}")
        print(json.dumps(manifest, indent=2, ensure_ascii=False)[:500])

    print(f"[INFO] noisy shape: {noisy.shape}, dtype: {noisy.dtype}")

    cleaned = apply_ecg_clean(noisy, fs=args.fs, method=CLEAN_METHOD)

    print_signal_stats(noisy, cleaned, args.channel)

    plot_comparison(
        record=args.record,
        noisy=noisy,
        cleaned=cleaned,
        clean_ref=clean_ref,
        fs=args.fs,
        start_sec=args.start,
        duration_sec=args.duration,
        channel=args.channel,
        save=args.save,
        show=args.show,
    )


if __name__ == "__main__":
    main()