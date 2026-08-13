from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import wfdb
import neurokit2 as nk

import config
from class_4_mapping import AAMI_CLASSES_4, SYMBOL_TO_AAMI_4
import split_config

EPS = 1e-8  # guards against divide-by-zero on flatline windows


# ---------------------------------------------------------------------------
@dataclass
class BeatSet:
    X: list = field(default_factory=list)         # each item: (250,) float32
    y: list = field(default_factory=list)          # AAMI class index
    record_id: list = field(default_factory=list)  # traceability
    symbol: list = field(default_factory=list)      # original annotation symbol


def clean_record(record: int) -> np.ndarray:
    """Load {record}_noisy.npy and apply nk.ecg_clean() once, full length."""
    path = config.DATASET_DIR / f"{record}_noisy.npy"
    noisy = np.load(path)

    if noisy.ndim != 1:
        raise ValueError(
            f"record {record}: expected 1D noisy signal, got shape {noisy.shape}. "
            "Multi-lead support was not wired into this script -- extend clean_record() "
            "and extract_beats() to loop over channels if you need it."
        )

    cleaned = nk.ecg_clean(noisy, sampling_rate=config.FS, method='neurokit')

    n_bad = int(np.isnan(cleaned).sum() + np.isinf(cleaned).sum())
    if n_bad:
        print(f"  [WARNING] record {record}: {n_bad} NaN/Inf samples after ecg_clean "
              f"-- likely filter divergence in a low-SNR window")

    return cleaned


def extract_beats(record: int, cleaned: np.ndarray, beats: BeatSet) -> dict[str, int]:
    """Segment beats around annotation R-peaks, Z-score normalize, append to beats."""
    rec_path = str(config.MITBIH_ROOT / str(record))
    ann = wfdb.rdann(rec_path, "atr")

    n_total = len(ann.symbol)
    n_unmapped = 0
    n_out_of_bounds = 0
    n_kept = 0
    sig_len = len(cleaned)

    for sample, symbol in zip(ann.sample, ann.symbol):
        aami = SYMBOL_TO_AAMI_4.get(symbol)
        if aami is None:
            n_unmapped += 1  # non-beat annotation (rhythm change, artifact marker, etc.)
            continue

        start = int(sample) - config.BEAT_PRE_SAMPLES
        end = int(sample) + config.BEAT_POST_SAMPLES
        if start < 0 or end > sig_len:
            n_out_of_bounds += 1  # beat too close to record start/end
            continue

        window = cleaned[start:end].astype(np.float32)
        mu, sigma = window.mean(), window.std()
        window = (window - mu) / (sigma + EPS)

        beats.X.append(window)
        beats.y.append(AAMI_CLASSES_4.index(aami))
        beats.record_id.append(record)
        beats.symbol.append(symbol)
        n_kept += 1

    return {"total_ann": n_total, "unmapped": n_unmapped,
            "out_of_bounds": n_out_of_bounds, "kept": n_kept}



def compute_split_masks(
    beats: BeatSet,
    ds1: set[int],
    ds2: set[int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pure record-level DS1/DS2 split.
    Class selection is handled by class_4_mapping.py.
    """

    rec = np.array(beats.record_id)

    is_train = np.isin(
        rec,
        list(ds1)
    )

    is_test = np.isin(
        rec,
        list(ds2)
    )

    return is_train, is_test


# ---------------------------------------------------------------------------
def print_report(beats: BeatSet, is_train: np.ndarray, is_test: np.ndarray) -> None:
    y = np.array(beats.y)

    print("\n" + "=" * 60)
    print(f"{'class':<8}{'train':>10}{'test':>10}{'total':>10}")
    for c, name in enumerate(AAMI_CLASSES_4):
        tr = int(((y == c) & is_train).sum())
        te = int(((y == c) & is_test).sum())
        print(f"{name:<8}{tr:>10}{te:>10}{tr + te:>10}")
    print("-" * 60)
    print(f"{'TOTAL':<8}{int(is_train.sum()):>10}{int(is_test.sum()):>10}{len(y):>10}")
    print("=" * 60)


def save_splits(beats: BeatSet, is_train: np.ndarray, is_test: np.ndarray) -> None:
    X = np.stack(beats.X).astype(np.float32)          # (N, 250)
    y = np.array(beats.y, dtype=np.int64)               # (N,)
    rec = np.array(beats.record_id, dtype=np.int64)     # (N,)

    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.DATASET_DIR / "X_train.npy", X[is_train])
    np.save(config.DATASET_DIR / "y_train.npy", y[is_train])
    np.save(config.DATASET_DIR / "X_test.npy", X[is_test])
    np.save(config.DATASET_DIR / "y_test.npy", y[is_test])
    # record_id kept alongside for traceability / later per-record error analysis
    np.save(config.DATASET_DIR / "record_id_train.npy", rec[is_train])
    np.save(config.DATASET_DIR / "record_id_test.npy", rec[is_test])

    print(f"\nSaved to {config.DATASET_DIR}:")
    print(f"  X_train {X[is_train].shape}  y_train {y[is_train].shape}")
    print(f"  X_test  {X[is_test].shape}  y_test  {y[is_test].shape}")


# ---------------------------------------------------------------------------
def main():
    ds1 = set(split_config.DS1_RECORDS)
    ds2 = set(split_config.DS2_RECORDS)

    beats = BeatSet()
    for record in config.ALL_RECORDS:
        print(f"[{record}] cleaning...", end=" ")
        cleaned = clean_record(record)
        stats = extract_beats(record, cleaned, beats)
        print(f"kept {stats['kept']}/{stats['total_ann']} "
              f"(unmapped={stats['unmapped']}, out_of_bounds={stats['out_of_bounds']})")

    is_train, is_test = compute_split_masks(beats, ds1, ds2)
    print_report(beats, is_train, is_test)
    save_splits(beats, is_train, is_test)


if __name__ == "__main__":
    main()
