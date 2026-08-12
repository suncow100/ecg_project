"""
verify_noisy_dataset.py

One-off diagnostic script (not part of the training pipeline) to sanity
check build_noisy_dataset.py's output before building anything on top of it.

Checks:
  1. noisy.npy length matches the original clean signal length exactly,
     and contains no NaN/Inf.
  2. The manifest's recorded partition matches what split_config.py says
     this record_id should be (train/test) -- catches any drift between
     the split and what was actually used at noise-injection time.
  3. Visual overlay of clean vs noisy signal across the whole record, with
     a color strip showing which SNR each window got.
  4. Zoomed-in plots of the lowest-SNR and highest-SNR windows, so you can
     visually confirm "low SNR" windows actually look noisy and "high SNR"
     windows look close to clean.
  5. Distribution of SNR levels actually used, to confirm sampling looks
     roughly uniform across config.SNR_LEVELS_DB.

Run:
    python verify_noisy_dataset.py --record 232
    python verify_noisy_dataset.py --record 210
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; script saves PNGs instead of plt.show()
import matplotlib.pyplot as plt
import numpy as np
import wfdb

import config
import split_config


def load_clean_signal(record_id: int, channel: int = 0) -> np.ndarray:
    rec = wfdb.rdrecord(str(config.MITBIH_ROOT / str(record_id)))
    return rec.p_signal[:, channel].astype(np.float64)


def run_checks(record_id: int, out_dir: Path) -> None:
    noisy_path = config.DATASET_DIR / f"{record_id}_noisy.npy"
    manifest_path = config.DATASET_DIR / f"{record_id}_noise_manifest.json"

    noisy = np.load(noisy_path)
    manifest = json.loads(manifest_path.read_text())
    clean = load_clean_signal(record_id)

    print(f"record {record_id}")
    print(f"  clean length={len(clean)}, noisy length={len(noisy)}, match={len(clean) == len(noisy)}")
    print(f"  NaN in noisy: {bool(np.isnan(noisy).any())}, Inf in noisy: {bool(np.isinf(noisy).any())}")

    manifest_partition = manifest["partition"]
    expected_partition = "train" if record_id in split_config.DS1_RECORDS else "test"
    match = manifest_partition == expected_partition
    print(f"  manifest partition='{manifest_partition}', expected(from split_config)='{expected_partition}', "
          f"match={match}")
    if not match:
        print("  !! MISMATCH -- this record's noise may have been drawn from the wrong "
              "NoiseBank partition. Check whether split_config.py changed after this "
              "record was generated.")

    windows = manifest["windows"]
    snrs = [w["snr_db"] for w in windows]
    print(f"  n_windows={len(windows)}")
    unique, counts = np.unique(snrs, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    SNR={u:>5} dB : {c:>4} windows ({100 * c / len(snrs):.1f}%)")

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: full-record overview with an SNR color strip underneath ---
    fig, (ax_sig, ax_snr) = plt.subplots(
        2, 1, figsize=(14, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    t = np.arange(len(noisy)) / config.FS
    ax_sig.plot(t, clean, lw=0.4, alpha=0.6, label="clean")
    ax_sig.plot(t, noisy, lw=0.4, alpha=0.6, label="noisy")
    ax_sig.legend(loc="upper right")
    ax_sig.set_ylabel("amplitude")
    ax_sig.set_title(f"record {record_id} (partition={manifest_partition}) -- full overview")

    snr_arr = np.array(snrs, dtype=float)
    starts = np.array([w["start"] for w in windows]) / config.FS
    ends = np.array([w["end"] for w in windows]) / config.FS
    cmap = plt.cm.RdYlGn  # red = low SNR (noisy), green = high SNR (near-clean)
    norm = plt.Normalize(vmin=min(config.SNR_LEVELS_DB), vmax=max(config.SNR_LEVELS_DB))
    for s, e, snr in zip(starts, ends, snr_arr):
        ax_snr.axvspan(s, e, color=cmap(norm(snr)))
    ax_snr.set_yticks([])
    ax_snr.set_xlabel("time (s)")
    ax_snr.set_ylabel("SNR")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax_snr, orientation="horizontal", pad=0.5, label="window SNR (dB)")
    fig.tight_layout()
    overview_path = out_dir / f"{record_id}_overview.png"
    fig.savefig(overview_path, dpi=120)
    plt.close(fig)
    print(f"  saved {overview_path}")

    # --- Plot 2: zoom into the lowest-SNR and highest-SNR windows ---
    lo_idx = int(np.argmin(snr_arr))
    hi_idx = int(np.argmax(snr_arr))
    for label, idx in [("lowest_snr", lo_idx), ("highest_snr", hi_idx)]:
        w = windows[idx]
        s, e = w["start"], w["end"]
        fig, ax = plt.subplots(figsize=(10, 3))
        tt = np.arange(s, e) / config.FS
        ax.plot(tt, clean[s:e], label="clean", lw=1.2)
        ax.plot(tt, noisy[s:e], label="noisy", lw=1.0, alpha=0.8)
        ax.set_title(f"record {record_id}, window {idx} ({label}, SNR={w['snr_db']} dB)")
        ax.set_xlabel("time (s)")
        ax.legend()
        fig.tight_layout()
        zoom_path = out_dir / f"{record_id}_{label}_window{idx}.png"
        fig.savefig(zoom_path, dpi=120)
        plt.close(fig)
        print(f"  saved {zoom_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("verify_plots"))
    args = parser.parse_args()
    run_checks(args.record, args.out_dir)


if __name__ == "__main__":
    main()