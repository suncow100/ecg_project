"""
build_noisy_dataset.py

Step 3: materialize the noise-synthesized continuous ECG signal for every
record, respecting each record's DS1/DS2 partition (via noise_synthesis's
PartitionedNoiseInjector). Output: one .npy per record under DATASET_DIR,
plus a JSON manifest recording which SNR / noise mixture each window got
(for later inspection / reproducibility, and for Track B noise-condition
breakdowns).

Why windowed injection instead of one draw over the whole ~30-minute record:
- NSTDB's train/test noise pools (after the 70/30 split) are only ~21 min /
  ~9 min long respectively -- shorter than a full ECG record, so a single
  contiguous draw covering the whole record isn't even possible for the
  test partition.
- Even if it were, using one fixed SNR/noise-mixture for an entire 30-minute
  recording is unrealistic; real noise characteristics drift over time
  (posture changes, activity level, electrode adhesion). Chunking gives
  every ~10s window an independently drawn SNR and BW/MA/EM mixture.

This script does NOT do R-peak detection, filtering, or segmentation --
that's the next step, once the noisy continuous signal exists on disk.

Run:
    python build_noisy_dataset.py
"""

from __future__ import annotations

import json

import numpy as np
import wfdb

import config
from Noise_synthesis import get_injector


def load_clean_signal(record_id: int, channel: int = 0) -> np.ndarray:
    """Channel 0 is typically MLII in MIT-BIH; kept configurable via `channel`
    in case a downstream script wants channel 1 (usually V1/V5) instead.
    """
    rec = wfdb.rdrecord(str(config.MITBIH_ROOT / str(record_id)))
    return rec.p_signal[:, channel].astype(np.float64)


def synthesize_noisy_record(
    clean_signal: np.ndarray, record_id: int, injector
) -> tuple[np.ndarray, list[dict]]:
    """Inject noise window-by-window, respecting record_id's DS1/DS2
    partition on every single window (the injector looks this up
    internally per call -- it cannot vary within a record).
    """
    window_len = int(config.NOISE_WINDOW_SEC * config.FS)
    n_samples = len(clean_signal)
    noisy = np.empty_like(clean_signal)
    manifest = []

    for start in range(0, n_samples, window_len):
        end = min(start + window_len, n_samples)
        chunk = clean_signal[start:end]
        noisy_chunk, snr_db = injector.inject(chunk, record_id)
        noisy[start:end] = noisy_chunk
        manifest.append({"start": int(start), "end": int(end), "snr_db": snr_db})

    return noisy, manifest


def main():
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    injector = get_injector()

    for record_id in config.ALL_RECORDS:
        print(f"processing record {record_id} ...")
        clean = load_clean_signal(record_id)
        noisy, manifest = synthesize_noisy_record(clean, record_id, injector)

        assert noisy.shape == clean.shape, "windowed reconstruction must preserve length exactly"

        out_path = config.DATASET_DIR / f"{record_id}_noisy.npy"
        np.save(out_path, noisy)

        manifest_path = config.DATASET_DIR / f"{record_id}_noise_manifest.json"
        manifest_path.write_text(json.dumps({
            "record_id": record_id,
            "partition": injector.record_partition[record_id],
            "window_sec": config.NOISE_WINDOW_SEC,
            "n_samples": int(len(noisy)),
            "n_windows": len(manifest),
            "windows": manifest,
        }, indent=2))

        print(f"  wrote {out_path} ({len(noisy)} samples, {len(manifest)} windows, "
              f"partition={injector.record_partition[record_id]})")


if __name__ == "__main__":
    main()