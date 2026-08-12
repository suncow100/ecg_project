"""
noise_synthesis.py -- LIBRARY ONLY. No CLI, no __main__.

Synthesizes combined BW+MA+EM noise onto ECG beats, strictly respecting
record-level train/test partitioning.

Critical invariant this module exists to protect:
    record in DS1_RECORDS  -> noise MUST come from NoiseBank's train-partition
    record in DS2_RECORDS  -> noise MUST come from NoiseBank's test-partition
A record's noise partition is never independent of its DS1/DS2 assignment --
if it were, the same NSTDB noise segment could shape both a training beat
and a test beat, silently reopening the leakage problem the disjoint 70/30
NoiseBank split (Phase 1) exists to prevent.

Typical usage from dataset.py:
    from noise_synthesis import get_injector
    injector = get_injector()                      # built once, cached
    noisy_beat, snr_db = injector.inject(clean_beat, record_id=101)

NOTE: `NoiseBank` below is a minimal reference implementation matching the
description of your Phase 1 NoiseBank (disjoint 70/30 time-axis split per
noise type). If phase1_data_prep.py's NoiseBank has a different method
signature, swap it in here -- the only contract PartitionedNoiseInjector
needs is:
    noise_bank.sample_segment(noise_type: str, partition: str, length: int,
                               rng: np.random.Generator) -> np.ndarray
"""

from __future__ import annotations

import numpy as np
import wfdb

import config

NSTDB_NOISE_TYPES = ("bw", "ma", "em")


# ---------------------------------------------------------------------------
def record_partition_map(ds1_records: list[int], ds2_records: list[int]) -> dict[int, str]:
    """{record_id: 'train'} for DS1, {record_id: 'test'} for DS2."""
    overlap = set(ds1_records) & set(ds2_records)
    if overlap:
        raise ValueError(f"DS1/DS2 overlap, this should never happen: {overlap}")
    m = {r: "train" for r in ds1_records}
    m.update({r: "test" for r in ds2_records})
    return m


# ---------------------------------------------------------------------------
class NoiseBank:
    """Loads bw/ma/em from NSTDB, splits each along the time axis into a
    disjoint train-pool / test-pool (ratio from config.NOISE_BANK_TRAIN_RATIO).
    """

    def __init__(self, nstdb_root=None, train_ratio: float | None = None, channel: int = 0):
        nstdb_root = nstdb_root or config.NSTDB_ROOT
        train_ratio = train_ratio if train_ratio is not None else config.NOISE_BANK_TRAIN_RATIO

        self.partitions: dict[str, dict[str, np.ndarray]] = {"train": {}, "test": {}}
        for noise_type in NSTDB_NOISE_TYPES:
            rec = wfdb.rdrecord(str(nstdb_root / noise_type))
            sig = rec.p_signal[:, channel].astype(np.float64)
            split_idx = int(len(sig) * train_ratio)
            self.partitions["train"][noise_type] = sig[:split_idx]
            self.partitions["test"][noise_type] = sig[split_idx:]

    def sample_segment(
        self, noise_type: str, partition: str, length: int, rng: np.random.Generator
    ) -> np.ndarray:
        pool = self.partitions[partition][noise_type]
        if length > len(pool):
            raise ValueError(
                f"requested length {length} exceeds {partition}/{noise_type} pool size {len(pool)}"
            )
        start = rng.integers(0, len(pool) - length + 1)
        return pool[start : start + length].copy()


# ---------------------------------------------------------------------------
def synthesize_combined_noise(
    noise_bank: NoiseBank,
    partition: str,
    length: int,
    rng: np.random.Generator,
    type_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Combine BW+MA+EM into one noise waveform with randomized mixing weights
    (Dirichlet(1,1,1) by default) so composition varies beat to beat, similar
    to how real noise composition varies with device fit and motion type.
    """
    if type_weights is None:
        type_weights = rng.dirichlet(np.ones(len(NSTDB_NOISE_TYPES)))

    combined = np.zeros(length, dtype=np.float64)
    for w, ntype in zip(type_weights, NSTDB_NOISE_TYPES):
        seg = noise_bank.sample_segment(ntype, partition, length, rng)
        seg = seg - seg.mean()
        combined += w * seg
    return combined


def add_noise_at_snr(clean_signal: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Additive noise scaled so that resulting SNR(clean, noise) == target_snr_db."""
    sig_power = np.mean(clean_signal**2)
    noise_power = np.mean(noise**2)
    if noise_power < 1e-12:
        return clean_signal.copy()
    desired_noise_power = sig_power / (10 ** (target_snr_db / 10))
    scale = np.sqrt(desired_noise_power / noise_power)
    return clean_signal + scale * noise


# ---------------------------------------------------------------------------
class PartitionedNoiseInjector:
    def __init__(
        self,
        noise_bank: NoiseBank,
        record_partition: dict[int, str],
        snr_levels_db: tuple[float, ...] | None = None,
        seed: int | None = None,
    ):
        self.noise_bank = noise_bank
        self.record_partition = record_partition
        self.snr_levels_db = snr_levels_db or config.SNR_LEVELS_DB
        self.rng = np.random.default_rng(seed if seed is not None else config.NOISE_SEED)

    def inject(self, clean_beat: np.ndarray, record_id: int) -> tuple[np.ndarray, float]:
        """Inject combined BW+MA+EM noise into `clean_beat`. The partition
        (train/test) used to sample noise is looked up from `record_id` --
        there is no parameter to override it, so a caller cannot accidentally
        request the wrong partition.
        """
        partition = self.record_partition.get(record_id)
        if partition is None:
            raise KeyError(
                f"record {record_id} is not in DS1_RECORDS or DS2_RECORDS -- "
                "refusing to guess a noise partition for an unregistered record"
            )
        target_snr_db = float(self.rng.choice(self.snr_levels_db))
        noise = synthesize_combined_noise(self.noise_bank, partition, len(clean_beat), self.rng)
        noisy_beat = add_noise_at_snr(clean_beat, noise, target_snr_db)
        return noisy_beat, target_snr_db


# ---------------------------------------------------------------------------
# Lazy singleton factory -- this is what dataset.py should actually import.
# Lazy on purpose: importing this module must not touch disk or require
# split_config.py to already exist (e.g. at test-collection time).
# ---------------------------------------------------------------------------
_injector_cache: PartitionedNoiseInjector | None = None


def get_injector(force_rebuild: bool = False) -> PartitionedNoiseInjector:
    """Build (once) and return the process-wide PartitionedNoiseInjector,
    wired to split_config.DS1_RECORDS / DS2_RECORDS and config.py settings.
    """
    global _injector_cache
    if _injector_cache is not None and not force_rebuild:
        return _injector_cache

    import split_config  # imported lazily: must exist by the time this runs,
    # i.e. after patient_split.py has been executed at least once.

    partition_map = record_partition_map(split_config.DS1_RECORDS, split_config.DS2_RECORDS)
    bank = NoiseBank()
    _injector_cache = PartitionedNoiseInjector(bank, partition_map)
    return _injector_cache