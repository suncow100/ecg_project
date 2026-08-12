"""
patient_split.py

MIT-BIH Arrhythmia Database -- patient(record)-level 8:2 split.
Run this once; it writes split_config.py, which every other script imports.

Why this exists (design rationale, for interview writeup):
- Beat-level random split leaks patient-specific morphology between
  train/test (the failure mode in most published papers).
- The canonical de Chazal DS1/DS2 split fixes this leakage but uses a
  22:22 (50:50) record split, which starves training of data and does
  NOT control for per-class beat distribution. In particular record 232
  alone holds >75% of all S-class (SVEB) beats in the whole database --
  if 232 lands in the test half, the model essentially never sees S
  morphology during training.
- This script keeps the leakage-safe principle (whole records go
  entirely to one side) but (a) targets an 8:2 overall beat-count split
  instead of a fixed 22:22 record split, and (b) explicitly optimizes
  for balanced per-class train/test ratios, with record 232 pinned to
  train (see config.FORCE_TRAIN_RECORDS) so the model actually gets
  exposure to S morphology.

Usage:
    python patient_split.py
    (reads MITBIH_ROOT / writes SPLIT_CONFIG_PATH, both from config.py)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import wfdb

import config

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]

SYMBOL_TO_AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}

# ---------------------------------------------------------------------------
def get_beat_counts_per_record(records: list[int]) -> dict[int, np.ndarray]:
    """Read annotations for each record and count beats per AAMI class."""
    counts: dict[int, np.ndarray] = {}
    for rec_id in records:
        rec_path = str(config.MITBIH_ROOT / str(rec_id))
        ann = wfdb.rdann(rec_path, "atr")
        vec = np.zeros(len(AAMI_CLASSES), dtype=np.int64)
        for sym in ann.symbol:
            aami = SYMBOL_TO_AAMI.get(sym)
            if aami is None:
                continue
            vec[AAMI_CLASSES.index(aami)] += 1
        counts[rec_id] = vec
        print(f"  record {rec_id}: {dict(zip(AAMI_CLASSES, vec.tolist()))}")
    return counts


# ---------------------------------------------------------------------------
@dataclass
class SplitResult:
    train_records: list[int]
    test_records: list[int]
    train_counts: np.ndarray
    test_counts: np.ndarray
    cost: float
    history: list[float] = field(default_factory=list)


def _cost(train_counts: np.ndarray, test_counts: np.ndarray, target_ratio: float) -> float:
    total = train_counts + test_counts
    cost = 0.0
    overall_ratio = train_counts.sum() / total.sum()
    cost += (overall_ratio - target_ratio) ** 2 * 5.0
    for c in range(len(AAMI_CLASSES)):
        if total[c] < 20:
            continue
        class_ratio = train_counts[c] / total[c]
        cost += (class_ratio - target_ratio) ** 2
    return cost


def optimize_split(
    counts: dict[int, np.ndarray],
    target_ratio: float,
    force_train: set[int],
    n_iter: int,
    seed: int,
) -> SplitResult:
    rng = random.Random(seed)
    records = list(counts.keys())

    def s_share(r):
        v = counts[r]
        return v[1] / max(v.sum(), 1)

    sorted_records = sorted(records, key=s_share, reverse=True)
    train, test = set(), set()
    for i, r in enumerate(sorted_records):
        if r in force_train:
            train.add(r)
        elif i % 5 == 0:
            test.add(r)
        else:
            train.add(r)

    def group_counts(group: set[int]) -> np.ndarray:
        return np.sum([counts[r] for r in group], axis=0)

    train_counts = group_counts(train)
    test_counts = group_counts(test)
    cost = _cost(train_counts, test_counts, target_ratio)
    history = [cost]

    temp = 1.0
    cooling = 0.9995
    for _ in range(n_iter):
        temp *= cooling
        r = rng.choice(records)
        if r in force_train:
            continue

        src, dst = (train, test) if r in train else (test, train)
        src.remove(r)
        dst.add(r)
        new_train_counts = group_counts(train)
        new_test_counts = group_counts(test)
        new_cost = _cost(new_train_counts, new_test_counts, target_ratio)

        accept = new_cost < cost or rng.random() < np.exp(-(new_cost - cost) / max(temp, 1e-6))
        if accept:
            cost = new_cost
            train_counts, test_counts = new_train_counts, new_test_counts
        else:
            dst.remove(r)
            src.add(r)

        history.append(cost)

    return SplitResult(
        train_records=sorted(train),
        test_records=sorted(test),
        train_counts=train_counts,
        test_counts=test_counts,
        cost=cost,
        history=history,
    )


# ---------------------------------------------------------------------------
def print_report(result: SplitResult) -> None:
    total = result.train_counts + result.test_counts
    print("\n" + "=" * 60)
    print(f"DS1 (train) records [{len(result.train_records)}]: {result.train_records}")
    print(f"DS2 (test)  records [{len(result.test_records)}]:  {result.test_records}")
    print("-" * 60)
    print(f"{'class':<8}{'train':>10}{'test':>10}{'total':>10}{'train %':>12}")
    for c in range(len(AAMI_CLASSES)):
        tr, te, tot = result.train_counts[c], result.test_counts[c], total[c]
        pct = 100 * tr / tot if tot else float("nan")
        print(f"{AAMI_CLASSES[c]:<8}{tr:>10}{te:>10}{tot:>10}{pct:>11.1f}%")
    overall_pct = 100 * result.train_counts.sum() / total.sum()
    print("-" * 60)
    print(f"{'TOTAL':<8}{result.train_counts.sum():>10}{result.test_counts.sum():>10}"
          f"{total.sum():>10}{overall_pct:>11.1f}%")
    print(f"final cost: {result.cost:.5f}")
    print("=" * 60)


def write_split_config(result: SplitResult, path) -> None:
    """Auto-generate split_config.py. Overwrites any previous version.

    Downstream scripts (dataset.py, noise_synthesis.py) import DS1_RECORDS /
    DS2_RECORDS from here -- never copy-paste record lists by hand again.
    """
    total = result.train_counts + result.test_counts
    lines = [
        '"""',
        "split_config.py -- AUTO-GENERATED by patient_split.py. Do not edit by hand;",
        "re-run patient_split.py if you need a different split.",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Final optimizer cost: {result.cost:.5f}",
        "",
        "Per-class train/test beat counts at generation time:",
    ]
    for c in range(len(AAMI_CLASSES)):
        tr, te, tot = result.train_counts[c], result.test_counts[c], total[c]
        pct = 100 * tr / tot if tot else float("nan")
        lines.append(f"  {AAMI_CLASSES[c]}: train={tr} test={te} total={tot} (train {pct:.1f}%)")
    lines += ['"""', "", f"DS1_RECORDS = {result.train_records}", f"DS2_RECORDS = {result.test_records}", ""]

    path.write_text("\n".join(lines))
    print(f"\nWrote {path}")


# ---------------------------------------------------------------------------
def main():
    print(f"Reading annotations from {config.MITBIH_ROOT} ...")
    counts = get_beat_counts_per_record(config.ALL_RECORDS)

    print("\nOptimizing patient-level split (this takes a few seconds)...")
    result = optimize_split(
        counts,
        target_ratio=config.SPLIT_TARGET_RATIO,
        force_train=config.FORCE_TRAIN_RECORDS,
        n_iter=config.SPLIT_N_ITER,
        seed=config.SPLIT_SEED,
    )
    print_report(result)
    write_split_config(result, config.SPLIT_CONFIG_PATH)


if __name__ == "__main__":
    main()