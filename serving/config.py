from __future__ import annotations

import os
from pathlib import Path

EXPECTED_FS = 360
BEAT_PRE_SAMPLES = 100
BEAT_POST_SAMPLES = 150
BEAT_WINDOW_SAMPLES = BEAT_PRE_SAMPLES + BEAT_POST_SAMPLES  # 250

MODEL_CHECKPOINT_PATH = Path(
    os.environ.get("MODEL_CHECKPOINT_PATH", "/app/checkpoints/best_model.pt")
)

MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")

DEVICE = os.environ.get("INFERENCE_DEVICE", "cpu")  

MIN_CHUNK_SEC = 1
MAX_CHUNK_SEC = 60
