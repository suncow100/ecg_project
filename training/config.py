from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MITBIH_ROOT = PROJECT_ROOT / "mit-bih-arrhythmia-database-1.0.0"
NSTDB_ROOT = PROJECT_ROOT / "nstdb_noise"

SPLIT_CONFIG_PATH = PROJECT_ROOT / "split_config.py"
DATASET_DIR = PROJECT_ROOT / "dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"


FS = 360  # Hz, MIT-BIH native sampling rate

# 44 non-paced records (excludes 102, 104, 107, 217 per AAMI convention).
# Single source of truth -- patient_split.py and build_noisy_dataset.py both
# import this instead of keeping their own copies.
ALL_RECORDS = [
    100, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117,
    118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209,
    210, 212, 213, 214, 215, 219, 220, 221, 222, 223, 228, 230, 231, 232,
    233, 234,
]

# ---------------------------------------------------------------------------
# patient_split.py settings
# ---------------------------------------------------------------------------
SPLIT_TARGET_RATIO = 0.8       # target train fraction, overall and per-class
SPLIT_N_ITER = 30000           # simulated annealing iterations
SPLIT_SEED = 42
# Records forced into train regardless of what the optimizer finds --
# record 232 alone holds >75% of all S-class (SVEB) beats in MIT-BIH, so it
# must not be allowed to land in test.
FORCE_TRAIN_RECORDS = {232}

# ---------------------------------------------------------------------------
# noise_synthesis.py settings
# ---------------------------------------------------------------------------
NOISE_BANK_TRAIN_RATIO = 0.7           # NSTDB disjoint time-axis split ratio
SNR_LEVELS_DB = (24, 18, 12, 6, 0, -6)  # randomly sampled per injected window
NOISE_SEED = 42

# Continuous-signal injection is done in windows, not over a whole ~30-min
# record at once: NSTDB's disjoint test-partition noise pool (~9 min) is
# shorter than a full ECG record, and a single fixed SNR/mixture for 30
# minutes is unrealistic anyway. Each window gets its own random draw.
NOISE_WINDOW_SEC = 10
BEAT_PRE_SAMPLES = 100     # R-peak 기준 이전 샘플 수 (비대칭 윈도우)
BEAT_POST_SAMPLES = 150    # R-peak 기준 이후 샘플 수


# train.py settings
INTERNAL_VAL_RATIO = 0.15      # DS1 record 중 internal validation 비율
INTERNAL_VAL_SEED = 42
BATCH_SIZE = 256
TRAIN_EPOCHS = 120
EARLY_STOP_PATIENCE = 20
LEARNING_RATE = 5e-4
Q_CLASS_WEIGHT_CAP = 50.0
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "ecg_arrhythmia_classifier"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pt"
