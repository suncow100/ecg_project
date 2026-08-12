"""
serving/config.py

학습용 config.py(PROJECT_ROOT 기준 로컬 절대경로)와는 의도적으로 분리했다.
서빙 컨테이너는 학습 컨테이너와 별개 이미지로 빌드되고(heavyweight training
image vs lightweight serving image), 로컬 파일시스템 경로가 아니라 환경변수로
설정을 주입받는 것이 12-factor 원칙에 맞고, docker-compose에서 볼륨 마운트
경로를 바꿔도 코드 수정이 필요 없다.

값이 없으면 기본값으로 fallback하되, MODEL_CHECKPOINT_PATH처럼 없으면 서버가
아예 뜨면 안 되는 값은 main.py의 startup에서 명시적으로 실패시킨다.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 모델 학습 시 고정된 값 -- config.py(학습)와 반드시 동기화되어야 함.
# TODO(백선우): 이 상수들은 학습 config.py에도 있어서 두 곳에 중복됨.
# 지금은 두 이미지가 별도 레포/빌드 컨텍스트라 import 공유가 번거로워 값만
# 복사했는데, 인터뷰에서 "왜 안 합쳤냐"는 질문이 나올 수 있음 --
# 답변: 서빙 이미지가 학습 코드 전체(wfdb, mlflow 등)에 의존하게 만들고
# 싶지 않았고, 이 상수 자체는 모델 아키텍처에 귀속된 값이라 사실상
# "모델 계약(contract)"으로 보고 체크포인트와 함께 버전 관리하는 게
# 더 안전하다고 판단함 (학습 config가 바뀌어도 이미 배포된 모델의
# 서빙 동작은 안 바뀌어야 하므로).
EXPECTED_FS = 360
BEAT_PRE_SAMPLES = 100
BEAT_POST_SAMPLES = 150
BEAT_WINDOW_SAMPLES = BEAT_PRE_SAMPLES + BEAT_POST_SAMPLES  # 250

# ---------------------------------------------------------------------------
# 배포 환경 설정 -- 환경변수로 주입
# ---------------------------------------------------------------------------
MODEL_CHECKPOINT_PATH = Path(
    os.environ.get("MODEL_CHECKPOINT_PATH", "/app/checkpoints/best_model.pt")
)

# MLflow run ID. train.py가 mlflow.start_run()으로 남긴 실제 run id를
# 배포 시점에 채워 넣는다 (pending TODO였던 "placeholder MLflow run ID 교체").
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")

DEVICE = os.environ.get("INFERENCE_DEVICE", "cpu")  # 서빙은 기본 CPU, GPU 필요시 "cuda"

# 청크 하나에 허용하는 신호 길이 (초). 너무 짧으면 R-peak가 몇 개 안 잡히고,
# 너무 길면 알림 지연이 커짐. 10~30초를 권장 범위로 설정.
MIN_CHUNK_SEC = 1
MAX_CHUNK_SEC = 60
