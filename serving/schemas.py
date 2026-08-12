"""
schemas.py

Pydantic 모델 -- FastAPI가 주고받는 JSON의 "모양"을 타입으로 명시.
여기서 정의된 필드/타입에 안 맞는 요청은 FastAPI가 자동으로 422를 반환하고,
/docs (Swagger UI)에 자동으로 스펙이 노출된다.

Design rationale (인터뷰용):
- /predict의 입력 단위는 "청크(chunk)"다. 웨어러블(Holter형) 기기가 3~7일치를
  기록하더라도, 서버는 그 전체를 한 번에 받지 않는다. 실시간 부정맥 알림
  (webhook)이 목적이므로, 기기/게이트웨이가 짧은 구간(권장 10~30초)을 주기적으로
  전송하고 서버는 매 청크를 독립적으로(stateless) 처리한다. "3~7일 기록"은
  이 청크 호출이 시간축으로 반복되는 것으로 구현된다 -- 서버가 여러 날짜의
  데이터를 세션 상태로 들고 있을 필요가 없다.
- sampling_rate를 요청에 명시적으로 받는 이유: 학습은 MIT-BIH 360Hz 기준이지만,
  실제 웨어러블 기기는 다른 sampling rate를 쓸 수 있다. 서버가 리샘플링 책임을
  지도록 설계해서, 기기 스펙이 바뀌어도 모델 재학습 없이 대응 가능하게 한다.
- 응답은 청크 안에서 검출된 "beat별" 리스트다. 청크 하나에 여러 beat가 있을 수
  있으므로 리스트 구조가 자연스럽다.
- signal_quality_flag는 지금은 placeholder다. Track B(NeuroKit2 런타임 R-peak
  검출) 평가에서 SNR별 정확도를 측정한 뒤, 특정 SNR 이하에서는 예측을 신뢰하지
  않고 flag만 세우는 threshold를 도출할 예정 (TODO, 아직 미구현).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# 공통 타입
# ---------------------------------------------------------------------------
class AAMIClass(str, Enum):
    """AAMI 4-class 레이블. class_4_mapping.AAMI_CLASSES_4 와 순서/값이 반드시 일치해야 함."""
    N = "N"  # Normal
    S = "S"  # Supraventricular ectopic
    V = "V"  # Ventricular ectopic
    F = "F"  # Fusion


# 임상적으로 즉시 알림이 필요하다고 보는 클래스.
# TODO(백선우): 이 기준이 과도하게 민감한지(webhook spam) 임상 근거로 방어 가능한지
# 검토. 우선은 V(심실 이소성) 단독 검출도 알림 대상으로 보수적으로 잡음.
ALERT_CLASSES = {AAMIClass.V, AAMIClass.S}


# ---------------------------------------------------------------------------
# /predict 요청
# ---------------------------------------------------------------------------
class ECGChunkRequest(BaseModel):
    device_id: str = Field(..., description="기기 고유 ID (환자/기기 매핑용)")
    sampling_rate: int = Field(
        ..., gt=0, le=2000,
        description="이 청크의 원본 샘플링레이트(Hz). 360이 아니면 서버가 360Hz로 리샘플링함",
    )
    chunk_start_time: datetime = Field(..., description="이 청크의 시작 시각 (ISO 8601, UTC 권장)")
    signal: list[float] = Field(
        ..., min_length=360, max_length=360 * 60,
        description="단일 리드 원시 ECG 샘플 (전처리/필터링 되지 않은 raw 값)",
    )

    @field_validator("signal")
    @classmethod
    def check_finite(cls, v: list[float]) -> list[float]:
        # NaN/Inf가 섞여 들어오면 nk.ecg_clean() 단계에서 조용히 전파되어
        # 디버깅이 어려워지므로, 여기서 바로 400으로 거른다.
        if any(x != x or x in (float("inf"), float("-inf")) for x in v):
            raise ValueError("signal에 NaN 또는 Inf 값이 포함되어 있습니다")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "device_id": "holter-0001",
                "sampling_rate": 360,
                "chunk_start_time": "2026-08-06T09:00:00Z",
                "signal": [0.01, 0.02, -0.01],  # 실제로는 sampling_rate * duration 개
            }
        }
    )


# ---------------------------------------------------------------------------
# /predict 응답
# ---------------------------------------------------------------------------
class BeatPrediction(BaseModel):
    r_peak_sample: int = Field(..., description="청크 내 R-peak 샘플 인덱스 (0-based)")
    r_peak_offset_sec: float = Field(..., description="청크 시작 기준 R-peak 시각 오프셋(초)")
    predicted_class: AAMIClass
    class_probabilities: dict[AAMIClass, float] = Field(
        ..., description="softmax 확률, 4-class 합=1.0"
    )
    signal_quality_flag: Optional[str] = Field(
        default=None,
        description="'low_snr' 등 품질 경고. threshold 미도출 상태라 현재는 항상 null (TODO)",
    )


class PredictResponse(BaseModel):
    device_id: str
    chunk_start_time: datetime
    num_beats_detected: int
    beats: list[BeatPrediction]
    alert: bool = Field(..., description="이 청크에서 ALERT_CLASSES에 해당하는 beat가 하나라도 검출됐는지")
    model_version: str = Field(..., description="추론에 사용된 MLflow run ID 또는 checkpoint 태그")


# ---------------------------------------------------------------------------
# /health 응답
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    expected_sampling_rate: int = Field(default=360, description="모델 학습 기준 sampling rate")
