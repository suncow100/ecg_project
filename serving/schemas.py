from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

class AAMIClass(str, Enum):
    N = "N"  # Normal
    S = "S"  # Supraventricular ectopic
    V = "V"  # Ventricular ectopic
    F = "F"  # Fusion

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

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    expected_sampling_rate: int = Field(default=360, description="모델 학습 기준 sampling rate")
