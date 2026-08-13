from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

import config
import model as model_module
import preprocessing
from schemas import (
    ECGChunkRequest,
    PredictResponse,
    BeatPrediction,
    HealthResponse,
    ALERT_CLASSES,
)

_state: dict = {"model": None, "device": None, "load_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model, device = model_module.load_model()
        _state["model"] = model
        _state["device"] = device
        print(f"[startup] 모델 로드 완료 device={device} version={config.MODEL_VERSION}")
    except Exception as e:

        _state["load_error"] = str(e)
        print(f"[startup][WARNING] 모델 로드 실패: {e}")
    yield
    _state.clear()


app = FastAPI(title="ECG Arrhythmia Classification API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model_loaded = _state["model"] is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=config.MODEL_VERSION if model_loaded else None,
        expected_sampling_rate=config.EXPECTED_FS,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: ECGChunkRequest) -> PredictResponse:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail=f"모델이 로드되지 않았습니다: {_state['load_error']}")

    duration_sec = len(request.signal) / request.sampling_rate
    if not (config.MIN_CHUNK_SEC <= duration_sec <= config.MAX_CHUNK_SEC):
        raise HTTPException(
            status_code=422,
            detail=f"청크 길이는 {config.MIN_CHUNK_SEC}~{config.MAX_CHUNK_SEC}초여야 합니다 "
                   f"(받은 값: {duration_sec:.1f}초)",
        )

    try:
        extracted = preprocessing.chunk_to_beats(request.signal, request.sampling_rate)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"신호 전처리 실패: {e}")

    if not extracted:
        return PredictResponse(
            device_id=request.device_id,
            chunk_start_time=request.chunk_start_time,
            num_beats_detected=0,
            beats=[],
            alert=False,
            model_version=config.MODEL_VERSION,
        )

    windows = [b.window for b in extracted]
    predictions = model_module.predict_beats(_state["model"], _state["device"], windows)

    beats = [
        BeatPrediction(
            r_peak_sample=ext.r_peak_sample,
            r_peak_offset_sec=ext.r_peak_offset_sec,
            predicted_class=pred["predicted_class"],
            class_probabilities=pred["class_probabilities"],
            signal_quality_flag=None,  
        )
        for ext, pred in zip(extracted, predictions)
    ]

    alert = any(b.predicted_class in ALERT_CLASSES for b in beats)

    return PredictResponse(
        device_id=request.device_id,
        chunk_start_time=request.chunk_start_time,
        num_beats_detected=len(beats),
        beats=beats,
        alert=alert,
        model_version=config.MODEL_VERSION,
    )
