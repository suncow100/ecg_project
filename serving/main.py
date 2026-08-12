"""
serving/main.py

FastAPI 진입점. 청크(raw ECG) 요청을 받아
preprocessing.chunk_to_beats() -> model.predict_beats() 순으로 조립하고
schemas.py 형태로 응답한다.

Design rationale (인터뷰용):
- 모델은 프로세스 시작 시(lifespan) 딱 한 번만 로드한다. 요청마다 torch.load()를
  하면 디스크 I/O + BatchNorm 재초기화 비용이 요청 latency에 그대로 얹혀서
  실시간 알림 목적에 맞지 않는다.
- /predict는 완전히 stateless다 -- 이전 청크의 어떤 상태도 참조하지 않는다.
  이건 "3~7일치 연속 기록"을 어떻게 다룰지에 대한 설계 결정과 직결된다:
  서버가 세션/윈도우 상태를 들고 있지 않으므로 수평 확장이 쉽고, 청크
  하나가 유실돼도 다음 청크 처리에 영향이 없다.
- alert 판단(webhook 트리거 여부)은 지금은 이 청크 안에서 ALERT_CLASSES에
  해당하는 beat가 하나라도 있으면 True로 잡는 단순 규칙이다. 실제 webhook
  전송(POST to notification endpoint)은 아직 미구현 -- main.py는 alert
  플래그만 응답에 실어 보내고, 실제 발송은 다음 단계에서 붙일 예정.
"""

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

# 요청마다 다시 로드하지 않도록 프로세스 전역에 보관.
# 사이즈가 크지 않은 포트폴리오 스코프라 별도 상태 관리 클래스 없이 dict로 충분.
_state: dict = {"model": None, "device": None, "load_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model, device = model_module.load_model()
        _state["model"] = model
        _state["device"] = device
        print(f"[startup] 모델 로드 완료 device={device} version={config.MODEL_VERSION}")
    except Exception as e:
        # 모델 로드 실패 시에도 프로세스는 뜨게 둔다 -- /health가 model_loaded=False를
        # 보고할 수 있어야 Docker healthcheck / 오케스트레이터가 원인을 구분할 수 있다.
        # (컨테이너가 아예 죽어버리면 "왜 죽었는지"가 로그에만 남고 API로는 안 보임)
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
        # 503: 클라이언트 요청 자체는 문제가 없는데 서버가 지금 응답할 수 없는 상태.
        # 400/422(요청 문제)와 명확히 구분해야 클라이언트가 재시도 로직을 올바르게 짤 수 있음.
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
        # nk.ecg_clean/ecg_peaks가 극단적으로 낮은 SNR 신호에서 예외를 던질 수 있음
        # (예: flatline, 필터 발산). 500 대신 422로 "이 입력을 처리할 수 없다"고 명시.
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
            signal_quality_flag=None,  # TODO: Track B 기반 SNR threshold 도출 후 채우기
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
