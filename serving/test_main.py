"""
serving/test_main.py

pending TODO였던 "fastapi.testclient + dummy checkpoint + nk.ecg_simulate 스모크
테스트"를 정식화. 실제 학습된 가중치가 아니라 랜덤 초기화된 체크포인트를 쓰므로
예측 정확도는 검증하지 않는다 -- 목적은 오직 "요청 -> 전처리 -> 추론 -> 응답"
파이프라인이 예외 없이 끝까지 도는지, 그리고 스키마 계약이 지켜지는지다.

실행:
    cd serving && pytest test_main.py -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import neurokit2 as nk
import pytest
import torch
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def dummy_checkpoint_path(tmp_path_factory):
    """랜덤 초기화된 ResNet1D state_dict를 임시 파일로 저장.
    실제 정확도는 검증 대상이 아니므로 학습된 체크포인트가 필요 없다."""
    from model import ResNet1D, NUM_CLASSES

    path = tmp_path_factory.mktemp("ckpt") / "dummy_model.pt"
    m = ResNet1D(num_classes=NUM_CLASSES, dropout=0.3)
    torch.save(m.state_dict(), path)
    return path


@pytest.fixture()
def client(dummy_checkpoint_path, monkeypatch):
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(dummy_checkpoint_path))
    monkeypatch.setenv("MODEL_VERSION", "pytest-dummy")
    monkeypatch.setenv("INFERENCE_DEVICE", "cpu")

    # config.py가 모듈 임포트 시점에 os.environ을 읽으므로, 이미 임포트된 캐시를
    # 지워서 monkeypatch한 환경변수가 실제로 반영되게 한다. model.py/preprocessing.py도
    # "import config"로 예전 config 모듈 객체를 들고 있을 수 있으므로 함께 무효화한다
    # (dummy_checkpoint_path fixture가 이미 model을 한 번 import해서 캐시를 남김).
    import sys
    for mod in ("config", "model", "preprocessing", "main"):
        sys.modules.pop(mod, None)
    import main as main_module

    with TestClient(main_module.app) as c:
        yield c


def make_chunk(sampling_rate=360, duration=10, heart_rate=75, **overrides):
    sig = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=heart_rate)
    payload = {
        "device_id": "pytest-device",
        "sampling_rate": sampling_rate,
        "chunk_start_time": "2026-08-06T09:00:00Z",
        "signal": sig.tolist(),
    }
    payload.update(overrides)
    return payload


def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["model_version"] == "pytest-dummy"
    assert body["expected_sampling_rate"] == 360


def test_predict_happy_path_360hz(client):
    r = client.post("/predict", json=make_chunk(sampling_rate=360))
    assert r.status_code == 200
    body = r.json()
    assert body["num_beats_detected"] > 0
    assert len(body["beats"]) == body["num_beats_detected"]
    for beat in body["beats"]:
        assert beat["predicted_class"] in ("N", "S", "V", "F")
        assert abs(sum(beat["class_probabilities"].values()) - 1.0) < 1e-4


def test_predict_resamples_non_native_rate(client):
    r = client.post("/predict", json=make_chunk(sampling_rate=200))
    assert r.status_code == 200
    assert r.json()["num_beats_detected"] > 0


def test_predict_flatline_returns_zero_beats_no_alert(client):
    payload = make_chunk()
    payload["signal"] = [0.0] * 3600
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["num_beats_detected"] == 0
    assert body["alert"] is False
    assert body["beats"] == []


def test_predict_rejects_chunk_shorter_than_schema_minimum(client):
    payload = make_chunk()
    payload["signal"] = [0.0] * 100  # schemas.py min_length=360
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_schema_rejects_nan_signal():
    # HTTP JSON은 NaN을 표현할 수 없으므로(RFC 8259), pydantic 모델을 직접 호출해
    # validator 동작만 확인한다.
    from schemas import ECGChunkRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ECGChunkRequest(
            device_id="d", sampling_rate=360,
            chunk_start_time="2026-08-06T09:00:00Z",
            signal=[float("nan")] * 400,
        )
