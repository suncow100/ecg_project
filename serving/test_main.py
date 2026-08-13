

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
    # validator 동작만 확인
    from schemas import ECGChunkRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ECGChunkRequest(
            device_id="d", sampling_rate=360,
            chunk_start_time="2026-08-06T09:00:00Z",
            signal=[float("nan")] * 400,
        )
