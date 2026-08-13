from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

CLASS_COLORS = {"N": "#888780", "S": "#EF9F27", "V": "#E24B4A", "F": "#7F77DD"}
# 로컬 실행 시 기본값은 localhost. docker-compose 안에서는 서비스명(serving)으로
# 서로를 찾으므로, compose가 환경변수로 http://serving:8000 을 주입해 덮어쓴다.
DEFAULT_API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="ECG Arrhythmia Demo", layout="wide")
st.title("ECG 부정맥 분류 데모")
st.caption("실제 웨어러블 기기의 상황을 가정하여, 업로드된 신호를 청크 단위로 잘라 서빙 API(/predict)에 순차 전송")

# ---------------------------------------------------------------------------
# 사이드바 설정
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("설정")
    api_url = st.text_input("서빙 API 주소", value=DEFAULT_API_URL)
    sampling_rate = st.number_input("신호 sampling rate (Hz)", value=360, min_value=1, max_value=2000)
    chunk_sec = st.slider("청크 길이 (초)", min_value=1, max_value=30, value=10)
    device_id = st.text_input("device_id", value="demo-upload")

    if st.button("서버 상태 확인"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            r.raise_for_status()
            body = r.json()
            if body["model_loaded"]:
                st.success(f"연결됨 (model_version={body['model_version']})")
            else:
                st.warning("server on, but model is none (MODEL_CHECKPOINT_PATH 확인)")
        except requests.exceptions.RequestException as e:
            st.error(f"연결 실패: {e}")

# ---------------------------------------------------------------------------
# 파일 업로드 + 파싱
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("ECG 신호 파일 업로드 (.npy 또는 .csv, 단일 리드)", type=["npy", "csv"])


def load_signal(file) -> np.ndarray:
    if file.name.endswith(".npy"):
        arr = np.load(io.BytesIO(file.read()))
    else:
        df = pd.read_csv(file, header=None)
        arr = df.iloc[:, 0].to_numpy()  # 첫 컬럼만 사용 (단일 리드 가정)
    return np.asarray(arr, dtype=np.float64).flatten()


if uploaded is not None:
    signal = load_signal(uploaded)
    duration_total_sec = len(signal) / sampling_rate
    st.write(f"신호 길이: {len(signal):,} 샘플 (~{duration_total_sec:.1f}초 @ {sampling_rate}Hz)")

    if st.button("분석 시작", type="primary"):
        chunk_len = int(chunk_sec * sampling_rate)
        n_chunks = int(np.ceil(len(signal) / chunk_len))

        all_beats = []  # (global_sample_idx, predicted_class)
        alert_chunks = []
        progress = st.progress(0.0, text="청크 전송 중...")
        errors = []

        for i in range(n_chunks):
            start = i * chunk_len
            end = min(start + chunk_len, len(signal))
            chunk = signal[start:end]

            # 스키마 min_length=360 미만인 마지막 자투리 청크는 건너뜀
            if len(chunk) < 360:
                continue

            try:
                r = requests.post(
                    f"{api_url}/predict",
                    json={
                        "device_id": device_id,
                        "sampling_rate": int(sampling_rate),
                        "chunk_start_time": "2026-01-01T00:00:00Z",
                        "signal": chunk.tolist(),
                    },
                    timeout=30,
                )
                r.raise_for_status()
                body = r.json()
            except requests.exceptions.RequestException as e:
                errors.append(f"청크 {i}: {e}")
                progress.progress((i + 1) / n_chunks)
                continue

            if body["alert"]:
                alert_chunks.append(i)
            for beat in body["beats"]:
                global_idx = start + beat["r_peak_sample"]
                all_beats.append((global_idx, beat["predicted_class"]))

            progress.progress((i + 1) / n_chunks, text=f"청크 {i + 1}/{n_chunks} 처리 중...")

        progress.empty()

        if errors:
            st.error(f"{len(errors)}개 청크에서 오류 발생 (첫 번째: {errors[0]})")

        if not all_beats:
            st.warning("검출된 beat가 없습니다. 신호/sampling rate를 확인하세요.")
        else:
            # --- 요약 ---
            classes = [c for _, c in all_beats]
            counts = pd.Series(classes).value_counts().reindex(["N", "S", "V", "F"], fill_value=0)

            col1, col2, col3 = st.columns(3)
            col1.metric("총 검출 beat 수", len(all_beats))
            col2.metric("알림 발생 청크 수", f"{len(alert_chunks)} / {n_chunks}")
            col3.metric("이상 소견(S/V) 비율", f"{(counts['S'] + counts['V']) / len(all_beats) * 100:.1f}%")

            st.subheader("클래스별 beat 분포")
            st.bar_chart(counts)

            # --- 신호 + R-peak 오버레이 ---
            st.subheader("신호 파형과 검출된 R-peak")
            max_plot_samples = 20 * sampling_rate  # 데모 렌더링 성능을 위해 앞 20초만 표시
            plot_signal = signal[: int(max_plot_samples)]
            plot_df = pd.DataFrame({"signal": plot_signal})
            st.line_chart(plot_df, height=250)
            st.caption(
                "표시 범위: 앞 20초 렌더링(전체 신호 대비 성능 목적). "
                "클래스별 beat 개수 = 위 요약 전체 업로드 신호 기준"
            )

            with st.expander("beat별 상세 결과 (표)"):
                detail_df = pd.DataFrame(all_beats, columns=["sample_index", "predicted_class"])
                detail_df["time_sec"] = detail_df["sample_index"] / sampling_rate
                st.dataframe(detail_df, use_container_width=True)
