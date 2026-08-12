import wfdb
import requests
import config  # serving/config.py 말고, 학습용 config.py가 있는 프로젝트 루트에서 실행

# record 100은 N(정상) beat가 절대다수인 레코드라, 예측도 N 위주로 나와야 정상
record = wfdb.rdrecord("/home/qortjsdn/projects/ecg_project/mit-bih-arrhythmia-database-1.0.0/100")
sig = record.p_signal[:, 0]          # 첫 채널
chunk = sig[:3600].tolist()          # 360Hz * 10초

resp = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "device_id": "sanity-check-100",
        "sampling_rate": 360,
        "chunk_start_time": "2026-08-08T00:00:00Z",
        "signal": chunk,
    },
)
result = resp.json()
print("beats detected:", result["num_beats_detected"])
from collections import Counter
print(Counter(b["predicted_class"] for b in result["beats"]))