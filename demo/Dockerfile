# demo/Dockerfile
#
# 이 컨테이너엔 torch도, neurokit2도 없다 -- 서빙 API를 HTTP로만 호출하는
# 순수 클라이언트라서 그렇다. serving/Dockerfile과 나란히 놓고 보면 "demo는
# 서빙 로직을 전혀 모른다"는 관심사 분리가 이미지 크기 차이로도 드러난다.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
