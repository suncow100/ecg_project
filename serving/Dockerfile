# serving/Dockerfile
#
# lightweight 서빙 이미지: torch도 CPU-only 휠을 따로 지정해서 설치한다.
# 기본 `pip install torch`는 CUDA 런타임까지 같이 받아와서 수 GB가 붙는데,
# 이 컨테이너는 추론만 하고 GPU도 안 쓰므로 그 용량이 전부 낭비다.
# (학습 이미지는 반대로 GPU가 필요하므로 이 CPU 전용 인덱스를 쓰지 않는다.)
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# /health가 model_loaded:false를 반환해도 HTTP 200이라 healthcheck 자체는 통과한다.
# healthcheck는 "프로세스가 응답하는가"만 보고, "모델이 실제로 로드됐는가"는
# 별개로 /health 응답 바디를 사람이/모니터링 시스템이 확인해야 한다.
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
