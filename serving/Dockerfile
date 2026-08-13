FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# healthcheck는 "프로세스가 응답하는가"만 보고, "모델이 실제로 로드됐는가"는
# 별개로 /health 응답 바디를 사람이/모니터링 시스템이 확인해야 함.
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
