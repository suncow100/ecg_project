
# UI 서빙 전용으로 mlflow 패키지 하나만 든 이미지
FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==3.14.0

WORKDIR /mlflow

EXPOSE 5000

CMD ["mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db", "--host", "0.0.0.0", "--port", "5000"]
