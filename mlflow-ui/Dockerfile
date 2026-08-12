# mlflow-ui/Dockerfile
#
# 학습 이미지 안에도 mlflow가 있지만, tracking UI를 "항상 떠있는 서비스"로
# 쓰려고 학습 이미지를 통째로 띄우면 wfdb/sklearn 등 안 쓰는 무게가 같이
# 따라온다. UI 서빙 전용으로 mlflow 패키지 하나만 든 이미지를 별도로 둔다.
FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==3.14.0

WORKDIR /mlflow

EXPOSE 5000

CMD ["mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db", "--host", "0.0.0.0", "--port", "5000"]
