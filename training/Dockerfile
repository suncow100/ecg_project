# training/Dockerfile
#
# serving/Dockerfile과 대조되는 지점: 여기선 torch를 CPU 전용 인덱스로
# 따로 설치하지 않는다. 기본 PyPI 휠(CUDA 런타임 포함)을 그대로 쓴다 --
# 학습은 GPU(RTX 4050)를 써야 하니 CUDA가 빠지면 안 된다.
# 컨테이너 자체엔 NVIDIA 드라이버가 없어도 되지만, 호스트(WSL2)에
# nvidia-container-toolkit이 설치돼 있어야 `--gpus all`로 GPU를 넘겨줄 수 있다.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 학습은 실험할 때만 수동으로 도는 잡(job)이지, 서빙처럼 항상 떠있는
# 프로세스가 아니다. 그래서 EXPOSE/헬스체크가 없고, CMD도 고정 스크립트가
# 아니라 실행 시점에 --optimizer/--dropout 등 인자를 바꿔가며 ablation 하기
# 좋게 열어둔다 (train.py가 argparse로 이미 그렇게 설계되어 있음).
ENTRYPOINT ["python", "train.py"]
