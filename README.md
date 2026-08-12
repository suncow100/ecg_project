## ECG Arrhythmia Classification Service
MIT-BIH Arrhythmia Database 기반 ECG 부정맥 분류 및 AI 서비스 구축 프로젝트

ECG(Electrocardiogram) 신호를 이용하여 심박동(beat) 단위의 부정맥을 분류하고, 학습된 딥러닝 모델을 실제 API 서비스 형태로 제공하는 End-to-End 의료 AI 프로젝트입니다.

본 프로젝트에서는 단순히 딥러닝 모델을 학습하는 것에 그치지 않고,

ECG 데이터 전처리 → 환자 단위 데이터 분할 → Noise Synthesis → 1D ResNet 학습 → MLflow 실험 관리 → FastAPI API 구축 → Docker 컨테이너화

까지 전체 머신러닝 파이프라인을 직접 구축했습니다.
