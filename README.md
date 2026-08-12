# ECG Arrhythmia Classification Service
MIT-BIH Arrhythmia Database 기반 ECG 부정맥 분류 및 AI 서비스 구축 프로젝트
ECG(Electrocardiogram) 신호를 이용하여 심박동(beat) 단위의 부정맥을 분류하고, 학습된 딥러닝 모델을 실제 API 서비스 형태로 제공하는 End-to-End 의료 AI 프로젝트

1. AAMI 권고안에 따른 부정맥 클래스를 분류하는 딥러닝 모델 구축: 5가지 심장박동 클래스(N, S, V, F, Q)
2.  de Chazal의 제안에 따른 환자 독립적 분할(Inter-patient Split): 동일한 환자의 심전도 데이터가 훈련 셋과 테스트 셋에 동시에 포함되지 않도록 분리
     즉,  MIT-BIH dataset의 환자군을 DS1/DS2로 split하는 방식에 따라 학습시킨 모델 활용



### 순서
ECG 데이터 전처리 → 환자 단위 데이터 분할 → Noise Synthesis → 1D ResNet 학습 → MLflow 실험 관리 → FastAPI API 구축 → Docker 컨테이너화
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/30f7061b-e4ac-4ce1-8676-e6428ed4227f" />


## MIT-BIH Arrhythmia Database
PhysioNet의 MIT-BIH Arrhythmia Database를 사용했습니다.

Sampling Rate: 360 Hz
ECG Lead: Modified Lead II (MLII)
Annotation: WFDB .atr
Signal: ECG waveform
Annotation: R peak annotation 정보를 이용하여 heartbeat segmentation

Window: [-100, +150] samples
총 입력 길이:250 samples
Sampling rate = 360 Hz이므로 하나의 heartbeat 입력은 약 0.694초의 ECG waveform으로 구성됨

## Preprocessing Pipeline
Raw ECG -> ECG Cleaning -> WFDB Annotation -> R-peak Detection from Annotation -> Beat Segmentation -> Z-score Normalization -> Training Dataset

ECG signal의 noise를 감소시키기 위해 NeuroKit2의 ECG cleaning 기능을 사용하여 필터링
+
MIT-BIH에서 제공되는 WFDB annotation을 ground truth로 사용하여 beat segmentation을 수행

## Noise Synthesis
실제 ECG 환경에서 발생할 수 있는 noise에 대한 모델의 robustness를 고려하기 위해 MIT-BIH Noise Stress Test Database(NSTDB) 합성

##### noise 유형:
Baseline Wander (BW)
Muscle Artifact (MA)
Electrode Motion Artifact (EM)

Noise 합성 과정에서도 DS1/DS2의 데이터 분할 원칙을 유지하도록 구성
DS1 ECG → DS1 Noise
DS2 ECG → DS2 Noise

## Dataset Split
random split 방식에서의 data leakage 발생: 하나의 환자에게서 여러 개의 heartbeat가 추출되기 때문에 beat 단위로 무작위 분할할 경우 동일 환자의 ECG morphology가 train/test에 동시에 포함될 수 있음
->  patient/record-level split 적용

DS1(80%): Train / Validation
DS2(20%): Test

## AAMI Classification
MIT-BIH의 annotation symbol을 AAMI EC57 기준 class로 매핑

Class:	Description
N:	Normal beat
S:	Supraventricular ectopic beat
V:	Ventricular ectopic beat
F:	Fusion beat
Q:	Unknown / Paced / Other

#### 문제점: class imbalance가 심각한 데이터셋임.
특히 N class가 대부분의 데이터를 차지하는 반면 S, V, F, Q는 상대적으로 매우 적다는 문제가 있음
즉, 단순 accuracy만으로 모델 평가 불가하므로, Macro F1 Score를 주요 평가 지표로 사용함
또한 training 과정에서는 class weight를 적용하여 소수 클래스의 학습 영향력을 보정
+ cosine annealing, weigry decay등, 다양한 파라미터 변경/추가 하는 등의 과정은 mlflow에 기록되어있으나,
+ 최종적으로는 test ONLY dropout
python train.py --optimizer adam --weight-decay 0 --scheduler none --dropout 0.3 --run-name dropout_only
버전으로 사용하였음

###### 실험 결과와 학습 parameter를 관리하기 위해 MLflow를 사용하였으며,

기록된 정보는 다음과 같음
classes: N/S/V/F
epochs: 120
batch_size: 256
lr: 0.0005
optimizer: AdamW
weight_decay: 0.0001
scheduler: CosineAnnealingLR
dropout: 0.3

##### Metric
train_loss, val_loss, val_f1_macro, lr, test_macro_f1



## 결과
최종적으로는 다음과 같음

Test Macro F1
Evaluation: Macro F1
4-Class (N/S/V/F): 	0.4259
5-Class (N/S/V/F/Q): 	0.3407
결과를 보면 알 수 있듯이, 5클래스에 대해서는 거의 무의미한 수준이며, 사실상 4개 클래스에서도 비슷한 모습을 보임. 그러나 4개 클래스에서는
smote, class weight 등의 방식으로 보정할 수 있기 때문에 최대한 모델 학습에 F 클래스를 활용하는 방안으로 진행함.

[RESULT] best epoch: 38
[RESULT] test macro F1: 0.42714893637771284
              precision    recall  f1-score   support

           N       0.96      0.88      0.92     19009
           S       0.11      0.24      0.15       547
           V       0.53      0.80      0.64      1675
           F       0.00      0.00      0.00        55

    accuracy                           0.86     21286
   macro avg       0.40      0.48      0.43     21286
weighted avg       0.90      0.86      0.88     21286

Confusion Matrix
[[16762  1091  1088    68]
 [  346   134    67     0]
 [  281    41  1344     9]
 [   31     3    21     0]]


 ## FastAPI Service& Docker
이후, 학습된 모델을 실제 inference API로 사용할 수 있도록 FastAPI 기반 서버 구축
API는 ECG 데이터를 입력으로 받아 모델의 prediction을 반환하도록 구성하였고,
이를 통해 Python script에서 직접 모델을 실행하는 방식에서 벗어나 REST API 형태로 모델을 사용할 수 있음

또한, 실행 환경의 일관성을 확보하고 배포를 쉽게 하기 위해 Docker를 사용했는데, 구조는 Main branch와 같음

docker-compose up --build 로 실행하는 구조
