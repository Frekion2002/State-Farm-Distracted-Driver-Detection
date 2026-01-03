# State-Farm-Distracted-Driver-Detection
본 프로젝트는 대시보드 카메라로 촬영된 운전자의 이미지를 분석하여 10가지 운전 행동(정상 운전, 휴대전화 사용, 음료 섭취 등)을 정확하게 분류하는 딥러닝 모델을 구축하는 것을 목표로 합니다. 


특히 데이터셋의 특성을 고려하여 Multi-Stream Architecture와 KNN Post-processing을 도입해 일반화 성능을 극대화했습니다.


# 1. 프로젝트 개요
- 목표: 정적인 이미지 한 장으로 운전자의 상태를 10개의 클래스로 분류
- 핵심 도전 과제:
  - 운전자 기준의 Train/Test 분리를 통한 정부 누수(Data Leakage)방지
  - 차량 내부 배경 노이즈 제거 및 핵심 행동 영역(머리, 손) 집중
  - 모델의 과적합(Overfitting) 방지 및 일반화 성능 향상

 
# 2. 주요 파이프라인 (System Architecture)
<img width="2816" height="1536" alt="Gemini_Generated_Image_b5fxxjb5fxxjb5fx" src="https://github.com/user-attachments/assets/63eb4647-f57b-4e15-9e4c-855d06384423" />


## 2.1 ROI(Region of Interest) 추출 (YOLOv8-pose)
행동 판별에 핵심적인 정보를 얻기 위해 yolov8n-pose 모델을 활용하여 핵심 부위를 Crop 했습니다.
- Head ROI: 운전자의 시선 및 고개 방향 파악
- Hand ROI: 핸들 조작, 문자 전송, 통화 등 손의 위치와 방향 분석
- 논리적 근거: 전체 이미지에서 배경 정보가 차지하는 비중이 커 모델이 핵심 특징에 집중하지 못하는 한계를 극복하기 위함입니다.


<img width="320" height="238" alt="yolopose" src="https://github.com/user-attachments/assets/4de20fd6-1c33-48f0-a068-64f83387a218" />


## 2.2 모델 아키텍처 (Multi-Stream ResNet)
추출된 전역(Global) 및 지역(Local) 특징을 결합하는 구조를 설계했습니다.
- Global Stream: ResNet50 (전체 이미지 특징 추출)
- Local Stream: ResNet18 (머리 및 손 영역의 미세 특징 추출)
- Feature Fusion: 각 스트림에서 추출된 특징 벡터를 Concatenate 한 후 최종 분류를 수행합니다.


## 2.3 과적합 방지 및 최적화 전략
- Driver-Wise Split: 동일 운전자의 데이터가 학습과 검증에 섞이지 않도록 GroupKFold를 사용하여 정보 누수를 원천 차단하였습니다.
- Head Simplification & High Dropout: 모델이 복잡해질수록 노이즈까지 학습하는 것을 방지하기 위해 분류기(Head)를 단순화하고 Dropout 확률을 0.8까지 높여 일반화 성능을 높였습니다.
- Data Augmentation: Color Jittering, Random Affine 등을 적용했으며, 방향이 중요한 클래스 특성상 Horizontal Flip은 제외했습니다.


# 3. 후처리: KNN Soft Voting (Smoothing)
모델이 추출한 Feature 공간에서 이미지 간 형태적 유사도를 계산하여 예측값을 보정했습니다.
- 작동 원리: 연속된 프레임으로 촬영된 데이터셋 특성상, 시간적으로 가까운 프레임은 유사한 Feature 벡터를 가집니다.
- 효과: 특정 프레임에서 모데르이 확신도가 낮더라도 유사한 이웃 프레임의 예측값을 참고하여 자연스러운 보정이 가능해졌습니다.


# 4. 실험 결과
## Augmentation Ablation Study
| Data Augmentation | Validation Acc | Validation Loss |
| :--- | :---: | :---: |
| Naive (Base) | 0.8986  | 0.4328  |
| **Color Jittering** | **0.9494**  | **0.2358**  |
| Random Affine | 0.9239  | 0.3871  |
| Horizontal Flip | 0.8842  | 0.4701  |
| Random Erasing | 0.9142  | 0.3334  |
| Multi-crop | 0.9245  | 0.4281  |

## Train/Val Loss graph
<img width="266" height="212" alt="learning_curve_before" src="https://github.com/user-attachments/assets/4fe68713-596c-4982-9b1b-7f2e6800f4ed" />


- 기존 ResNet의 Head 및 Dropout 0.5


<img width="329" height="259" alt="learning_curve_after" src="https://github.com/user-attachments/assets/52de90c8-a3f7-40fa-b25e-a222d85ec526" />


- Head 단순화 및 Dropout 0.8


- 결론: 단순한 모델 구조와 강한 Dropout 적용이 복잡한 모델보다 더 나은 일반화 성능을 보임을 확인했습니다.


# 5. 추가 실험 성공/실패 사례 분석
<img width="266" height="230" alt="confusion_matric" src="https://github.com/user-attachments/assets/9929b72f-c98b-42c8-a3d2-b8e33d87cdad" />


- 학습 후 confusion matrix 결과 c0(안전 운전) 클래스와 c9(동승자와 대화) 클래스에 대해 오분류가 심하다는 것을 확인했습니다.
- 이를 해결하기 위해 c0와 c9 클래스만 분류하는 binary classifier(ResNet18)를 추가했지만 의미 있는 성능 개선을 이끌어낼 수 없었습니다.
- 분석 결과 이미지 데이터에는 소리 데이터가 존재하지 않으므로 정면을 바라본 채 동승자와 대화하는 경우 이를 정확히 분류하는 것은 거의 불가능에 가깝다는 결론을 내렸습니다.
  - 이 한계를 극복하기 위해 **"향후 오디오 데이터 결합이나 시계열 분석이 필요하다."**


# 6. 실행 방법
**1. ROI 추출**: 머리/손 BBox 정보를 담은 `yolo_pose_bboxes.json' 생성
```bash
python YOLO.py
```
**2. 모델 학습**: GropKFold 기반 5-Fold 학습 진행
```bash
python train_best.py
```
**3. 후처리 및 추론**: KNN Smoothing을 적용한 최종 Submission 생성
```bash
python knn_postprocess_best.py
```
