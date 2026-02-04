# HAD (Hyperspectral Anomaly Detection) - Pavia 데이터셋

## 1차 검증: 논문 기반

### 논문 요약

HyperSIGMA 논문에서 HAD 태스크는 **Segmentation 기반 Fine-tuning** 프로토콜을 따른다.

#### 데이터 생성 (Pseudo-label)
- **알고리즘**: RX (Reed-Xiaoli) detector
- **타겟 라벨링**: 상위 **0.15%** 픽셀을 이상치(Anomaly)로 설정
- **배경 라벨링**: (논문에 명시적 언급 없음)

#### 입력 구성
- **패치 크기**: 32×32
- **패치 중복**: 16 픽셀

#### 학습 설정
| 항목 | 값 |
|------|-----|
| 에포크 | 10 |
| 배치 사이즈 | 1 |
| 학습률 | 6e-5 |
| 옵티마이저 | AdamW |
| 스펙트럼 토큰 수 | 100 |
| 손실 함수 | Cross-Entropy |

#### 인코더 설정
- 사전학습 파라미터 로드 후 전체 모델 Fine-tuning
- Spatial patch embedding: 채널 수가 다르므로 랜덤 초기화 후 재학습
- Positional embedding: 입력 크기에 맞게 보간(Interpolation)

#### 기타
- 반복 실행 횟수: 명시 없음 (단일 실험 추정)


---

## 2차 검증: 원저자 코드 기반

**원저자 코드 경로**: `~/gsjo/paper/HyperSIGMA/HyperspectralDetection/Anomaly_Detection/`

### 원저자 코드 분석 결과

#### 데이터 전처리
| 항목 | 원저자 코드 | 비고 |
|------|-------------|------|
| Pseudo-label 알고리즘 | RX (Mahalanobis 거리) | `EmpiricalCovariance` 사용 |
| 타겟 라벨링 | 상위 0.15% | `all_idxs[-int(0.0015 * r * c):]` |
| 배경 라벨링 | 하위 30% | `all_idxs[:int(0.3 * r * c)]` |
| 정규화 | StandardScaler | Z-score 정규화 |

#### 입력 설정
| 항목 | 원저자 코드 |
|------|-------------|
| 패치 크기 | 32×32 |
| 패치 중복 | 16 |
| Step Size | 16 (= 32 - 16) |

#### 학습 설정
| 항목 | 원저자 코드 |
|------|-------------|
| 에포크 | 10 |
| 배치 사이즈 | 1 |
| 학습률 | 6e-5 |
| Weight Decay | 5e-4 |
| 옵티마이저 | AdamW (betas=0.9, 0.999) |
| Layer Decay Rate | 0.9 |
| Num Layers | 12 |
| Scheduler | CosineAnnealingLR (eta_min=0) |
| 손실 함수 | CrossEntropyLoss (ignore_index=255) |

#### 모델 아키텍처
| 항목 | 원저자 코드 |
|------|-------------|
| SpatViT embed_dim | 768 |
| SpatViT depth | 12 |
| SpatViT num_heads | 12 |
| SpatViT out_indices | [3, 5, 7, 11] |
| Drop Path Rate | 0.1 |
| 스펙트럼 토큰 수 | 100 |
| Deformable Attention n_points | 8 |

#### 예측 방식
- **Segmentation 방식**: 패치 전체 픽셀에 대한 dense prediction
- 출력 형태: (B, 2, H, W) — 2클래스 (배경/이상치)
- 중복 영역 처리: 후반부 패치의 중복 절반만 사용


---

## 검증 결과 종합

### i) 논문에 있는데 다른 것

**없음** — Pavia 기준 모두 일치

### ii) 논문에 언급되지 않은 디테일 (원저자 코드로 확인)

| 항목 | 원저자 코드 | 현재 구현 | 일치 |
|------|-------------|-----------|------|
| Weight Decay | 5e-4 | 5e-4 | ✅ |
| Layer Decay Rate | 0.9 | 0.9 | ✅ |
| Num Layers | 12 | 12 | ✅ |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR | ✅ |
| 정규화 방식 | StandardScaler | StandardScaler | ✅ |
| ignore_label | 255 | 255 | ✅ |
| 배경 라벨링 | 하위 30% | 하위 30% | ✅ |
| SpatViT (dim/depth/heads) | 768/12/12 | 768/12/12 | ✅ |
| out_indices | [3,5,7,11] | [3,5,7,11] | ✅ |
| Drop Path Rate | 0.1 | 0.1 | ✅ |
| 예측 방식 | Segmentation | Segmentation | ✅ |

### iii) 논문과 일치하는 것

| 항목 | 논문 | 원저자 코드 | 현재 구현 | 일치 |
|------|------|-------------|-----------|------|
| Pseudo-label 알고리즘 | RX | RX (Mahalanobis) | RX (Mahalanobis) | ✅ |
| 타겟 라벨링 | 상위 0.15% | 상위 0.15% | 상위 0.15% | ✅ |
| 패치 크기 | 32×32 | 32×32 | 32×32 | ✅ |
| 패치 중복 | 16px | 16px | 16px | ✅ |
| 배치 사이즈 | 1 | 1 | 1 | ✅ |
| 에포크 | 10 | 10 | 10 | ✅ |
| 학습률 | 6e-5 | 6e-5 | 6e-5 | ✅ |
| 옵티마이저 | AdamW | AdamW | AdamW | ✅ |
| 손실 함수 | Cross-Entropy | CrossEntropyLoss | CrossEntropyLoss | ✅ |
| 스펙트럼 토큰 수 | 100 | 100 | 100 | ✅ |


---

## 최종 결론

| 구분 | 개수 | 상태 |
|------|------|------|
| **논문과 일치** | 10개 | ✅ |
| **논문과 불일치** | 0개 | - |
| **논문 미언급 (원저자와 일치)** | 11개 | ✅ |

**HAD Pavia: 논문 / 원저자 코드 / 현재 구현 모두 정합성 확인 완료 ✅**
