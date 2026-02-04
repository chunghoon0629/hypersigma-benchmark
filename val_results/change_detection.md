# Change Detection (변화 탐지) - Hermiston / Bay Area 데이터셋

## 1차 검증: 논문 기반

### 논문 요약

HyperSIGMA 논문에서 Change Detection 태스크는 **SST-Former 모델의 평가 프로토콜**을 따른다.

#### 데이터 생성
- **학습 데이터**: Changed 500개 + Unchanged 500개 무작위 샘플링
- **테스트 데이터**: 학습 패치와 중복 최소화하여 데이터 유출 방지

#### 입력 구성 (데이터셋별)

| 데이터셋 | 입력 패치 크기 | 임베딩 Patch Size |
|----------|----------------|-------------------|
| Hermiston | 5×5 | 1 |
| Bay Area | 15×15 | 2 |

#### 학습 설정
| 항목 | 값 |
|------|-----|
| 에포크 | 50 |
| 배치 사이즈 | 32 |
| 학습률 | 6e-5 |
| 스펙트럼 토큰 수 | 144 |
| 옵티마이저 | AdamW |

#### 인코더 설정
- 사전학습 파라미터 로드 후 전체 모델 Fine-tuning
- 채널 수/입력 크기가 맞지 않는 경우 임베딩 레이어 무작위 초기화
- Positional Embedding은 보간(Interpolation)하여 사용

#### 기타
- 반복 실행 횟수: 명시 없음 (단일 실험 추정)


---

## 2차 검증: 원저자 코드 기반

**원저자 코드 경로**: `~/gsjo/paper/HyperSIGMA/ChangeDetection/`

### 원저자 코드 분석 결과

#### 데이터 전처리
| 항목 | 원저자 코드 | 비고 |
|------|-------------|------|
| 정규화 | 밴드별 MinMax [0, 1] | T1, T2 통합 min/max |
| 패딩 | Mirror Padding | 경계 픽셀 대칭 반사 |
| 라벨 인코딩 | 0=Changed, 1=Unchanged | 모델 출력 기준 |

#### 학습 설정
| 항목 | 원저자 코드 |
|------|-------------|
| 에포크 | 50 |
| 배치 사이즈 | 32 |
| 학습률 | 6e-5 |
| Weight Decay | 0.05 |
| 옵티마이저 | AdamW (betas=0.9, 0.999) |
| Layer Decay Rate | 0.9 |
| Scheduler | CosineAnnealingLR (eta_min=0) |
| 손실 함수 | CrossEntropyLoss |

#### 평가 메트릭
- OA (Overall Accuracy)
- Kappa 계수
- F1 Score
- Precision
- Recall


---

## 검증 결과 종합

### i) 논문에 있는데 다른 것

**없음** — 논문 명시 항목 모두 일치

### ii) 논문에 언급되지 않은 디테일 (원저자 코드로 확인)

| 항목 | 원저자 코드 | 현재 구현 | 일치 |
|------|-------------|-----------|------|
| Weight Decay | 0.05 | 0.05 | ✅ |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR | ✅ |
| 손실 함수 | CrossEntropyLoss | CrossEntropyLoss | ✅ |
| 정규화 | 밴드별 MinMax [0,1] | 밴드별 MinMax [0,1] | ✅ |
| 패딩 | Mirror Padding | Mirror Padding | ✅ |
| 라벨 인코딩 | 0=Changed, 1=Unchanged | 0=Changed, 1=Unchanged | ✅ |
| 평가 메트릭 | OA, F1, Kappa 등 | OA, F1, Kappa 등 | ✅ |
| **Layer Decay** | **0.9** | **미적용** | ❌ |

### iii) 논문과 일치하는 것

| 항목 | 논문 | 원저자 코드 | 현재 구현 | 일치 |
|------|------|-------------|-----------|------|
| 학습 샘플 수 | 500 (각 클래스) | 500 | 500 | ✅ |
| Hermiston 패치 크기 | 5×5 | 5×5 | 5×5 | ✅ |
| Bay Area 패치 크기 | 15×15 | 15×15 | 15×15 | ✅ |
| Hermiston 임베딩 patch_size | 1 | 1 | 1 | ✅ |
| Bay Area 임베딩 patch_size | 2 | 2 | 2 | ✅ |
| 에포크 | 50 | 50 | 50 | ✅ |
| 배치 사이즈 | 32 | 32 | 32 | ✅ |
| 학습률 | 6e-5 | 6e-5 | 6e-5 | ✅ |
| 스펙트럼 토큰 수 | 144 | 144 | 144 | ✅ |
| 옵티마이저 | AdamW | AdamW | AdamW | ✅ |


---

## 최종 결론

| 구분 | 개수 | 상태 |
|------|------|------|
| **논문과 일치** | 10개 | ✅ |
| **논문과 불일치** | 0개 | - |
| **논문 미언급 (원저자와 일치)** | 7개 | ✅ |
| **논문 미언급 (원저자와 불일치)** | 1개 | ❌ |

### 수정 필요 사항

**Layer Decay 적용 필요**

현재 구현 (`tasks/change_detection/train.py`):
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
```

원저자 코드처럼 `LayerDecayOptimizerConstructor_ViT` 적용 필요:
- `layer_decay_rate=0.9`
- `num_layers=12`
