# 검증 프로세스

## 배경 및 목적

초분광 파운데이션 모델 HyperSIGMA의 평가방식을 AI로 작성한 clean 버전.
논문 및 원저자와의 정합성을 따져야 함.

지원하는 태스크 및 벤치마크 데이터셋 별로 검증을 진행한다.


## 검증 방법론

### 1차 검증: 논문 기반

벤치마크 평가와 관련한 논문 원문 및 요약 정보를 기반으로 검증한다.

**확인 항목**:
- 데이터 전처리 방식 (Pseudo-label 생성, 정규화 등)
- 입력 구성 (패치 크기, 중복 등)
- 학습 설정 (에포크, 배치사이즈, 학습률, 옵티마이저, 손실함수)
- 모델 설정 (스펙트럼 토큰 수, 인코더 설정)


### 2차 검증: 원저자 코드 기반

원저자의 리포(`~/gsjo/paper/HyperSIGMA`)를 기반으로 2차 검증한다.

**확인 항목**:
- 논문에 명시되지 않은 세부 구현 사항
- Weight decay, Layer decay 등 옵티마이저 세부 설정
- 모델 아키텍처 세부 사항 (embed_dim, depth, num_heads 등)
- 예측 방식 (Segmentation vs Center pixel classification)


## 검증 결과 분류

검증 결과는 다음 3가지로 분류:

1. **논문에 있는데 다른 것**: 수정 필요
2. **논문에 언급되지 않은 디테일**: 원저자 코드로 확인
3. **논문과 일치하는 것**: 정합성 확인 완료


## 검증 결과 문서

태스크별 검증 결과는 `val_results/` 디렉토리에 작성:

| 태스크 | 데이터셋 | 문서 | 상태 |
|--------|----------|------|------|
| HAD (Anomaly Detection) | Pavia | [had_pavia.md](val_results/had_pavia.md) | ✅ 완료 |
| Change Detection | Hermiston / Bay Area | [change_detection.md](val_results/change_detection.md) | ⚠️ Layer Decay 미적용 |
