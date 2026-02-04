# HyperSIGMA Benchmark Results

## Summary

This document contains benchmark results for HyperSIGMA on 6 hyperspectral imaging tasks. All experiments use the pretrained HyperSIGMA backbone with SpatViT and SpecViT encoders.

**Date**: 2026-02-03

---

## 1. Classification (50-shot)

| Dataset | OA (%) | Kappa | AA (%) | Runs |
|---------|--------|-------|--------|------|
| **IndianPines** | **93.22 ± 0.99** | 0.922 ± 0.011 | 96.22 ± 0.51 | 10 |
| **PaviaU** | **96.82 ± 1.03** | 0.956 ± 0.014 | 97.36 ± 0.74 | 10 |

### Configuration
- Epochs: 200
- Batch size: 64
- Learning rate: 1e-4
- Patch size: 2
- Image size: 33
- PCA components: 30

---

## 2. Change Detection

| Dataset | OA (%) | Kappa | F1 (%) | Precision (%) | Recall (%) | Runs |
|---------|--------|-------|--------|---------------|------------|------|
| **Hermiston** | **98.48 ± 0.19** | 0.935 ± 0.008 | 94.37 ± 0.67 | 89.80 ± 1.37 | 99.45 ± 0.23 | 10 |
| **BayArea** | **98.56 ± 0.13** | 0.971 ± 0.003 | 98.65 ± 0.12 | 98.76 ± 0.39 | 98.56 ± 0.37 | 10 |

### Configuration
- Mode: ss (spectral-spatial)
- Epochs: 10
- Batch size: 32
- Learning rate: 6e-5
- Train samples: 500
- Patch size: 15

---

## 3. Target Detection (SanDiego)

| Mode | AUC-ROC (%) | AUC-PR (%) | F1 (%) | Runs |
|------|-------------|------------|--------|------|
| **SS (spectral-spatial)** | **99.85 ± 0.09** | 83.70 ± 9.78 | 79.23 ± 7.80 | 10 |
| **SA (spatial only)** | **99.84 ± 0.09** | 81.32 ± 10.60 | 79.12 ± 7.91 | 10 |

### Configuration
- Epochs: 50
- Batch size: 32
- Learning rate: 1e-4
- Patch size: 7
- Train ratio: 0.5

---

## 4. Denoising (IndianPines)

| Metric | Value | Runs |
|--------|-------|------|
| **PSNR** | **23.57 ± 0.52** | 5 |
| **SSIM** | **0.684 ± 0.029** | 5 |
| **SAM** | 14.80 ± 0.67 | 5 |

### Configuration
- Epochs: 50
- Batch size: 8
- Learning rate: 1e-4
- Noise sigma range: [10, 70]

---

## 5. Anomaly Detection (Pavia)

| Metric | Value | Runs |
|--------|-------|------|
| **AUC-ROC** | **83.71 ± 0.00** | 10 |

### Configuration
- Mode: ss (spectral-spatial)
- Epochs: 10
- Batch size: 1
- Learning rate: 6e-5
- Input size: 32x32
- Overlap size: 16

---

## 6. Unmixing (Urban4)

| Mode | Abundance RMSE | Reconstruction SAM | Runs |
|------|----------------|-------------------|------|
| **SS (spectral-spatial)** | **0.0250** | 3.92 | 1* |
| **SA (spatial only)** | **0.0253** | 3.89 | 1* |

*Note: Unmixing experiments did not fully complete. Results shown are from single runs.

### Configuration
- Epochs: 50
- Batch size: 64
- Learning rate: 1e-4
- Patch size: 7
- Train ratio: 0.8

---

## Summary Table

| Task | Dataset | Primary Metric | HyperSIGMA |
|------|---------|----------------|------------|
| Classification | IndianPines (50-shot) | OA (%) | **93.22 ± 0.99** |
| Classification | PaviaU (50-shot) | OA (%) | **96.82 ± 1.03** |
| Change Detection | Hermiston | OA (%) | **98.48 ± 0.19** |
| Change Detection | BayArea | OA (%) | **98.56 ± 0.13** |
| Target Detection | SanDiego (SS) | AUC-ROC (%) | **99.85 ± 0.09** |
| Target Detection | SanDiego (SA) | AUC-ROC (%) | **99.84 ± 0.09** |
| Denoising | IndianPines | PSNR (dB) | **23.57 ± 0.52** |
| Anomaly Detection | Pavia | AUC-ROC (%) | **83.71** |
| Unmixing | Urban4 (SS) | RMSE | **0.0250** |
| Unmixing | Urban4 (SA) | RMSE | **0.0253** |

---

## Notes

1. **Classification**: Both IndianPines and PaviaU show strong performance with 50 samples per class.

2. **Change Detection**: Excellent results on both datasets with OA > 98%.

3. **Target Detection**: Very high AUC-ROC (>99.8%) on SanDiego dataset. Both spatial-only (SA) and spectral-spatial (SS) modes perform similarly.

4. **Denoising**: Moderate PSNR improvement with the foundation model.

5. **Anomaly Detection**: Results show 83.71% AUC-ROC. Note that only 10 epochs were used.

6. **Unmixing**: Low abundance RMSE (~0.025) indicates good performance. Full multi-seed runs were incomplete.

---

## Experimental Details

- **Hardware**: NVIDIA B200 GPUs (8 available)
- **Seeds used**: 42, 123, 456, 789, 2024, 1234, 5678, 9999, 1111, 7777
- **Pretrained weights**:
  - SpatViT: `spat-vit-base-ultra-checkpoint-1599.pth`
  - SpecViT: `spec-vit-base-ultra-checkpoint-1599.pth`
