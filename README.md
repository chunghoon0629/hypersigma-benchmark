# HyperSIGMA Benchmark

Reproducible benchmark suite for [HyperSIGMA](https://github.com/WHU-Sigma/HyperSIGMA) - Hyperspectral Intelligence Comprehension Foundation Model.

This repository provides clean, unified implementations for running HyperSIGMA on standard hyperspectral imaging benchmarks.

## Quick Start

### 1. Clone and Setup

```bash
git clone <this-repo>
cd hypersigma-benchmark

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pretrained Weights

```bash
bash scripts/download_weights.sh
```

Or manually download from [HuggingFace](https://huggingface.co/WHU-Sigma/HyperSIGMA):
- `spat-vit-base-ultra-checkpoint-1599.pth`
- `spec-vit-base-ultra-checkpoint-1599.pth`

Place weights in `pretrained/` directory.

### 3. Prepare Data

Place datasets in the `data/` directory following this structure:

```
data/
├── anomaly_detection/
│   ├── Pavia_150_150_102.mat
│   ├── Pavia_coarse_det_map.mat
│   └── ...
├── classification/
│   ├── Indian_pines_corrected.mat
│   ├── Indian_pines_gt.mat
│   ├── PaviaU.mat
│   ├── PaviaU_gt.mat
│   └── ...
└── ...
```

### 4. Run Benchmarks

```bash
# Anomaly Detection
python tasks/anomaly_detection/train.py --dataset pavia --mode ss --epochs 10

# Classification
python tasks/classification/train.py --dataset indian_pines --samples_per_class 10

# Or use unified runner
python scripts/run_benchmark.py --task anomaly --dataset pavia --mode ss
python scripts/run_benchmark.py --task classification --dataset indian_pines --samples 10
```

## Available Tasks

| Task | Status | Datasets |
|------|--------|----------|
| Anomaly Detection | ✅ Ready | Pavia, CRI |
| Classification | ✅ Ready | Indian Pines, PaviaU, Houston |
| Change Detection | 🔲 Planned | Bay Area |
| Denoising | 🔲 Planned | WDC |
| Unmixing | 🔲 Planned | Urban4 |

## Results

### Anomaly Detection (Pavia)

| Mode | AUC-ROC |
|------|---------|
| sa (spatial-only) | ~86.17% |
| ss (spectral-spatial) | ~84.45% |

### Classification

Results with 10 samples per class:

| Dataset | OA | AA | Kappa |
|---------|----|----|-------|
| Indian Pines | TBD | TBD | TBD |
| PaviaU | TBD | TBD | TBD |

## Repository Structure

```
hypersigma-benchmark/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── RESULTS.md                   # Auto-generated results
│
├── configs/
│   └── paths.yaml               # Path configuration
│
├── scripts/
│   ├── download_weights.sh      # Download pretrained weights
│   ├── run_benchmark.py         # Unified benchmark runner
│   └── collect_results.py       # Aggregate results
│
├── pretrained/                  # Weights (gitignored)
│   ├── spat-vit-base-ultra-checkpoint-1599.pth
│   └── spec-vit-base-ultra-checkpoint-1599.pth
│
├── hypersigma/                  # Core model code
│   ├── models/
│   │   ├── spat_vit.py          # Spatial ViT encoder
│   │   ├── spec_vit.py          # Spectral ViT encoder
│   │   └── task_heads.py        # Task-specific heads
│   ├── utils/
│   │   ├── checkpoint.py        # Weight loading
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── data_utils.py        # Data utilities
│   └── mmcv_custom/             # Optimizer constructors
│
├── tasks/
│   ├── anomaly_detection/
│   │   ├── train.py
│   │   ├── config.py
│   │   └── README.md
│   ├── classification/
│   │   ├── train.py
│   │   ├── config.py
│   │   └── README.md
│   └── ...
│
├── data/                        # Datasets (gitignored)
└── results/                     # Outputs (gitignored)
```

## Model Architecture

HyperSIGMA uses a dual-encoder architecture:

- **SpatViT**: Spatial Vision Transformer for spatial feature extraction
- **SpecViT**: Spectral Vision Transformer for spectral feature extraction

Both encoders are pretrained on large-scale hyperspectral data and can be used independently or together.

## Citation

If you use this benchmark, please cite the original HyperSIGMA paper:

```bibtex
@article{hypersigma2024,
  title={HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This benchmark code is released under MIT License. The HyperSIGMA model weights are subject to their original license.

## Acknowledgments

- Original HyperSIGMA: [WHU-Sigma/HyperSIGMA](https://github.com/WHU-Sigma/HyperSIGMA)
- This benchmark is created for fair comparison with [SpectralFM](https://github.com/...)
