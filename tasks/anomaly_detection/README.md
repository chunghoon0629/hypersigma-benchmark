# HyperSIGMA Anomaly Detection

Hyperspectral anomaly detection using HyperSIGMA foundation model.

## Quick Start

```bash
# Run on Pavia dataset with spectral-spatial mode
python train.py --dataset pavia --mode ss --epochs 10

# Run spatial-only mode
python train.py --dataset pavia --mode sa --epochs 10

# Run on CRI dataset
python train.py --dataset cri --mode ss --epochs 10
```

## Modes

- **sa**: Spatial-only mode using SpatViT encoder
- **ss**: Spectral-Spatial mode using both SpatViT and SpecViT encoders

## Expected Results

| Dataset | Mode | AUC-ROC |
|---------|------|---------|
| Pavia   | sa   | ~86.17% |
| Pavia   | ss   | ~84.45% |

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --dataset | pavia | Dataset name (pavia, cri) |
| --mode | ss | Model mode (sa, ss) |
| --epochs | 10 | Number of training epochs |
| --batch_size | 1 | Training batch size |
| --lr | 6e-5 | Learning rate |
| --input_size | 32 32 | Input patch size |
| --overlap_size | 16 | Overlap between patches |

## Data Format

Expected data files in `data/anomaly_detection/`:
- `Pavia_150_150_102.mat` - Pavia HSI data
- `Pavia_coarse_det_map.mat` - Pavia coarse detection map
- `Nuance_Cri_400_400_46_1254.mat` - CRI dataset
- `Cri_coarse_det_map.mat` - CRI coarse detection map

MAT files should contain:
- `hsi`: Hyperspectral image (H x W x C)
- `hsi_gt`: Ground truth anomaly map (H x W)
