# HyperSIGMA Classification

Hyperspectral image classification using HyperSIGMA foundation model.

## Quick Start

```bash
# Run on Indian Pines with 10 samples per class
python train.py --dataset indian_pines --samples_per_class 10 --epochs 100

# Few-shot classification (5 samples per class)
python train.py --dataset indian_pines --samples_per_class 5

# Run on Pavia University
python train.py --dataset pavia_university --samples_per_class 10

# Specify number of runs for statistical validation
python train.py --dataset indian_pines --samples_per_class 10 --num_runs 10
```

## Datasets

| Dataset | Size | Bands | Classes |
|---------|------|-------|---------|
| Indian Pines | 145x145 | 200 | 16 |
| Pavia University | 610x340 | 103 | 9 |
| Houston | 349x1905 | 144 | 15 |

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --dataset | indian_pines | Dataset name |
| --samples_per_class | 10 | Training samples per class |
| --epochs | 100 | Number of training epochs |
| --batch_size | 64 | Training batch size |
| --lr | 6e-5 | Learning rate |
| --img_size | 33 | Input patch size |
| --pca_components | 30 | PCA components |
| --model_size | base | Model size (base/large/huge) |
| --num_runs | 10 | Number of experiment runs |

## Data Format

Expected data files in `data/classification/`:

**Indian Pines:**
- `Indian_pines_corrected.mat` - HSI data (key: `data` or `indian_pines_corrected`)
- `Indian_pines_gt.mat` - Ground truth (key: `groundT` or `indian_pines_gt`)

**Pavia University:**
- `PaviaU.mat` - HSI data (key: `paviaU`)
- `PaviaU_gt.mat` - Ground truth (key: `paviaU_gt`)

**Houston:**
- `Houston.mat` - HSI data
- `Houston_gt.mat` - Ground truth

## Output

Results are saved as JSON files containing:
- Overall Accuracy (OA)
- Average Accuracy (AA)
- Kappa coefficient
- Per-class accuracy
- Standard deviation across runs
