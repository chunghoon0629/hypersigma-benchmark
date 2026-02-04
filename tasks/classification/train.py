#!/usr/bin/env python
"""
Classification Training Script for HyperSIGMA Benchmark.

Usage:
    python train.py --dataset indian_pines --samples_per_class 10 --epochs 100
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.decomposition import PCA

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HYPERSIGMA_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
PROJECT_ROOT = os.path.join(HYPERSIGMA_ROOT, '..')
sys.path.insert(0, HYPERSIGMA_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from downstream_task_head.utils.result_manager import ResultManager, ClassificationMetrics

# Default data directory (relative to hypersigma-benchmark root)
DEFAULT_DATA_DIR = os.path.join(HYPERSIGMA_ROOT, 'data', 'classification')

from hypersigma.models.task_heads import SSClassificationHead, ClassificationHead
from hypersigma.utils.metrics import compute_classification_metrics
from hypersigma.utils.checkpoint import load_hypersigma_weights
from hypersigma.mmcv_custom import LayerDecayOptimizerConstructor_ViT

from mmengine.optim import build_optim_wrapper


def load_dataset(dataset: str, data_dir: str):
    """Load classification dataset."""
    dataset_lower = dataset.lower()

    if dataset_lower in ['indian_pines', 'indianpines']:
        data = sio.loadmat(os.path.join(data_dir, 'Indian_pines_corrected.mat'))
        gt = sio.loadmat(os.path.join(data_dir, 'Indian_pines_gt.mat'))
        # Handle different key names
        if 'data' in data:
            cube = data['data']
        else:
            cube = data['indian_pines_corrected']
        if 'groundT' in gt:
            ground_truth = gt['groundT']
        else:
            ground_truth = gt['indian_pines_gt']
    elif dataset_lower in ['pavia_university', 'paviau', 'pavia']:
        data = sio.loadmat(os.path.join(data_dir, 'paviaU.mat'))
        gt = sio.loadmat(os.path.join(data_dir, 'paviaU_gt.mat'))
        # Handle different key names
        if 'ori_data' in data:
            cube = data['ori_data']
        else:
            cube = data['paviaU']
        if 'map' in gt:
            ground_truth = gt['map']
        else:
            ground_truth = gt['paviaU_gt']
    elif dataset_lower == 'houston':
        data = sio.loadmat(os.path.join(data_dir, 'Houston.mat'))
        gt = sio.loadmat(os.path.join(data_dir, 'Houston_gt.mat'))
        if 'Houston' in data:
            cube = data['Houston']
        else:
            cube = data['data']
        if 'Houston_gt' in gt:
            ground_truth = gt['Houston_gt']
        else:
            ground_truth = gt['gt']
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return cube.astype(np.float32), ground_truth.astype(np.float32)


def normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] range."""
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def apply_pca(data: np.ndarray, num_components: int):
    """Apply PCA dimensionality reduction."""
    h, w, c = data.shape
    data_2d = data.reshape(-1, c)
    pca = PCA(n_components=num_components)
    data_pca = pca.fit_transform(data_2d)
    return data_pca.reshape(h, w, num_components), pca


def create_patches(data: np.ndarray, gt: np.ndarray, patch_size: int, remove_zero=False):
    """Create patches centered on each pixel."""
    h, w, c = data.shape
    margin = patch_size // 2

    # Pad data
    padded = np.pad(data, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')

    patches = []
    labels = []

    for i in range(h):
        for j in range(w):
            label = int(gt[i, j])
            if remove_zero and label == 0:
                continue
            patch = padded[i:i + patch_size, j:j + patch_size, :]
            patches.append(patch)
            labels.append(label)

    return np.array(patches), np.array(labels)


def split_data(labels: np.ndarray, num_classes: int, train_num: int, val_num: int, seed: int):
    """Split data into train/val/test."""
    np.random.seed(seed)

    train_idx, val_idx, test_idx = [], [], []

    for c in range(1, num_classes + 1):
        class_idx = np.where(labels == c)[0]
        np.random.shuffle(class_idx)

        n = len(class_idx)
        n_train = min(train_num, n // 3)
        n_val = min(val_num, (n - n_train) // 2)

        train_idx.extend(class_idx[:n_train])
        val_idx.extend(class_idx[n_train:n_train + n_val])
        test_idx.extend(class_idx[n_train + n_val:])

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


class HSIDataset(torch.utils.data.Dataset):
    """Dataset for hyperspectral image classification."""

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.FloatTensor(data.transpose(0, 3, 1, 2))
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())

    return np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="HyperSIGMA Classification")

    # Dataset
    parser.add_argument('--dataset', type=str, default='indian_pines',
                        choices=['indian_pines', 'indianpines', 'pavia_university', 'paviau', 'houston'])
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--gpus', type=str, default='0', help='GPU IDs to use (comma-separated)')

    # Model
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['base', 'large', 'huge'])
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=33)
    parser.add_argument('--pca_components', type=int, default=30)

    # Weights
    parser.add_argument('--spat_weights', type=str,
                        default='pretrained/spat-vit-base-ultra-checkpoint-1599.pth')
    parser.add_argument('--spec_weights', type=str,
                        default='pretrained/spec-vit-base-ultra-checkpoint-1599.pth')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--wd', type=float, default=0.05)

    # Data split
    parser.add_argument('--samples_per_class', type=int, default=10)
    parser.add_argument('--val_samples', type=int, default=5)

    # Experiment
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    # Output
    parser.add_argument('--output_dir', type=str, default='results/classification')

    args = parser.parse_args()

    # Set CUDA devices
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using GPUs: {args.gpus}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading {args.dataset}...")
    data, gt = load_dataset(args.dataset, args.data_dir)
    h, w, bands = data.shape
    num_classes = int(np.max(gt))
    print(f"Data shape: {data.shape}, Classes: {num_classes}")

    # Normalize
    data = normalize_data(data)

    # Apply PCA
    print(f"Applying PCA ({args.pca_components} components)...")
    data, pca = apply_pca(data, args.pca_components)

    # Create patches
    print(f"Creating patches (size {args.img_size})...")
    patches, labels = create_patches(data, gt, args.img_size, remove_zero=False)

    # Results storage
    all_results = []

    for run in range(args.num_runs):
        seed = args.seed + run
        print(f"\n===== Run {run + 1}/{args.num_runs} (seed={seed}) =====")

        # Split data
        gt_flat = gt.reshape(-1)
        train_idx, val_idx, test_idx = split_data(
            gt_flat, num_classes, args.samples_per_class, args.val_samples, seed
        )

        # Filter to labeled samples only
        labeled_mask = labels > 0
        labeled_indices = np.where(labeled_mask)[0]

        # Map to labeled-only indices
        train_idx_labeled = np.intersect1d(train_idx, labeled_indices)
        val_idx_labeled = np.intersect1d(val_idx, labeled_indices)
        test_idx_labeled = np.intersect1d(test_idx, labeled_indices)

        train_data = patches[train_idx_labeled]
        train_labels = labels[train_idx_labeled] - 1  # 0-indexed
        val_data = patches[val_idx_labeled]
        val_labels = labels[val_idx_labeled] - 1
        test_data = patches[test_idx_labeled]
        test_labels = labels[test_idx_labeled] - 1

        print(f"Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}")

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            HSIDataset(train_data, train_labels),
            batch_size=args.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            HSIDataset(val_data, val_labels),
            batch_size=args.batch_size, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            HSIDataset(test_data, test_labels),
            batch_size=args.batch_size, shuffle=False
        )

        # Create model
        model = SSClassificationHead(
            img_size=args.img_size,
            in_channels=args.pca_components,
            num_classes=num_classes,
            patch_size=args.patch_size,
            spat_weights=args.spat_weights,
            spec_weights=args.spec_weights,
            model_size=args.model_size,
        )
        # Use DataParallel if multiple GPUs
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        model = model.to(device)

        # Optimizer
        optim_wrapper = dict(
            optimizer=dict(type='AdamW', lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd),
            constructor='LayerDecayOptimizerConstructor_ViT',
            paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9)
        )
        optimizer = build_optim_wrapper(model, optim_wrapper)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer.optimizer, args.epochs, eta_min=0
        )
        criterion = nn.CrossEntropyLoss()

        # Training
        best_loss = float('inf')
        best_model_state = None

        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}: loss={train_loss:.4f}, acc={train_acc:.4f}")

            if train_loss < best_loss:
                best_loss = train_loss
                best_model_state = model.state_dict().copy()

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        # Evaluate
        test_pred, test_true = evaluate(model, test_loader, device)
        metrics_dict = compute_classification_metrics(test_pred, test_true, num_classes)

        print(f"OA: {metrics_dict['overall_accuracy']:.4f}, "
              f"AA: {metrics_dict['average_accuracy']:.4f}, "
              f"Kappa: {metrics_dict['kappa']:.4f}")

        all_results.append(metrics_dict)

    # Aggregate results
    oa_list = [r['overall_accuracy'] for r in all_results]
    aa_list = [r['average_accuracy'] for r in all_results]
    kappa_list = [r['kappa'] for r in all_results]

    print("\n===== Final Results =====")
    print(f"OA: {np.mean(oa_list)*100:.2f}% +/- {np.std(oa_list)*100:.2f}%")
    print(f"AA: {np.mean(aa_list)*100:.2f}% +/- {np.std(aa_list)*100:.2f}%")
    print(f"Kappa: {np.mean(kappa_list):.4f} +/- {np.std(kappa_list):.4f}")

    # Save results using ResultManager
    manager = ResultManager(
        task="classification",
        dataset=args.dataset,
        checkpoint_path=args.spat_weights,
        experiment_name="hypersigma_benchmark"
    )
    manager.set_config(experiment_config={
        'dataset': args.dataset,
        'data_dir': args.data_dir,
        'model_size': args.model_size,
        'samples_per_class': args.samples_per_class,
        'val_samples': args.val_samples,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'wd': args.wd,
        'img_size': args.img_size,
        'patch_size': args.patch_size,
        'pca_components': args.pca_components,
        'spat_weights': args.spat_weights,
        'spec_weights': args.spec_weights,
        'model': 'HyperSIGMA',
    })

    # Log mean metrics with seed indicating it's the final aggregated result
    manager.log_run(
        seed=args.seed,
        metrics=ClassificationMetrics(
            overall_accuracy=float(np.mean(oa_list)),
            kappa=float(np.mean(kappa_list)),
            average_accuracy=float(np.mean(aa_list)),
            F1_macro=float(np.mean([r.get('f1_macro', 0) for r in all_results])) if 'f1_macro' in all_results[0] else 0.0,
        ),
        extra={
            'num_runs': args.num_runs,
            'oa_std': float(np.std(oa_list)),
            'aa_std': float(np.std(aa_list)),
            'kappa_std': float(np.std(kappa_list)),
            'per_run_results': all_results,
        }
    )
    manager.try_auto_aggregate()

    print(f"\nResults saved via ResultManager")


if __name__ == '__main__':
    main()
