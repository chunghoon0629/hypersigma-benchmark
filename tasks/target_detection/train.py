"""
HyperSIGMA Target Detection Training Script.

Trains target detection models on hyperspectral images.
Target detection aims to find pixels matching a known target spectrum.
"""

import os
import sys
import argparse
import time
import random
from datetime import datetime

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc

# Add parent directories to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HYPERSIGMA_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
PROJECT_ROOT = os.path.join(HYPERSIGMA_ROOT, '..')
sys.path.insert(0, HYPERSIGMA_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from downstream_task_head.utils.result_manager import ResultManager, TargetDetectionMetrics

# Default data directory (relative to hypersigma-benchmark root)
DEFAULT_DATA_DIR = os.path.join(HYPERSIGMA_ROOT, 'data', 'target_detection')

from hypersigma.models.task_heads import TargetDetectionHead, SSTargetDetectionHead


def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AvgMeter:
    """Running average meter."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class TargetDetectionDataset(Data.Dataset):
    """Dataset for target detection with patch extraction."""

    def __init__(self, data, target_map, target_spectrum, patch_size=7, mode='train',
                 train_ratio=0.8, seed=42):
        """
        Args:
            data: HSI cube (H, W, C)
            target_map: Binary target map (H, W), 1=target, 0=background
            target_spectrum: Target spectrum to detect (C,)
            patch_size: Size of patches to extract
            mode: 'train' or 'test'
            train_ratio: Ratio of data for training
        """
        self.data = data.astype(np.float32)
        self.target_map = target_map
        self.target_spectrum = target_spectrum
        self.patch_size = patch_size
        self.half_size = patch_size // 2

        # Normalize data
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min() + 1e-8)

        self.H, self.W, self.C = self.data.shape

        # Get all valid positions (away from edges)
        valid_h = range(self.half_size, self.H - self.half_size)
        valid_w = range(self.half_size, self.W - self.half_size)

        # Separate target and background pixels
        target_positions = []
        background_positions = []

        for h in valid_h:
            for w in valid_w:
                if self.target_map[h, w] > 0:
                    target_positions.append((h, w))
                else:
                    background_positions.append((h, w))

        # Split train/test
        np.random.seed(seed)
        np.random.shuffle(target_positions)
        np.random.shuffle(background_positions)

        n_target_train = int(len(target_positions) * train_ratio)
        n_bg_train = int(len(background_positions) * train_ratio)

        if mode == 'train':
            # Balance training set
            train_targets = target_positions[:n_target_train]
            train_bg = background_positions[:n_bg_train]
            # Undersample background to balance
            n_samples = min(len(train_targets) * 5, len(train_bg))  # 1:5 ratio
            train_bg = train_bg[:n_samples]
            self.positions = train_targets + train_bg
        else:
            # Full test set for evaluation
            self.positions = target_positions[n_target_train:] + background_positions[n_bg_train:]

        np.random.shuffle(self.positions)
        print(f"{mode} set: {len(self.positions)} samples "
              f"({sum(1 for p in self.positions if self.target_map[p[0], p[1]] > 0)} targets)")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        h, w = self.positions[idx]

        # Extract patch
        patch = self.data[
            h - self.half_size:h + self.half_size + 1,
            w - self.half_size:w + self.half_size + 1,
            :
        ]

        # Label
        label = 1 if self.target_map[h, w] > 0 else 0

        # Convert to tensor [C, H, W]
        patch = torch.from_numpy(patch.transpose(2, 0, 1)).float()
        target_spec = torch.from_numpy(self.target_spectrum).float()
        label = torch.tensor(label, dtype=torch.float32)

        return patch, target_spec, label


class TargetDetectionFullImage(Data.Dataset):
    """Dataset for full image target detection (sliding window)."""

    def __init__(self, data, target_spectrum, patch_size=7):
        self.data = data.astype(np.float32)
        self.target_spectrum = target_spectrum
        self.patch_size = patch_size
        self.half_size = patch_size // 2

        # Normalize
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min() + 1e-8)

        self.H, self.W, self.C = self.data.shape

        # Generate all valid positions
        self.positions = [
            (h, w)
            for h in range(self.half_size, self.H - self.half_size)
            for w in range(self.half_size, self.W - self.half_size)
        ]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        h, w = self.positions[idx]
        patch = self.data[
            h - self.half_size:h + self.half_size + 1,
            w - self.half_size:w + self.half_size + 1,
            :
        ]
        patch = torch.from_numpy(patch.transpose(2, 0, 1)).float()
        target_spec = torch.from_numpy(self.target_spectrum).float()
        return patch, target_spec, h, w


def get_target_spectrum(data, target_map):
    """Extract mean target spectrum from labeled target pixels."""
    target_pixels = data[target_map > 0]
    target_spectrum = target_pixels.mean(axis=0)
    return target_spectrum.astype(np.float32)


def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    losses = AvgMeter()

    for patches, target_specs, labels in train_loader:
        patches = patches.cuda()
        target_specs = target_specs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(patches, target_specs)
        # Extract center pixel score from detection map [B, H, W]
        center_h = outputs.shape[1] // 2
        center_w = outputs.shape[2] // 2
        outputs = outputs[:, center_h, center_w]  # [B]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), patches.size(0))

    return losses.avg


def evaluate(model, test_loader, criterion):
    """Evaluate model on test set."""
    model.eval()
    losses = AvgMeter()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for patches, target_specs, labels in test_loader:
            patches = patches.cuda()
            target_specs = target_specs.cuda()
            labels = labels.cuda()

            outputs = model(patches, target_specs)
            # Extract center pixel score from detection map [B, H, W]
            center_h = outputs.shape[1] // 2
            center_w = outputs.shape[2] // 2
            outputs = outputs[:, center_h, center_w]  # [B]
            loss = criterion(outputs, labels)

            losses.update(loss.item(), patches.size(0))
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    auc_roc = roc_auc_score(all_labels, all_preds)

    # Find best threshold using PR curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    binary_preds = (all_preds >= best_threshold).astype(int)
    f1 = f1_score(all_labels, binary_preds)

    auc_pr = auc(recall, precision)

    metrics = {
        'loss': round(losses.avg, 6),
        'AUC_ROC': round(auc_roc, 4),
        'AUC_PR': round(auc_pr, 4),
        'F1': round(f1, 4),
        'threshold': round(float(best_threshold), 4),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='HyperSIGMA Target Detection')
    parser.add_argument('--dataset', type=str, default='Sandiego',
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--mode', type=str, default='sa', choices=['sa', 'ss'],
                        help='Mode: sa (spatial-only) or ss (spectral-spatial)')
    parser.add_argument('--spat_weights', type=str,
                        default='pretrained/spat-vit-base-ultra-checkpoint-1599.pth')
    parser.add_argument('--spec_weights', type=str,
                        default='pretrained/spec-vit-base-ultra-checkpoint-1599.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--train_ratio', type=float, default=0.5)
    parser.add_argument('--num_tokens', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/target_detection')
    args = parser.parse_args()

    setup_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode} ({'spectral-spatial' if args.mode == 'ss' else 'spatial-only'})")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    if args.dataset.lower() == 'sandiego':
        data_file = os.path.join(args.data_dir, 'Sandiego.mat')
        mat = sio.loadmat(data_file)
        hsi = mat['data']
        target_map = mat['map']
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Data shape: {hsi.shape}")
    print(f"Target map shape: {target_map.shape}")
    print(f"Number of target pixels: {(target_map > 0).sum()}")

    # Get target spectrum from labeled pixels
    target_spectrum = get_target_spectrum(hsi.astype(np.float32), target_map)
    # Normalize target spectrum
    hsi_normalized = (hsi.astype(np.float32) - hsi.min()) / (hsi.max() - hsi.min() + 1e-8)
    target_spectrum = get_target_spectrum(hsi_normalized, target_map)

    print(f"Target spectrum shape: {target_spectrum.shape}")

    # Create datasets
    train_dataset = TargetDetectionDataset(
        hsi, target_map, target_spectrum,
        patch_size=args.patch_size, mode='train',
        train_ratio=args.train_ratio, seed=args.seed
    )
    test_dataset = TargetDetectionDataset(
        hsi, target_map, target_spectrum,
        patch_size=args.patch_size, mode='test',
        train_ratio=args.train_ratio, seed=args.seed
    )

    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Create model
    print("\nCreating model...")
    H, W, C = hsi.shape

    if args.mode == 'ss':
        model = SSTargetDetectionHead(
            img_size=args.patch_size,
            in_channels=C,
            spat_weights=args.spat_weights,
            spec_weights=args.spec_weights,
            num_tokens=args.num_tokens,
        )
    else:
        model = TargetDetectionHead(
            img_size=args.patch_size,
            in_channels=C,
            spat_weights=args.spat_weights,
        )

    model = model.cuda()

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)

    # Training
    print("\nStarting training...")
    t0 = time.time()
    best_auc = 0
    best_metrics = None

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            metrics = evaluate(model, test_loader, criterion)
            print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.6f}, "
                  f"AUC_ROC={metrics['AUC_ROC']:.4f}, F1={metrics['F1']:.4f}")

            if metrics['AUC_ROC'] > best_auc:
                best_auc = metrics['AUC_ROC']
                best_metrics = metrics.copy()

    train_time = time.time() - t0
    print(f"\nTraining completed in {train_time:.1f}s")

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate(model, test_loader, criterion)

    print("\nResults:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v}")

    # Save results using ResultManager
    manager = ResultManager(
        task="target_detection",
        dataset=args.dataset,
        checkpoint_path=args.spat_weights,
        experiment_name="hypersigma_benchmark"
    )
    manager.set_config(experiment_config={
        'dataset': args.dataset,
        'data_dir': args.data_dir,
        'mode': args.mode,
        'spat_weights': args.spat_weights,
        'spec_weights': args.spec_weights if args.mode == 'ss' else None,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'patch_size': args.patch_size,
        'train_ratio': args.train_ratio,
        'num_tokens': args.num_tokens if args.mode == 'ss' else None,
        'model': 'HyperSIGMA',
    })

    # Use best metrics if available, otherwise final metrics
    metrics_to_save = best_metrics if best_metrics else final_metrics

    manager.log_run(
        seed=args.seed,
        metrics=TargetDetectionMetrics(
            AUC_ROC=metrics_to_save['AUC_ROC'],
            F1=metrics_to_save['F1'],
            precision=metrics_to_save.get('precision', 0.0),
            recall=metrics_to_save.get('recall', 0.0),
            average_precision=metrics_to_save.get('AUC_PR'),
        )
    )
    manager.try_auto_aggregate()


if __name__ == '__main__':
    main()
