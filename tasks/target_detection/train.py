"""
HyperSIGMA Target Detection Training Script.

Trains target detection models on hyperspectral images using pseudo-labels
generated from RX anomaly detection. This matches the original HyperSIGMA
implementation approach.

Key differences from supervised approach:
- Uses RX detector to generate pseudo-labels (not ground truth)
- Top 0.15% pixels → target (label=1)
- Bottom 30% pixels → background (label=0)
- Rest → ignore (label=255)
- Uses StandardScaler normalization instead of Min-Max
- Uses overlapping patches (32x32 with 16-pixel overlap)
- CrossEntropyLoss with ignore_index=255
"""

import os
import sys
import argparse
import json
import time
import random
from datetime import datetime

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance

# Add parent directories to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HYPERSIGMA_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, HYPERSIGMA_ROOT)

# Default data directory (relative to hypersigma-benchmark root)
DEFAULT_DATA_DIR = os.path.join(HYPERSIGMA_ROOT, 'data', 'target_detection')

from hypersigma.models.task_heads import TargetDetectionHead, SSTargetDetectionHead
from hypersigma.mmcv_custom import LayerDecayOptimizerConstructor_ViT
from mmengine.optim import build_optim_wrapper


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


def compute_rx_scores(hsi):
    """Compute RX anomaly detection scores using Mahalanobis distance.

    Args:
        hsi: HSI cube (H, W, C) - should be normalized

    Returns:
        rx_scores: Anomaly scores (H, W), higher = more anomalous
    """
    H, W, C = hsi.shape
    X = hsi.reshape(-1, C)  # (N, C)

    # Compute mean and covariance
    mean = np.mean(X, axis=0)
    cov = EmpiricalCovariance().fit(X)

    # Compute Mahalanobis distance for each pixel
    rx_scores = cov.mahalanobis(X)
    rx_scores = rx_scores.reshape(H, W)

    return rx_scores


def generate_pseudo_labels(hsi, target_ratio=0.0015, background_ratio=0.3):
    """Generate pseudo-labels using RX anomaly detector.

    Following the original HyperSIGMA target detection approach:
    - Top target_ratio (0.15%) → target (label=1)
    - Bottom background_ratio (30%) → background (label=0)
    - Rest → ignore (label=255)

    Args:
        hsi: HSI cube (H, W, C)
        target_ratio: Fraction of pixels to label as target (default: 0.0015 = 0.15%)
        background_ratio: Fraction of pixels to label as background (default: 0.3 = 30%)

    Returns:
        pseudo_labels: Label map (H, W) with values 0, 1, or 255
    """
    H, W, C = hsi.shape
    N = H * W

    # Compute RX scores
    rx_scores = compute_rx_scores(hsi)
    rx_flat = rx_scores.flatten()

    # Determine thresholds
    n_target = max(1, int(N * target_ratio))
    n_background = int(N * background_ratio)

    sorted_indices = np.argsort(rx_flat)
    target_threshold = np.partition(rx_flat, -n_target)[-n_target]
    background_threshold = np.partition(rx_flat, n_background)[n_background]

    # Create pseudo-labels
    pseudo_labels = np.full((H, W), 255, dtype=np.uint8)  # Default: ignore
    pseudo_labels[rx_scores >= target_threshold] = 1  # Target
    pseudo_labels[rx_scores <= background_threshold] = 0  # Background

    print(f"Pseudo-labels generated:")
    print(f"  Target (label=1): {(pseudo_labels == 1).sum()} pixels ({100*target_ratio:.2f}%)")
    print(f"  Background (label=0): {(pseudo_labels == 0).sum()} pixels ({100*background_ratio:.1f}%)")
    print(f"  Ignore (label=255): {(pseudo_labels == 255).sum()} pixels")

    return pseudo_labels, rx_scores


class TargetDetectionPatchDataset(Data.Dataset):
    """Dataset for target detection with overlapping patch extraction.

    Matches original HyperSIGMA implementation with:
    - 32x32 patches with 16-pixel overlap
    - StandardScaler normalization
    - Pseudo-labels from RX detector
    """

    def __init__(self, data, pseudo_labels, target_spectrum, patch_size=32, overlap=16,
                 mode='train', train_ratio=0.8, seed=42):
        """
        Args:
            data: HSI cube (H, W, C) - already StandardScaler normalized
            pseudo_labels: Pseudo-label map (H, W), values 0/1/255
            target_spectrum: Target spectrum derived from pseudo-target pixels [C]
            patch_size: Size of patches (default: 32)
            overlap: Overlap between patches (default: 16)
            mode: 'train' or 'test'
            train_ratio: Ratio of data for training
        """
        self.data = data.astype(np.float32)
        self.pseudo_labels = pseudo_labels
        self.target_spectrum = target_spectrum.astype(np.float32)
        self.patch_size = patch_size
        self.step = patch_size - overlap  # 16

        self.H, self.W, self.C = self.data.shape

        # Generate patch positions with overlap
        positions = []
        for h in range(0, self.H - patch_size + 1, self.step):
            for w in range(0, self.W - patch_size + 1, self.step):
                # Check if patch has any labeled pixels (not all 255)
                patch_labels = pseudo_labels[h:h+patch_size, w:w+patch_size]
                if not np.all(patch_labels == 255):
                    positions.append((h, w))

        # Split train/test
        np.random.seed(seed)
        np.random.shuffle(positions)

        n_train = int(len(positions) * train_ratio)

        if mode == 'train':
            self.positions = positions[:n_train]
        else:
            self.positions = positions[n_train:]

        print(f"{mode} set: {len(self.positions)} patches (size={patch_size}, overlap={overlap})")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        h, w = self.positions[idx]

        # Extract patch
        patch = self.data[h:h+self.patch_size, w:w+self.patch_size, :]
        labels = self.pseudo_labels[h:h+self.patch_size, w:w+self.patch_size]

        # Convert to tensor [C, H, W]
        patch = torch.from_numpy(patch.transpose(2, 0, 1)).float()
        labels = torch.from_numpy(labels.astype(np.int64))
        target_spec = torch.from_numpy(self.target_spectrum).float()

        return patch, target_spec, labels


class TargetDetectionFullImage(Data.Dataset):
    """Dataset for full image target detection evaluation."""

    def __init__(self, data, patch_size=32, overlap=16):
        self.data = data.astype(np.float32)
        self.patch_size = patch_size
        self.step = patch_size - overlap

        self.H, self.W, self.C = self.data.shape

        # Generate all patch positions
        self.positions = []
        for h in range(0, self.H - patch_size + 1, self.step):
            for w in range(0, self.W - patch_size + 1, self.step):
                self.positions.append((h, w))

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        h, w = self.positions[idx]
        patch = self.data[h:h+self.patch_size, w:w+self.patch_size, :]
        patch = torch.from_numpy(patch.transpose(2, 0, 1)).float()
        return patch, h, w


def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    losses = AvgMeter()

    for patches, target_specs, labels in train_loader:
        patches = patches.cuda()
        target_specs = target_specs.cuda()
        labels = labels.cuda()  # [B, H, W] with values 0, 1, or 255

        optimizer.zero_grad()

        # Forward pass - model outputs detection scores [B, H, W]
        outputs = model(patches, target_specs)  # [B, H, W]

        # Create mask for valid pixels (not 255)
        valid_mask = labels != 255

        # Flatten outputs and labels, apply mask
        outputs_flat = outputs.view(-1)
        labels_flat = labels.view(-1).float()
        valid_mask_flat = valid_mask.view(-1)

        # Only compute loss on valid pixels
        if valid_mask_flat.sum() > 0:
            valid_outputs = outputs_flat[valid_mask_flat]
            valid_labels = labels_flat[valid_mask_flat]
            loss = criterion(valid_outputs, valid_labels)
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), valid_mask_flat.sum().item())

    return losses.avg


def evaluate(model, test_loader, criterion, gt_map=None):
    """Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        criterion: Loss function
        gt_map: Ground truth binary map (H, W) for final evaluation metrics
    """
    model.eval()
    losses = AvgMeter()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for patches, target_specs, labels in test_loader:
            patches = patches.cuda()
            target_specs = target_specs.cuda()
            labels = labels.cuda()

            # Forward pass - model outputs detection scores [B, H, W]
            outputs = model(patches, target_specs)

            # Create mask for valid pixels (not 255)
            valid_mask = labels != 255

            # Flatten and compute loss on valid pixels
            outputs_flat = outputs.view(-1)
            labels_flat = labels.view(-1).float()
            valid_mask_flat = valid_mask.view(-1)

            if valid_mask_flat.sum() > 0:
                valid_outputs = outputs_flat[valid_mask_flat]
                valid_labels = labels_flat[valid_mask_flat]
                loss = criterion(valid_outputs, valid_labels)
                losses.update(loss.item(), valid_mask_flat.sum().item())

            # Get predictions (apply sigmoid for probabilities)
            probs = torch.sigmoid(outputs)  # [B, H, W]

            # Collect predictions and labels for valid pixels (not 255)
            all_preds.extend(probs[valid_mask].cpu().numpy())
            all_labels.extend(labels[valid_mask].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    if len(np.unique(all_labels)) > 1:
        auc_roc = roc_auc_score(all_labels, all_preds)

        # Find best threshold using PR curve
        precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        binary_preds = (all_preds >= best_threshold).astype(int)
        f1 = f1_score(all_labels, binary_preds)

        auc_pr = auc(recall, precision)
    else:
        auc_roc = 0.0
        auc_pr = 0.0
        f1 = 0.0
        best_threshold = 0.5

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
    # Updated parameters to match original implementation
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--overlap', type=int, default=16)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--num_tokens', type=int, default=100)
    parser.add_argument('--target_ratio', type=float, default=0.0015,
                        help='Fraction of pixels for pseudo-target labels (default: 0.15%)')
    parser.add_argument('--background_ratio', type=float, default=0.3,
                        help='Fraction of pixels for pseudo-background labels (default: 30%)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/target_detection')
    args = parser.parse_args()

    setup_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode} ({'spectral-spatial' if args.mode == 'ss' else 'spatial-only'})")
    print(f"Pseudo-label generation: RX detector")
    print(f"  Target ratio: {args.target_ratio*100:.2f}%")
    print(f"  Background ratio: {args.background_ratio*100:.1f}%")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    if args.dataset.lower() == 'sandiego':
        data_file = os.path.join(args.data_dir, 'Sandiego.mat')
        mat = sio.loadmat(data_file)
        hsi = mat['data']
        gt_map = mat['map']
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Data shape: {hsi.shape}")
    print(f"Ground truth shape: {gt_map.shape}")
    print(f"Number of true target pixels: {(gt_map > 0).sum()}")

    # Auto-adjust patch size for small datasets
    H, W, C = hsi.shape
    total_pixels = H * W
    if total_pixels < 50000 and args.patch_size == 32:
        # For small datasets like Sandiego (100x100 = 10000 pixels),
        # use smaller patches to get more training samples
        args.patch_size = 16
        args.overlap = 8
        print(f"\nSmall dataset detected ({H}x{W}={total_pixels} pixels)")
        print(f"Auto-adjusted: patch_size={args.patch_size}, overlap={args.overlap}")

    # Normalize using StandardScaler (instead of Min-Max)
    print("\nNormalizing with StandardScaler...")
    H, W, C = hsi.shape
    hsi_flat = hsi.reshape(-1, C).astype(np.float32)
    scaler = StandardScaler()
    hsi_normalized = scaler.fit_transform(hsi_flat).reshape(H, W, C)

    # Generate pseudo-labels using RX detector
    print("\nGenerating pseudo-labels using RX detector...")
    pseudo_labels, rx_scores = generate_pseudo_labels(
        hsi_normalized,
        target_ratio=args.target_ratio,
        background_ratio=args.background_ratio
    )

    # Extract target spectrum from pseudo-target pixels
    target_mask = pseudo_labels == 1
    if target_mask.sum() > 0:
        target_spectrum = hsi_normalized[target_mask].mean(axis=0)
    else:
        # Fallback: use pixels with highest RX scores
        flat_rx = rx_scores.flatten()
        top_indices = np.argsort(flat_rx)[-max(1, int(H*W*0.001)):]
        target_spectrum = hsi_normalized.reshape(-1, C)[top_indices].mean(axis=0)
    print(f"Target spectrum extracted from {target_mask.sum()} pseudo-target pixels")

    # Create datasets
    train_dataset = TargetDetectionPatchDataset(
        hsi_normalized, pseudo_labels, target_spectrum,
        patch_size=args.patch_size, overlap=args.overlap,
        mode='train', train_ratio=args.train_ratio, seed=args.seed
    )
    test_dataset = TargetDetectionPatchDataset(
        hsi_normalized, pseudo_labels, target_spectrum,
        patch_size=args.patch_size, overlap=args.overlap,
        mode='test', train_ratio=args.train_ratio, seed=args.seed
    )

    train_loader = Data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2
    )
    test_loader = Data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2
    )

    # Create model
    print("\nCreating model...")

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

    # Loss and optimizer (with Layer Decay)
    # Use BCEWithLogitsLoss since model outputs detection scores [B, H, W]
    criterion = nn.BCEWithLogitsLoss().cuda()

    optim_wrapper = dict(
        optimizer=dict(type='AdamW', lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay),
        constructor='LayerDecayOptimizerConstructor_ViT',
        paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9)
    )
    optimizer = build_optim_wrapper(model, optim_wrapper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, args.epochs, eta_min=0)

    # Training
    print("\nStarting training...")
    t0 = time.time()
    best_auc = 0
    best_metrics = None

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()

        # Evaluate every epoch (since we only have 10 epochs)
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

    print("\nResults (on pseudo-labels):")
    for k, v in final_metrics.items():
        print(f"  {k}: {v}")

    # Save results
    mode_dir = os.path.join(args.output_dir, args.mode)
    os.makedirs(mode_dir, exist_ok=True)

    result = {
        'task': 'target_detection',
        'dataset': args.dataset,
        'model': 'HyperSIGMA',
        'mode': args.mode,
        'seed': args.seed,
        'metrics': final_metrics,
        'best_metrics': best_metrics if best_metrics else final_metrics,
        'config': {
            'dataset': args.dataset,
            'data_dir': args.data_dir,
            'mode': args.mode,
            'spat_weights': args.spat_weights,
            'spec_weights': args.spec_weights if args.mode == 'ss' else None,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'patch_size': args.patch_size,
            'overlap': args.overlap,
            'train_ratio': args.train_ratio,
            'num_tokens': args.num_tokens if args.mode == 'ss' else None,
            'target_ratio': args.target_ratio,
            'background_ratio': args.background_ratio,
            'normalization': 'StandardScaler',
            'loss': 'BCEWithLogitsLoss (masked for ignore=255)',
            'optimizer': 'AdamW with LayerDecay(0.9)',
            'seed': args.seed,
        },
        'pseudo_label_stats': {
            'n_target': int((pseudo_labels == 1).sum()),
            'n_background': int((pseudo_labels == 0).sum()),
            'n_ignore': int((pseudo_labels == 255).sum()),
        },
        'timestamp': datetime.now().isoformat(),
    }

    result_file = os.path.join(mode_dir, f'result_{args.dataset.lower()}_{args.mode}_seed{args.seed}.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_file}")


if __name__ == '__main__':
    main()
