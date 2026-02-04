"""
HyperSIGMA Spectral Unmixing Training Script.

Trains spectral unmixing models on hyperspectral images.
Uses blind unmixing approach (learns both endmembers and abundances).
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
import torch.nn.functional as F
import torch.utils.data as Data

# Add parent directories to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HYPERSIGMA_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, HYPERSIGMA_ROOT)

# Default data directory (relative to hypersigma-benchmark root)
DEFAULT_DATA_DIR = os.path.join(HYPERSIGMA_ROOT, 'data', 'unmixing')

from hypersigma.models.task_heads import UnmixingHead, SSUnmixingHead


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


def SAD_loss(pred, target):
    """Spectral Angle Distance loss."""
    dot_product = (pred * target).sum(dim=-1)
    pred_norm = torch.norm(pred, dim=-1) + 1e-8
    target_norm = torch.norm(target, dim=-1) + 1e-8
    cos_angle = dot_product / (pred_norm * target_norm)
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle = torch.acos(cos_angle)
    return angle.mean()


def compute_rmse(pred, target):
    """Compute RMSE between predictions and targets."""
    mse = torch.mean((pred - target) ** 2)
    return torch.sqrt(mse).item()


def compute_sam_degrees(pred, target):
    """Compute mean SAM in degrees."""
    dot_product = (pred * target).sum(dim=-1)
    pred_norm = torch.norm(pred, dim=-1) + 1e-8
    target_norm = torch.norm(target, dim=-1) + 1e-8
    cos_angle = dot_product / (pred_norm * target_norm)
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle = torch.acos(cos_angle)
    return (angle.mean() * 180 / np.pi).item()


class UnmixingDataset(Data.Dataset):
    """Dataset for spectral unmixing with patch extraction."""

    def __init__(self, hsi, abundances, patch_size=7, mode='train',
                 train_ratio=0.8, seed=42):
        """
        Args:
            hsi: HSI cube (H, W, C)
            abundances: Ground truth abundances (num_end, H*W) or (H, W, num_end)
            patch_size: Size of patches to extract
            mode: 'train' or 'test'
            train_ratio: Ratio of data for training
        """
        self.patch_size = patch_size
        self.half_size = patch_size // 2

        # Normalize HSI
        self.hsi = hsi.astype(np.float32)
        self.hsi_min = self.hsi.min()
        self.hsi_max = self.hsi.max()
        self.hsi = (self.hsi - self.hsi_min) / (self.hsi_max - self.hsi_min + 1e-8)

        self.H, self.W, self.C = self.hsi.shape

        # Handle abundances shape
        if abundances.ndim == 2:
            # (num_end, H*W) -> (H, W, num_end)
            num_end = abundances.shape[0]
            self.abundances = abundances.T.reshape(self.H, self.W, num_end)
        else:
            self.abundances = abundances

        self.num_endmembers = self.abundances.shape[-1]

        # Generate valid positions
        valid_positions = [
            (h, w)
            for h in range(self.half_size, self.H - self.half_size)
            for w in range(self.half_size, self.W - self.half_size)
        ]

        # Split train/test
        np.random.seed(seed)
        np.random.shuffle(valid_positions)

        n_train = int(len(valid_positions) * train_ratio)

        if mode == 'train':
            self.positions = valid_positions[:n_train]
        else:
            self.positions = valid_positions[n_train:]

        print(f"{mode} set: {len(self.positions)} samples, {self.num_endmembers} endmembers")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        h, w = self.positions[idx]

        # Extract patch
        patch = self.hsi[
            h - self.half_size:h + self.half_size + 1,
            w - self.half_size:w + self.half_size + 1,
            :
        ]

        # Get center pixel abundance and spectrum
        abundance = self.abundances[h, w, :]
        center_spectrum = self.hsi[h, w, :]

        # Convert to tensors [C, H, W] for patch
        patch = torch.from_numpy(patch.transpose(2, 0, 1)).float()
        abundance = torch.from_numpy(abundance.astype(np.float32))
        center_spectrum = torch.from_numpy(center_spectrum).float()

        return patch, abundance, center_spectrum


def train_epoch(model, train_loader, optimizer, recon_weight=0.1):
    """Train for one epoch."""
    model.train()
    abundance_losses = AvgMeter()
    recon_losses = AvgMeter()

    for patches, abundances_gt, center_spectra in train_loader:
        patches = patches.cuda()
        abundances_gt = abundances_gt.cuda()
        center_spectra = center_spectra.cuda()

        optimizer.zero_grad()

        # Forward pass
        pred_abundances, reconstructed = model(patches)

        # Extract center pixel abundances from spatial map [B, num_end, H, W]
        center_h = pred_abundances.shape[2] // 2
        center_w = pred_abundances.shape[3] // 2
        pred_center_abund = pred_abundances[:, :, center_h, center_w]  # [B, num_end]

        # Abundance loss (MSE)
        abundance_loss = F.mse_loss(pred_center_abund, abundances_gt)

        # Reconstruction loss (SAD) - compare center pixel
        if reconstructed is not None:
            center_recon = reconstructed[:, :, center_h, center_w]  # [B, C]
            recon_loss = SAD_loss(center_recon, center_spectra)
            loss = abundance_loss + recon_weight * recon_loss
            recon_losses.update(recon_loss.item(), patches.size(0))
        else:
            loss = abundance_loss

        loss.backward()
        optimizer.step()

        abundance_losses.update(abundance_loss.item(), patches.size(0))

    return abundance_losses.avg, recon_losses.avg


def evaluate(model, test_loader):
    """Evaluate model on test set."""
    model.eval()
    all_pred_abundances = []
    all_gt_abundances = []
    all_reconstructed = []
    all_center_spectra = []

    with torch.no_grad():
        for patches, abundances_gt, center_spectra in test_loader:
            patches = patches.cuda()

            pred_abundances, reconstructed = model(patches)

            # Extract center pixel
            center_h = pred_abundances.shape[2] // 2
            center_w = pred_abundances.shape[3] // 2
            pred_center_abund = pred_abundances[:, :, center_h, center_w]

            all_pred_abundances.append(pred_center_abund.cpu())
            all_gt_abundances.append(abundances_gt)

            if reconstructed is not None:
                center_recon = reconstructed[:, :, center_h, center_w]
                all_reconstructed.append(center_recon.cpu())
                all_center_spectra.append(center_spectra)

    all_pred_abundances = torch.cat(all_pred_abundances, dim=0)
    all_gt_abundances = torch.cat(all_gt_abundances, dim=0)

    # Compute abundance RMSE
    abundance_rmse = compute_rmse(all_pred_abundances, all_gt_abundances)

    # Compute reconstruction SAM if available
    if all_reconstructed:
        all_reconstructed = torch.cat(all_reconstructed, dim=0)
        all_center_spectra = torch.cat(all_center_spectra, dim=0)
        recon_sam = compute_sam_degrees(all_reconstructed, all_center_spectra)
    else:
        recon_sam = 0.0

    metrics = {
        'abundance_RMSE': round(abundance_rmse, 6),
        'reconstruction_SAM': round(recon_sam, 4),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='HyperSIGMA Spectral Unmixing')
    parser.add_argument('--dataset', type=str, default='Urban4',
                        help='Dataset name (Urban4, Urban5, Urban6)')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--mode', type=str, default='sa', choices=['sa', 'ss'],
                        help='Mode: sa (spatial-only) or ss (spectral-spatial)')
    parser.add_argument('--spat_weights', type=str,
                        default='pretrained/spat-vit-base-ultra-checkpoint-1599.pth')
    parser.add_argument('--spec_weights', type=str,
                        default='pretrained/spec-vit-base-ultra-checkpoint-1599.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--num_tokens', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/unmixing')
    args = parser.parse_args()

    setup_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode} ({'spectral-spatial' if args.mode == 'ss' else 'spatial-only'})")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")

    # Load HSI
    hsi_file = os.path.join(args.data_dir, 'Urban_R162.mat')
    hsi_data = sio.loadmat(hsi_file)

    # Reshape Y from (C, N) to (H, W, C)
    Y = hsi_data['Y']
    nRow = int(hsi_data['nRow'][0, 0])
    nCol = int(hsi_data['nCol'][0, 0])
    hsi = Y.T.reshape(nRow, nCol, -1)

    print(f"HSI shape: {hsi.shape}")

    # Load ground truth based on number of endmembers
    if args.dataset.lower() == 'urban4':
        # Try benchmark path first
        gt_file = os.path.join(args.data_dir, 'groundTruth_4_end', 'groundTruth', 'end4_groundTruth.mat')
        if not os.path.exists(gt_file):
            gt_file = os.path.join(args.data_dir, 'end4_groundTruth.mat')
        num_end = 4
    elif args.dataset.lower() == 'urban5':
        gt_file = os.path.join(args.data_dir, 'groundTruth_Urban_end5', 'groundTruth', 'end5_groundTruth.mat')
        if not os.path.exists(gt_file):
            gt_file = os.path.join(args.data_dir, 'end5_groundTruth.mat')
        num_end = 5
    elif args.dataset.lower() == 'urban6':
        gt_file = os.path.join(args.data_dir, 'groundTruth_Urban_end6', 'groundTruth', 'end6_groundTruth.mat')
        if not os.path.exists(gt_file):
            gt_file = os.path.join(args.data_dir, 'end6_groundTruth.mat')
        num_end = 6
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    gt_data = sio.loadmat(gt_file)
    abundances = gt_data['A']  # (num_end, N)

    print(f"Abundances shape: {abundances.shape}")
    print(f"Number of endmembers: {num_end}")

    # Create datasets
    train_dataset = UnmixingDataset(
        hsi, abundances,
        patch_size=args.patch_size, mode='train',
        train_ratio=args.train_ratio, seed=args.seed
    )
    test_dataset = UnmixingDataset(
        hsi, abundances,
        patch_size=args.patch_size, mode='test',
        train_ratio=args.train_ratio, seed=args.seed
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
    H, W, C = hsi.shape

    if args.mode == 'ss':
        model = SSUnmixingHead(
            img_size=args.patch_size,
            in_channels=C,
            num_endmembers=num_end,
            spat_weights=args.spat_weights,
            spec_weights=args.spec_weights,
            num_tokens=args.num_tokens,
        )
    else:
        model = UnmixingHead(
            img_size=args.patch_size,
            in_channels=C,
            num_endmembers=num_end,
            spat_weights=args.spat_weights,
        )

    model = model.cuda()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)

    # Training
    print("\nStarting training...")
    t0 = time.time()
    best_rmse = float('inf')
    best_metrics = None

    for epoch in range(args.epochs):
        abundance_loss, recon_loss = train_epoch(model, train_loader, optimizer)
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            metrics = evaluate(model, test_loader)
            print(f"Epoch {epoch+1}/{args.epochs}: abundance_loss={abundance_loss:.6f}, "
                  f"RMSE={metrics['abundance_RMSE']:.6f}, SAM={metrics['reconstruction_SAM']:.2f}")

            if metrics['abundance_RMSE'] < best_rmse:
                best_rmse = metrics['abundance_RMSE']
                best_metrics = metrics.copy()

    train_time = time.time() - t0
    print(f"\nTraining completed in {train_time:.1f}s")

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate(model, test_loader)

    print("\nResults:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v}")

    # Save results
    mode_dir = os.path.join(args.output_dir, args.mode)
    os.makedirs(mode_dir, exist_ok=True)

    result = {
        'task': 'unmixing',
        'dataset': args.dataset,
        'model': 'HyperSIGMA',
        'mode': args.mode,
        'num_endmembers': num_end,
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
            'patch_size': args.patch_size,
            'train_ratio': args.train_ratio,
            'num_tokens': args.num_tokens if args.mode == 'ss' else None,
            'seed': args.seed,
        },
        'timestamp': datetime.now().isoformat(),
    }

    result_file = os.path.join(mode_dir, f'result_{args.dataset.lower()}_{args.mode}.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_file}")


if __name__ == '__main__':
    main()
