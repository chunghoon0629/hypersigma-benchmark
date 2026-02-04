"""
HyperSIGMA Denoising Training Script.

Trains denoising models on hyperspectral images with synthetic noise.
Uses Gaussian noise for simplicity (can be extended to complex noise).
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
import torch.nn.functional as F
import torch.utils.data as Data

# Add parent directories to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HYPERSIGMA_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
PROJECT_ROOT = os.path.join(HYPERSIGMA_ROOT, '..')
sys.path.insert(0, HYPERSIGMA_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from downstream_task_head.utils.result_manager import ResultManager, DenoisingMetrics

# WDC Mall data directory (project benchmark data)
WDC_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'benchmark', 'denoising', 'WDC')

from hypersigma.models.task_heads import DenoisingHead


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


def add_gaussian_noise(image, sigma_range=(10, 70)):
    """Add Gaussian noise to image."""
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    noise = np.random.randn(*image.shape) * sigma / 255.0
    noisy = image + noise
    return np.clip(noisy, 0, 1), sigma


def compute_psnr(img1, img2):
    """Compute PSNR between two images."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_val = 1.0
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(img1, img2, window_size=11):
    """Compute SSIM between two images (simplified version)."""
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def compute_sam(img1, img2):
    """Compute Spectral Angle Mapper (SAM) between two images."""
    # img1, img2: [B, C, H, W]
    dot_product = (img1 * img2).sum(dim=1)
    norm1 = torch.sqrt((img1 ** 2).sum(dim=1))
    norm2 = torch.sqrt((img2 ** 2).sum(dim=1))

    cos_angle = dot_product / (norm1 * norm2 + 1e-8)
    cos_angle = torch.clamp(cos_angle, -1, 1)
    sam = torch.acos(cos_angle)

    return sam.mean().item() * 180 / np.pi  # Convert to degrees


class WDCTrainDataset(Data.Dataset):
    """WDC Training dataset - loads pre-extracted patches."""

    def __init__(self, data_dir, sigma_range=(10, 70)):
        self.data_dir = data_dir
        self.sigma_range = sigma_range
        self.patch_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])
        print(f"Found {len(self.patch_files)} training patches")

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        mat_path = os.path.join(self.data_dir, self.patch_files[idx])
        data = sio.loadmat(mat_path)
        clean = data['data'].astype(np.float32)  # [H, W, C]

        # Add noise on-the-fly
        noisy, _ = add_gaussian_noise(clean.copy(), self.sigma_range)

        # Convert to tensor [C, H, W]
        clean = torch.from_numpy(clean.transpose(2, 0, 1)).float()
        noisy = torch.from_numpy(noisy.transpose(2, 0, 1)).float()

        return noisy, clean


class WDCTestDataset(Data.Dataset):
    """WDC Test dataset - loads pre-generated noisy/clean pairs."""

    def __init__(self, data_dir, sigma=50):
        self.data_dir = data_dir
        self.sigma = sigma

        # Load test data for specified sigma
        mat_path = os.path.join(data_dir, f'wdc_sigma{sigma}.mat')
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Test file not found: {mat_path}")

        data = sio.loadmat(mat_path)
        self.noisy = data['input'].astype(np.float32)  # [H, W, C]
        self.clean = data['gt'].astype(np.float32)  # [H, W, C]

        self.H, self.W, self.C = self.clean.shape
        print(f"Loaded WDC test data (sigma={sigma}): {self.clean.shape}")

    def __len__(self):
        return 1  # Single full image

    def __getitem__(self, idx):
        # Convert to tensor [C, H, W]
        clean = torch.from_numpy(self.clean.transpose(2, 0, 1)).float()
        noisy = torch.from_numpy(self.noisy.transpose(2, 0, 1)).float()
        return noisy, clean


def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    losses = AvgMeter()

    for noisy, clean in train_loader:
        noisy = noisy.cuda()
        clean = clean.cuda()

        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), noisy.size(0))

    return losses.avg


def evaluate(model, test_loader):
    """Evaluate model on test set."""
    model.eval()
    psnr_list = []
    ssim_list = []
    sam_list = []

    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy = noisy.cuda()
            clean = clean.cuda()

            output = model(noisy)
            output = torch.clamp(output, 0, 1)

            # Compute metrics
            psnr = compute_psnr(output, clean)
            ssim = compute_ssim(output, clean)
            sam = compute_sam(output, clean)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            sam_list.append(sam)

    metrics = {
        'PSNR': round(np.mean(psnr_list), 4),
        'SSIM': round(np.mean(ssim_list), 4),
        'SAM': round(np.mean(sam_list), 4),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='HyperSIGMA Denoising')
    parser.add_argument('--dataset', type=str, default='WDC',
                        help='Dataset name (WDC Mall)')
    parser.add_argument('--spat_weights', type=str,
                        default='pretrained/spat-vit-base-ultra-checkpoint-1599.pth')
    parser.add_argument('--spec_weights', type=str,
                        default='pretrained/spec-vit-base-ultra-checkpoint-1599.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--sigma_min', type=int, default=10)
    parser.add_argument('--sigma_max', type=int, default=70)
    parser.add_argument('--num_tokens', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/denoising')
    args = parser.parse_args()

    setup_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Noise sigma range: [{args.sigma_min}, {args.sigma_max}]")
    print(f"{'='*60}\n")

    # Load data - WDC Mall dataset
    print("Loading data...")
    if args.dataset.lower() != 'wdc':
        raise ValueError(f"Only WDC dataset is supported. Got: {args.dataset}")

    # WDC has pre-processed train/test split
    train_dir = os.path.join(WDC_DATA_DIR, 'train')
    test_dir = os.path.join(WDC_DATA_DIR, 'test')

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"WDC train directory not found: {train_dir}")

    # Load one patch to get dimensions
    sample_patch = sio.loadmat(os.path.join(train_dir, 'patch_0000.mat'))['data']
    patch_size = sample_patch.shape[0]
    C = sample_patch.shape[2]
    print(f"WDC patch size: {patch_size}, channels: {C}")

    # WDC uses pre-extracted patches
    train_dataset = WDCTrainDataset(
        train_dir,
        sigma_range=(args.sigma_min, args.sigma_max)
    )
    test_dataset = WDCTestDataset(
        test_dir,
        sigma=args.sigma_max  # Use max sigma for testing
    )

    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = Data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create model
    print("\nCreating model...")
    model = DenoisingHead(
        img_size=patch_size,
        in_channels=C,
        spat_weights=args.spat_weights,
        spec_weights=args.spec_weights,
        num_tokens=args.num_tokens,
    )
    model = model.cuda()

    # Loss and optimizer
    criterion = nn.L1Loss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)

    # Training
    print("\nStarting training...")
    t0 = time.time()
    best_psnr = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            metrics = evaluate(model, test_loader)
            print(f"Epoch {epoch+1}/{args.epochs}: loss={train_loss:.6f}, "
                  f"PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.4f}, SAM={metrics['SAM']:.2f}")

            if metrics['PSNR'] > best_psnr:
                best_psnr = metrics['PSNR']
                best_metrics = metrics.copy()

    train_time = time.time() - t0
    print(f"\nTraining completed in {train_time:.1f}s")

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate(model, test_loader)

    print("\nResults:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v}")

    # Save results using ResultManager
    results_dir = os.path.join(HYPERSIGMA_ROOT, 'results')
    manager = ResultManager(
        task="denoising",
        dataset=args.dataset,
        checkpoint_path=args.spat_weights,
        output_base_dir=results_dir,
        experiment_name="hypersigma_benchmark"
    )
    manager.set_config(experiment_config={
        'dataset': args.dataset,
        'spat_weights': args.spat_weights,
        'spec_weights': args.spec_weights,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'patch_size': patch_size,
        'sigma_range': [args.sigma_min, args.sigma_max],
        'num_tokens': args.num_tokens,
        'model': 'HyperSIGMA',
    })

    manager.log_run(
        seed=args.seed,
        metrics=DenoisingMetrics(
            PSNR=final_metrics['PSNR'],
            SSIM=final_metrics['SSIM'],
            SAM=final_metrics['SAM'],
        )
    )
    manager.try_auto_aggregate()


if __name__ == '__main__':
    main()
