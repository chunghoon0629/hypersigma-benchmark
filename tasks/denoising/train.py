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

# Default data directory (uses classification data for denoising)
DEFAULT_DATA_DIR = os.path.join(HYPERSIGMA_ROOT, 'data', 'classification')

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


class DenoisingDataset(Data.Dataset):
    """Dataset for denoising with on-the-fly noise addition."""

    def __init__(self, data, patch_size=64, sigma_range=(10, 70), train=True):
        """
        Args:
            data: HSI cube (H, W, C)
            patch_size: Size of patches to extract
            sigma_range: Range of noise sigma values
            train: Whether this is training mode
        """
        self.data = data
        self.patch_size = patch_size
        self.sigma_range = sigma_range
        self.train = train

        # Normalize to [0, 1]
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min() + 1e-8)

        self.H, self.W, self.C = self.data.shape

        if train:
            # Generate random patch positions
            self.n_patches = max(100, (self.H // patch_size) * (self.W // patch_size) * 10)
        else:
            # Use non-overlapping patches for evaluation
            self.n_patches = (self.H // patch_size) * (self.W // patch_size)
            self.grid_h = self.H // patch_size
            self.grid_w = self.W // patch_size

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        if self.train:
            # Random patch
            h = np.random.randint(0, self.H - self.patch_size)
            w = np.random.randint(0, self.W - self.patch_size)
        else:
            # Grid-based patch
            grid_idx = idx
            h = (grid_idx // self.grid_w) * self.patch_size
            w = (grid_idx % self.grid_w) * self.patch_size

        clean = self.data[h:h + self.patch_size, w:w + self.patch_size, :]
        noisy, sigma = add_gaussian_noise(clean.copy(), self.sigma_range)

        # Convert to tensor [C, H, W]
        clean = torch.from_numpy(clean.transpose(2, 0, 1)).float()
        noisy = torch.from_numpy(noisy.transpose(2, 0, 1)).float()

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
    parser.add_argument('--dataset', type=str, default='IndianPines',
                        help='Dataset name (uses classification data)')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
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

    # Load data - use classification data for denoising
    print("Loading data...")
    if args.dataset.lower() == 'indianpines':
        data_file = os.path.join(args.data_dir, 'Indian_pines_corrected.mat')
        data = sio.loadmat(data_file)
        # Try different possible keys
        if 'data' in data:
            hsi = data['data']
        elif 'indian_pines_corrected' in data:
            hsi = data['indian_pines_corrected']
        else:
            # Find the main data array
            for k, v in data.items():
                if not k.startswith('_') and isinstance(v, np.ndarray) and v.ndim == 3:
                    hsi = v
                    break
    elif args.dataset.lower() == 'paviau':
        data_file = os.path.join(args.data_dir, 'PaviaU.mat')
        data = sio.loadmat(data_file)
        if 'paviaU' in data:
            hsi = data['paviaU']
        else:
            for k, v in data.items():
                if not k.startswith('_') and isinstance(v, np.ndarray) and v.ndim == 3:
                    hsi = v
                    break
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Data shape: {hsi.shape}")

    # Split into train/test (80/20)
    H, W, C = hsi.shape
    split_h = int(H * 0.8)

    train_data = hsi[:split_h, :, :]
    test_data = hsi[split_h:, :, :]

    print(f"Train region: {train_data.shape}")
    print(f"Test region: {test_data.shape}")

    # Adjust patch size if needed
    patch_size = min(args.patch_size, min(train_data.shape[0], train_data.shape[1]))
    patch_size = min(patch_size, min(test_data.shape[0], test_data.shape[1]))

    # Create datasets
    train_dataset = DenoisingDataset(
        train_data, patch_size=patch_size,
        sigma_range=(args.sigma_min, args.sigma_max), train=True
    )
    test_dataset = DenoisingDataset(
        test_data, patch_size=patch_size,
        sigma_range=(args.sigma_min, args.sigma_max), train=False
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
    manager = ResultManager(
        task="denoising",
        dataset=args.dataset,
        checkpoint_path=args.spat_weights,
        experiment_name="hypersigma_benchmark"
    )
    manager.set_config(experiment_config={
        'dataset': args.dataset,
        'data_dir': args.data_dir,
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
