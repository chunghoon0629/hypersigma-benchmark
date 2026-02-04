"""
HyperSIGMA Denoising Training Script.

Trains denoising models on hyperspectral images with synthetic noise.
Uses Gaussian noise for simplicity (can be extended to complex noise).

Supports:
- IndianPines, PaviaU: Single HSI cube, noise added on-the-fly
- WDC: Pre-processed patches with pre-generated noisy test images
"""

import os
import sys
import argparse
import json
import time
import random
from datetime import datetime
from glob import glob

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

# Default data directory (uses classification data for denoising)
DEFAULT_DATA_DIR = os.path.join(HYPERSIGMA_ROOT, 'data', 'classification')

from hypersigma.models.task_heads import DenoisingHead
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


class WDCTrainDataset(Data.Dataset):
    """Dataset for WDC denoising using pre-processed clean patches with on-the-fly noise."""

    def __init__(self, data_dir, sigma_range=(10, 70)):
        """
        Args:
            data_dir: Path to WDC data directory containing train/patch_*.mat
            sigma_range: Range of noise sigma values for training
        """
        self.sigma_range = sigma_range
        train_dir = os.path.join(data_dir, 'train')
        self.patch_files = sorted(glob(os.path.join(train_dir, 'patch_*.mat')))

        if len(self.patch_files) == 0:
            raise ValueError(f"No training patches found in {train_dir}")

        # Load first patch to get shape
        sample = sio.loadmat(self.patch_files[0])
        self.patch_shape = sample['data'].shape  # (64, 64, 191)
        self.n_channels = self.patch_shape[2]

        print(f"Found {len(self.patch_files)} training patches, shape: {self.patch_shape}")

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        # Load clean patch
        data = sio.loadmat(self.patch_files[idx])
        clean = data['data'].astype(np.float32)  # (64, 64, 191)

        # Normalize to [0, 1]
        clean = (clean - clean.min()) / (clean.max() - clean.min() + 1e-8)

        # Add noise on-the-fly
        noisy, sigma = add_gaussian_noise(clean.copy(), self.sigma_range)

        # Convert to tensor [C, H, W]
        clean = torch.from_numpy(clean.transpose(2, 0, 1)).float()
        noisy = torch.from_numpy(noisy.transpose(2, 0, 1)).float()

        return noisy, clean


class WDCTestDataset(Data.Dataset):
    """Dataset for WDC denoising using pre-generated noisy test images."""

    def __init__(self, data_dir, sigma, patch_size=64):
        """
        Args:
            data_dir: Path to WDC data directory containing test/wdc_sigma*.mat
            sigma: Noise level (10, 30, 50, or 70)
            patch_size: Size of patches for evaluation
        """
        self.patch_size = patch_size

        test_file = os.path.join(data_dir, 'test', f'wdc_sigma{sigma}.mat')
        if not os.path.exists(test_file):
            raise ValueError(f"Test file not found: {test_file}")

        data = sio.loadmat(test_file)
        self.noisy = data['input'].astype(np.float32)  # (256, 256, 191)
        self.clean = data['gt'].astype(np.float32)     # (256, 256, 191)

        # Normalize both to [0, 1] using ground truth statistics
        gt_min, gt_max = self.clean.min(), self.clean.max()
        self.clean = (self.clean - gt_min) / (gt_max - gt_min + 1e-8)
        self.noisy = (self.noisy - gt_min) / (gt_max - gt_min + 1e-8)

        self.H, self.W, self.C = self.clean.shape
        self.grid_h = self.H // patch_size
        self.grid_w = self.W // patch_size
        self.n_patches = self.grid_h * self.grid_w

        print(f"Loaded test data for sigma={sigma}, shape: {self.clean.shape}, "
              f"{self.n_patches} patches")

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        h = (idx // self.grid_w) * self.patch_size
        w = (idx % self.grid_w) * self.patch_size

        clean = self.clean[h:h + self.patch_size, w:w + self.patch_size, :]
        noisy = self.noisy[h:h + self.patch_size, w:w + self.patch_size, :]

        # Convert to tensor [C, H, W]
        clean = torch.from_numpy(clean.transpose(2, 0, 1)).float()
        noisy = torch.from_numpy(noisy.transpose(2, 0, 1)).float()

        return noisy, clean

    def get_full_images(self):
        """Return full images for whole-image evaluation."""
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


def evaluate_wdc_per_sigma(model, data_dir, patch_size):
    """Evaluate WDC model on each sigma level separately."""
    sigma_levels = [10, 30, 50, 70]
    all_metrics = {}

    for sigma in sigma_levels:
        print(f"\nEvaluating sigma={sigma}...")
        test_dataset = WDCTestDataset(data_dir, sigma=sigma, patch_size=patch_size)
        test_loader = Data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        metrics = evaluate(model, test_loader)
        all_metrics[f'sigma_{sigma}'] = metrics
        print(f"  PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.4f}, SAM={metrics['SAM']:.2f}")

    # Compute average across sigma levels
    avg_metrics = {
        'PSNR': round(np.mean([m['PSNR'] for m in all_metrics.values()]), 4),
        'SSIM': round(np.mean([m['SSIM'] for m in all_metrics.values()]), 4),
        'SAM': round(np.mean([m['SAM'] for m in all_metrics.values()]), 4),
    }
    all_metrics['average'] = avg_metrics

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='HyperSIGMA Denoising')
    parser.add_argument('--dataset', type=str, default='IndianPines',
                        help='Dataset name (IndianPines, PaviaU, or WDC)')
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

    # Handle WDC dataset separately
    if args.dataset.lower() == 'wdc':
        print("Loading WDC data...")
        train_dataset = WDCTrainDataset(args.data_dir, sigma_range=(args.sigma_min, args.sigma_max))
        patch_size = 64  # WDC patches are 64x64
        C = train_dataset.n_channels  # 191 bands

        train_loader = Data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )

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

        # Loss and optimizer with Layer Decay (matching original HyperSIGMA implementation)
        criterion = nn.L1Loss().cuda()
        optim_wrapper = dict(
            optimizer=dict(type='AdamW', lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05),
            constructor='LayerDecayOptimizerConstructor_ViT',
            paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9)
        )
        optimizer = build_optim_wrapper(model, optim_wrapper)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, args.epochs, eta_min=0)

        # Training
        print("\nStarting training...")
        t0 = time.time()
        best_psnr = 0
        best_metrics = None

        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                # Evaluate on sigma 30 during training as proxy
                test_dataset = WDCTestDataset(args.data_dir, sigma=30, patch_size=patch_size)
                test_loader = Data.DataLoader(test_dataset, batch_size=1, shuffle=False)
                metrics = evaluate(model, test_loader)
                print(f"Epoch {epoch+1}/{args.epochs}: loss={train_loss:.6f}, "
                      f"PSNR(σ=30)={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.4f}, SAM={metrics['SAM']:.2f}")

                if metrics['PSNR'] > best_psnr:
                    best_psnr = metrics['PSNR']
                    # Model saving disabled to save disk space

        train_time = time.time() - t0
        print(f"\nTraining completed in {train_time:.1f}s")

        # Load best model for final evaluation
        if os.path.exists(os.path.join(args.output_dir, 'model_wdc_best.pth')):
            model.load_state_dict(torch.load(os.path.join(args.output_dir, 'model_wdc_best.pth')))
            print("Loaded best model for final evaluation")

        # Final evaluation on all sigma levels
        print("\n" + "="*60)
        print("Final Evaluation (per sigma level)")
        print("="*60)
        all_sigma_metrics = evaluate_wdc_per_sigma(model, args.data_dir, patch_size)

        print("\n" + "-"*40)
        print("Average across all sigma levels:")
        avg = all_sigma_metrics['average']
        print(f"  PSNR: {avg['PSNR']:.2f}")
        print(f"  SSIM: {avg['SSIM']:.4f}")
        print(f"  SAM: {avg['SAM']:.2f}")

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)

        result = {
            'task': 'denoising',
            'dataset': 'WDC',
            'model': 'HyperSIGMA',
            'seed': args.seed,
            'metrics_per_sigma': all_sigma_metrics,
            'metrics': all_sigma_metrics['average'],
            'config': {
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
                'seed': args.seed,
            },
            'timestamp': datetime.now().isoformat(),
        }

        result_file = os.path.join(args.output_dir, f'result_wdc_seed{args.seed}.json')
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {result_file}")

        # Model saving disabled to save disk space

        return

    # Original code for IndianPines/PaviaU
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
        raise ValueError(f"Unknown dataset: {args.dataset}. Supported: IndianPines, PaviaU, WDC")

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

    # Loss and optimizer with Layer Decay (matching original HyperSIGMA implementation)
    criterion = nn.L1Loss().cuda()
    optim_wrapper = dict(
        optimizer=dict(type='AdamW', lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05),
        constructor='LayerDecayOptimizerConstructor_ViT',
        paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9)
    )
    optimizer = build_optim_wrapper(model, optim_wrapper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, args.epochs, eta_min=0)

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

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    result = {
        'task': 'denoising',
        'dataset': args.dataset,
        'model': 'HyperSIGMA',
        'seed': args.seed,
        'metrics': final_metrics,
        'best_metrics': best_metrics if best_metrics is not None else final_metrics,
        'config': {
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
            'seed': args.seed,
        },
        'timestamp': datetime.now().isoformat(),
    }

    result_file = os.path.join(args.output_dir, f'result_{args.dataset.lower()}_seed{args.seed}.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_file}")

    # Model saving disabled to save disk space


if __name__ == '__main__':
    main()
