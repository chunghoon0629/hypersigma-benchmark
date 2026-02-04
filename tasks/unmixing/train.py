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


def SAD_loss(pred, target):
    """Spectral Angle Distance loss."""
    dot_product = (pred * target).sum(dim=-1)
    pred_norm = torch.norm(pred, dim=-1) + 1e-8
    target_norm = torch.norm(target, dim=-1) + 1e-8
    cos_angle = dot_product / (pred_norm * target_norm)
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle = torch.acos(cos_angle)
    return angle.mean()


def sparsity_loss(abundance, alpha=0.35):
    """Sparsity regularization on abundance maps.

    Encourages sparse abundance maps by penalizing the square root of abundances.

    Args:
        abundance: Abundance tensor [B, num_end, H, W] or [B, num_end]
        alpha: Sparsity weight (default: 0.35)

    Returns:
        Sparsity loss scalar
    """
    return alpha * torch.sqrt(abundance + 1e-8).mean()


def tv_loss(endmember, beta=0.1):
    """Total Variation loss on endmember spectra.

    Encourages smoothness in the spectral dimension of endmembers.

    Args:
        endmember: Endmember tensor [num_end, C] or [B, num_end, C]
        beta: TV weight (default: 0.1)

    Returns:
        TV loss scalar
    """
    # Handle different input shapes
    if endmember.dim() == 2:
        # [num_end, C]
        return beta * torch.abs(endmember[:, 1:] - endmember[:, :-1]).mean()
    else:
        # [B, num_end, C]
        return beta * torch.abs(endmember[:, :, 1:] - endmember[:, :, :-1]).mean()


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


def vca(data, num_endmembers, seed=None):
    """
    Vertex Component Analysis for endmember extraction.

    VCA is a geometric approach that finds pure pixels (endmembers) by
    iteratively projecting data onto directions orthogonal to previously
    selected endmembers and finding extreme points.

    Args:
        data: [N, C] spectral data (N pixels, C bands)
        num_endmembers: number of endmembers to extract
        seed: random seed for reproducibility

    Returns:
        endmembers: [num_endmembers, C] extracted endmembers
        indices: indices of selected pixels
    """
    if seed is not None:
        np.random.seed(seed)

    N, C = data.shape

    # 1. Dimensionality reduction via SVD
    data_centered = data - data.mean(axis=0)
    U, S, Vt = np.linalg.svd(data_centered.T, full_matrices=False)

    # Project to num_endmembers-1 dimensions (affine subspace)
    num_proj = min(num_endmembers - 1, C - 1, N - 1)
    if num_proj < 1:
        num_proj = 1
    proj = U[:, :num_proj].T @ data_centered.T  # [num_proj, N]

    # 2. Iteratively find vertices (extreme points)
    indices = []

    # Random initial direction
    w = np.random.randn(num_proj)
    w /= (np.linalg.norm(w) + 1e-10)

    for i in range(num_endmembers):
        # Project data onto direction w
        projections = proj.T @ w  # [N]

        # Find extreme point (furthest from origin in direction w)
        idx = np.argmax(np.abs(projections))
        indices.append(idx)

        # Update direction (orthogonal to selected points)
        if i < num_endmembers - 1:
            selected = proj[:, indices]  # [num_proj, i+1]
            # Gram-Schmidt orthogonalization: find direction orthogonal to selected
            w = np.random.randn(num_proj)
            for j in range(len(indices)):
                v = selected[:, j]
                v_norm = np.dot(v, v) + 1e-10
                w = w - (np.dot(w, v) / v_norm) * v
            norm_w = np.linalg.norm(w)
            if norm_w > 1e-10:
                w /= norm_w
            else:
                # If w becomes zero, use random direction
                w = np.random.randn(num_proj)
                w /= (np.linalg.norm(w) + 1e-10)

    endmembers = data[indices]
    return endmembers, indices


class UnmixingDataset(Data.Dataset):
    """Dataset for spectral unmixing with patch extraction."""

    def __init__(self, hsi, abundances, patch_size=7, mode='train',
                 train_ratio=0.8, seed=42, augment=False):
        """
        Args:
            hsi: HSI cube (H, W, C)
            abundances: Ground truth abundances (num_end, H*W) or (H, W, num_end)
            patch_size: Size of patches to extract
            mode: 'train' or 'test'
            train_ratio: Ratio of data for training
            seed: Random seed for train/test split
            augment: Whether to apply data augmentation (random flips)
        """
        self.patch_size = patch_size
        self.half_size = patch_size // 2
        self.augment = augment

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

        aug_str = " (with augmentation)" if augment else ""
        print(f"{mode} set: {len(self.positions)} samples, {self.num_endmembers} endmembers{aug_str}")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        h, w = self.positions[idx]

        # Extract patch
        patch = self.hsi[
            h - self.half_size:h + self.half_size + 1,
            w - self.half_size:w + self.half_size + 1,
            :
        ].copy()

        # Get abundance patch (for augmentation consistency)
        abundance_patch = self.abundances[
            h - self.half_size:h + self.half_size + 1,
            w - self.half_size:w + self.half_size + 1,
            :
        ].copy()

        # Apply augmentation (training only)
        if self.augment:
            # Random Horizontal Flip
            if np.random.random() > 0.5:
                patch = np.flip(patch, axis=1).copy()
                abundance_patch = np.flip(abundance_patch, axis=1).copy()

            # Random Vertical Flip
            if np.random.random() > 0.5:
                patch = np.flip(patch, axis=0).copy()
                abundance_patch = np.flip(abundance_patch, axis=0).copy()

        # Get center pixel abundance and spectrum (after augmentation)
        center_h = self.half_size
        center_w = self.half_size
        abundance = abundance_patch[center_h, center_w, :]
        center_spectrum = patch[center_h, center_w, :]

        # Convert to tensors [C, H, W] for patch
        patch = torch.from_numpy(patch.transpose(2, 0, 1)).float()
        abundance = torch.from_numpy(abundance.astype(np.float32))
        center_spectrum = torch.from_numpy(center_spectrum).float()

        return patch, abundance, center_spectrum


def train_epoch(model, train_loader, optimizer, sparsity_alpha=0.35, tv_beta=0.1):
    """Train for one epoch.

    Loss = SAD_loss + sparsity_loss + TV_loss
    Following the original HyperSIGMA unmixing implementation.
    """
    model.train()
    sad_losses = AvgMeter()
    sparse_losses = AvgMeter()
    tv_losses = AvgMeter()
    total_losses = AvgMeter()

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

        # Compute losses following original HyperSIGMA formula:
        # Loss = SAD_loss + α × √(Abundance).mean() + β × TV(Endmember)

        # 1. SAD loss for reconstruction
        if reconstructed is not None:
            center_recon = reconstructed[:, :, center_h, center_w]  # [B, C]
            sad_loss_val = SAD_loss(center_recon, center_spectra)
        else:
            # If no reconstruction, use MSE on abundance as fallback
            sad_loss_val = F.mse_loss(pred_center_abund, abundances_gt)

        # 2. Sparsity loss on abundances
        sparse_loss_val = sparsity_loss(pred_center_abund, alpha=sparsity_alpha)

        # 3. TV loss on endmembers (if model has endmember weights)
        tv_loss_val = torch.tensor(0.0, device=patches.device)
        if hasattr(model, 'endmember') and model.endmember is not None:
            tv_loss_val = tv_loss(model.endmember, beta=tv_beta)
        elif hasattr(model, 'decoder') and hasattr(model.decoder, 'weight'):
            # Decoder weight can be interpreted as endmember matrix
            tv_loss_val = tv_loss(model.decoder.weight, beta=tv_beta)

        # Total loss
        loss = sad_loss_val + sparse_loss_val + tv_loss_val

        loss.backward()
        optimizer.step()

        # Update meters
        n = patches.size(0)
        sad_losses.update(sad_loss_val.item(), n)
        sparse_losses.update(sparse_loss_val.item(), n)
        tv_losses.update(tv_loss_val.item(), n)
        total_losses.update(loss.item(), n)

    return total_losses.avg, sad_losses.avg, sparse_losses.avg, tv_losses.avg


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
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--spat_patch_size', type=int, default=2,
                        help='Spatial ViT patch size')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--num_tokens', type=int, default=64)
    parser.add_argument('--sparsity_alpha', type=float, default=0.35,
                        help='Sparsity loss weight')
    parser.add_argument('--tv_beta', type=float, default=0.1,
                        help='Total Variation loss weight')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/unmixing')
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation (random horizontal/vertical flips)')
    parser.add_argument('--no_vca', action='store_true',
                        help='Disable VCA initialization for endmembers')
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

    # Extract initial endmembers using VCA
    vca_endmembers = None
    if not args.no_vca:
        print("\nExtracting initial endmembers using VCA...")
        # Normalize HSI for VCA (same normalization as dataset)
        hsi_norm = hsi.astype(np.float32)
        hsi_min, hsi_max = hsi_norm.min(), hsi_norm.max()
        hsi_norm = (hsi_norm - hsi_min) / (hsi_max - hsi_min + 1e-8)
        flat_data = hsi_norm.reshape(-1, hsi_norm.shape[-1])  # [H*W, C]
        vca_endmembers, vca_indices = vca(flat_data, num_end, seed=args.seed)
        print(f"VCA extracted {num_end} endmembers, shape: {vca_endmembers.shape}")
    else:
        print("\nVCA initialization disabled, using random initialization")

    # Create datasets
    train_dataset = UnmixingDataset(
        hsi, abundances,
        patch_size=args.patch_size, mode='train',
        train_ratio=args.train_ratio, seed=args.seed,
        augment=args.augment
    )
    test_dataset = UnmixingDataset(
        hsi, abundances,
        patch_size=args.patch_size, mode='test',
        train_ratio=args.train_ratio, seed=args.seed,
        augment=False  # Never augment test set
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
            init_endmembers=vca_endmembers,
        )
    else:
        model = UnmixingHead(
            img_size=args.patch_size,
            in_channels=C,
            num_endmembers=num_end,
            spat_weights=args.spat_weights,
            init_endmembers=vca_endmembers,
        )

    model = model.cuda()

    # Optimizer with Layer Decay (matching original HyperSIGMA implementation)
    optim_wrapper = dict(
        optimizer=dict(type='AdamW', lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay),
        constructor='LayerDecayOptimizerConstructor_ViT',
        paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9)
    )
    optimizer = build_optim_wrapper(model, optim_wrapper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, args.epochs, eta_min=0)

    # Training
    print("\nStarting training...")
    print(f"Using loss: SAD + sparsity(α={args.sparsity_alpha}) + TV(β={args.tv_beta})")
    t0 = time.time()
    best_rmse = float('inf')
    best_metrics = None

    for epoch in range(args.epochs):
        total_loss, sad_loss, sparse_loss, tv_loss_val = train_epoch(
            model, train_loader, optimizer,
            sparsity_alpha=args.sparsity_alpha,
            tv_beta=args.tv_beta
        )
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            metrics = evaluate(model, test_loader)
            print(f"Epoch {epoch+1}/{args.epochs}: loss={total_loss:.6f} "
                  f"(SAD={sad_loss:.4f}, sparse={sparse_loss:.4f}, TV={tv_loss_val:.4f}), "
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
            'weight_decay': args.weight_decay,
            'patch_size': args.patch_size,
            'spat_patch_size': args.spat_patch_size,
            'train_ratio': args.train_ratio,
            'num_tokens': args.num_tokens if args.mode == 'ss' else None,
            'sparsity_alpha': args.sparsity_alpha,
            'tv_beta': args.tv_beta,
            'seed': args.seed,
            'augment': args.augment,
            'vca_init': not args.no_vca,
        },
        'timestamp': datetime.now().isoformat(),
    }

    result_file = os.path.join(mode_dir, f'result_{args.dataset.lower()}_{args.mode}_seed{args.seed}.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_file}")


if __name__ == '__main__':
    main()
