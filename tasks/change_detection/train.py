"""
HyperSIGMA Change Detection Training Script.

Trains change detection models on bi-temporal hyperspectral data.
Supports: Hermiston, BayArea, Barbara, Farmland datasets.
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
from sklearn.metrics import confusion_matrix

# Add parent directories to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HYPERSIGMA_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
PROJECT_ROOT = os.path.join(HYPERSIGMA_ROOT, '..')
sys.path.insert(0, HYPERSIGMA_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from downstream_task_head.utils.result_manager import ResultManager, ChangeDetectionMetrics

# Default data directory (relative to hypersigma-benchmark root)
DEFAULT_DATA_DIR = os.path.join(HYPERSIGMA_ROOT, 'data', 'change_detection')

from hypersigma.models.task_heads import ChangeDetectionHead, SSChangeDetectionHead


def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


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


def get_dataset_info(dataset):
    """Get dataset-specific parameters."""
    info = {
        'Hermiston': {'channels': 242, 'patch_size': 5, 'seg_patches': 1},
        'Farmland': {'channels': 155, 'patch_size': 5, 'seg_patches': 1},
        'BayArea': {'channels': 224, 'patch_size': 15, 'seg_patches': 2},
        'Barbara': {'channels': 224, 'patch_size': 15, 'seg_patches': 2},
    }
    if dataset not in info:
        raise ValueError(f"Unknown dataset: {dataset}")
    return info[dataset]


def load_hermiston_data(data_dir):
    """Load Hermiston dataset."""
    hermiston_dir = os.path.join(data_dir, 'Hermiston')
    t1 = sio.loadmat(os.path.join(hermiston_dir, 'hermiston2004.mat'))['HypeRvieW']
    t2 = sio.loadmat(os.path.join(hermiston_dir, 'hermiston2007.mat'))['HypeRvieW']
    gt = sio.loadmat(os.path.join(hermiston_dir, 'rdChangesHermiston_5classes.mat'))['gt5clasesHermiston']

    # Convert to binary: 0=unchanged, 1=changed (classes 1-5)
    binary_gt = (gt > 0).astype(np.int32)

    # Get positions
    uc_position = np.array(np.where(binary_gt == 0)).T  # unchanged
    c_position = np.array(np.where(binary_gt == 1)).T   # changed

    return t1, t2, binary_gt, uc_position, c_position


def load_bayarea_data(data_dir):
    """Load Bay Area dataset."""
    t1 = sio.loadmat(os.path.join(data_dir, 'Bay_Area_2013.mat'))['HypeRvieW']
    t2 = sio.loadmat(os.path.join(data_dir, 'Bay_Area_2015.mat'))['HypeRvieW']
    gt = sio.loadmat(os.path.join(data_dir, 'bayArea_gtChanges2.mat.mat'))['HypeRvieW']

    # GT: 0=unknown, 1=changed, 2=unchanged
    # Convert: 0=unchanged (was 2), 1=changed (was 1)
    binary_gt = np.zeros_like(gt)
    binary_gt[gt == 1] = 1  # changed
    binary_gt[gt == 2] = 0  # unchanged

    # Get labeled positions only (exclude unknown class 0)
    labeled_mask = gt > 0
    uc_position = np.array(np.where((gt == 2) & labeled_mask)).T
    c_position = np.array(np.where((gt == 1) & labeled_mask)).T

    return t1, t2, gt, uc_position, c_position


def load_data(dataset, data_dir):
    """Load dataset based on name."""
    if dataset == 'Hermiston':
        return load_hermiston_data(data_dir)
    elif dataset == 'BayArea':
        # Try multiple path patterns
        possible_paths = [
            os.path.join(data_dir, 'BayArea', 'mat'),
            os.path.join(data_dir, 'bayArea', 'mat'),
            data_dir,
        ]
        for path in possible_paths:
            if os.path.exists(os.path.join(path, 'Bay_Area_2013.mat')):
                return load_bayarea_data(path)
        raise ValueError(f"BayArea data not found in {data_dir}")
    else:
        raise ValueError(f"Dataset {dataset} not yet supported")


def mirror_hsi(height, width, band, data, patch):
    """Mirror pad HSI for patch extraction."""
    padding = patch // 2
    mirror = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    mirror[padding:padding + height, padding:padding + width, :] = data

    # Left mirror
    for i in range(padding):
        mirror[padding:height + padding, i, :] = data[:, padding - i - 1, :]
    # Right mirror
    for i in range(padding):
        mirror[padding:height + padding, width + padding + i, :] = data[:, width - 1 - i, :]
    # Top mirror
    for i in range(padding):
        mirror[i, :, :] = mirror[padding * 2 - i - 1, :, :]
    # Bottom mirror
    for i in range(padding):
        mirror[height + padding + i, :, :] = mirror[height + padding - 1 - i, :, :]

    return mirror


def extract_patches(mirror_image, positions, patch_size):
    """Extract patches at given positions."""
    n_samples = positions.shape[0]
    band = mirror_image.shape[2]
    patches = np.zeros((n_samples, patch_size, patch_size, band), dtype=float)

    for i in range(n_samples):
        x, y = positions[i, 0], positions[i, 1]
        patches[i] = mirror_image[x:x + patch_size, y:y + patch_size, :]

    return patches


def prepare_data(data_t1, data_t2, gt, uc_position, c_position, train_number, patch_size, seed=1):
    """Prepare training and testing data."""
    setup_seed(seed)

    height, width, band = data_t1.shape

    # Normalize per band
    t1_norm = np.zeros_like(data_t1, dtype=float)
    t2_norm = np.zeros_like(data_t2, dtype=float)
    for i in range(band):
        input_max = max(np.max(data_t1[:, :, i]), np.max(data_t2[:, :, i]))
        input_min = min(np.min(data_t1[:, :, i]), np.min(data_t2[:, :, i]))
        if input_max > input_min:
            t1_norm[:, :, i] = (data_t1[:, :, i] - input_min) / (input_max - input_min)
            t2_norm[:, :, i] = (data_t2[:, :, i] - input_min) / (input_max - input_min)

    # Select training samples
    selected_uc = np.random.choice(uc_position.shape[0], min(train_number, uc_position.shape[0]), replace=False)
    selected_c = np.random.choice(c_position.shape[0], min(train_number, c_position.shape[0]), replace=False)

    train_pos = np.vstack([c_position[selected_c], uc_position[selected_uc]])
    train_labels = np.array([0] * len(selected_c) + [1] * len(selected_uc))  # 0=changed, 1=unchanged

    # Mirror padding
    mirror_t1 = mirror_hsi(height, width, band, t1_norm, patch_size)
    mirror_t2 = mirror_hsi(height, width, band, t2_norm, patch_size)

    # Extract training patches
    x_train_t1 = extract_patches(mirror_t1, train_pos, patch_size)
    x_train_t2 = extract_patches(mirror_t2, train_pos, patch_size)

    # Test on all labeled pixels
    all_pos = np.vstack([c_position, uc_position])
    all_labels = np.array([0] * len(c_position) + [1] * len(uc_position))

    x_test_t1 = extract_patches(mirror_t1, all_pos, patch_size)
    x_test_t2 = extract_patches(mirror_t2, all_pos, patch_size)

    return (x_train_t1, x_train_t2, train_labels, train_pos,
            x_test_t1, x_test_t2, all_labels, all_pos)


def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    losses = AvgMeter()
    accs = AvgMeter()

    for batch_t1, batch_t2, batch_target in train_loader:
        batch_t1 = batch_t1.cuda()
        batch_t2 = batch_t2.cuda()
        batch_target = batch_target.cuda()

        optimizer.zero_grad()
        output = model(batch_t1, batch_t2)
        loss = criterion(output, batch_target)
        loss.backward()
        optimizer.step()

        # Accuracy
        pred = output.argmax(dim=1) if output.dim() > 1 else (output > 0).long()
        acc = (pred == batch_target).float().mean()

        n = batch_t1.size(0)
        losses.update(loss.item(), n)
        accs.update(acc.item(), n)

    return losses.avg, accs.avg


def evaluate(model, test_loader, gt_shape, test_positions, dataset):
    """Evaluate model and compute metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_t1, batch_t2, batch_target in test_loader:
            batch_t1 = batch_t1.cuda()
            batch_t2 = batch_t2.cuda()

            output = model(batch_t1, batch_t2)
            pred = output.argmax(dim=1) if output.dim() > 1 else (output > 0).long()

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch_target.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute metrics
    # For binary change detection: 0=changed, 1=unchanged
    # TP = correctly predicted as changed (pred=0, target=0)
    tp = np.sum((all_preds == 0) & (all_targets == 0))
    tn = np.sum((all_preds == 1) & (all_targets == 1))
    fp = np.sum((all_preds == 0) & (all_targets == 1))
    fn = np.sum((all_preds == 1) & (all_targets == 0))

    total = len(all_preds)
    oa = (tp + tn) / total if total > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Kappa
    pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (total * total) if total > 0 else 0
    kappa = (oa - pe) / (1 - pe) if (1 - pe) > 0 else 0

    metrics = {
        'OA': round(oa, 4),
        'kappa': round(kappa, 4),
        'F1': round(f1, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
    }

    return metrics, all_preds


def main():
    parser = argparse.ArgumentParser(description='HyperSIGMA Change Detection')
    parser.add_argument('--dataset', type=str, default='Hermiston',
                        choices=['Hermiston', 'BayArea', 'Barbara', 'Farmland'])
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--mode', type=str, default='ss', choices=['sa', 'ss'],
                        help='sa=spatial-only, ss=spectral-spatial')
    parser.add_argument('--spat_weights', type=str,
                        default='pretrained/spat-vit-base-ultra-checkpoint-1599.pth')
    parser.add_argument('--spec_weights', type=str,
                        default='pretrained/spec-vit-base-ultra-checkpoint-1599.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--train_number', type=int, default=500)
    parser.add_argument('--num_tokens', type=int, default=144)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/change_detection')
    args = parser.parse_args()

    setup_seed(args.seed)

    # Get dataset info
    ds_info = get_dataset_info(args.dataset)
    patch_size = ds_info['patch_size']
    seg_patches = ds_info['seg_patches']
    channels = ds_info['channels']

    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"Channels: {channels}, Patch size: {patch_size}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    data_t1, data_t2, gt, uc_pos, c_pos = load_data(args.dataset, args.data_dir)
    print(f"Data shape: T1={data_t1.shape}, T2={data_t2.shape}")
    print(f"Changed pixels: {len(c_pos)}, Unchanged pixels: {len(uc_pos)}")

    # Prepare data
    print("\nPreparing data...")
    (x_train_t1, x_train_t2, y_train, train_pos,
     x_test_t1, x_test_t2, y_test, test_pos) = prepare_data(
        data_t1, data_t2, gt, uc_pos, c_pos, args.train_number, patch_size, args.seed
    )
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")

    # Create data loaders
    x_train_t1 = torch.from_numpy(x_train_t1.transpose(0, 3, 1, 2)).float()
    x_train_t2 = torch.from_numpy(x_train_t2.transpose(0, 3, 1, 2)).float()
    y_train = torch.from_numpy(y_train).long()

    x_test_t1 = torch.from_numpy(x_test_t1.transpose(0, 3, 1, 2)).float()
    x_test_t2 = torch.from_numpy(x_test_t2.transpose(0, 3, 1, 2)).float()
    y_test_tensor = torch.from_numpy(y_test).long()

    train_dataset = Data.TensorDataset(x_train_t1, x_train_t2, y_train)
    test_dataset = Data.TensorDataset(x_test_t1, x_test_t2, y_test_tensor)

    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    print("\nCreating model...")
    if args.mode == 'sa':
        model = ChangeDetectionHead(
            img_size=patch_size,
            in_channels=channels,
            patch_size=patch_size,
            seg_patches=seg_patches,
            spat_weights=args.spat_weights,
        )
    else:
        model = SSChangeDetectionHead(
            img_size=patch_size,
            in_channels=channels,
            patch_size=patch_size,
            seg_patches=seg_patches,
            spat_weights=args.spat_weights,
            spec_weights=args.spec_weights,
            num_tokens=args.num_tokens,
        )

    model = model.cuda()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Training
    print("\nStarting training...")
    t0 = time.time()

    for epoch in range(args.epochs):
        scheduler.step()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: loss={train_loss:.4f}, acc={train_acc:.4f}")

    train_time = time.time() - t0
    print(f"\nTraining completed in {train_time:.1f}s")

    # Evaluation
    print("\nEvaluating...")
    metrics, predictions = evaluate(model, test_loader, gt.shape, test_pos, args.dataset)

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save results using ResultManager
    checkpoint_name = f"hypersigma_{args.mode}"
    manager = ResultManager(
        task="change_detection",
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
        'train_number': args.train_number,
        'patch_size': patch_size,
        'model': 'HyperSIGMA' if args.mode == 'ss' else 'SpatSIGMA',
    })

    manager.log_run(
        seed=args.seed,
        metrics=ChangeDetectionMetrics(
            overall_accuracy=metrics['OA'],
            kappa=metrics['kappa'],
            F1=metrics['F1'],
            precision=metrics['precision'],
            recall=metrics['recall'],
        )
    )
    manager.try_auto_aggregate()


if __name__ == '__main__':
    main()
