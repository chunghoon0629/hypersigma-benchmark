#!/usr/bin/env python
"""
Anomaly Detection Training Script for HyperSIGMA Benchmark.

Usage:
    python train.py --dataset pavia --mode ss --epochs 10
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import scipy.io as scio
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HYPERSIGMA_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
PROJECT_ROOT = os.path.join(HYPERSIGMA_ROOT, '..')
sys.path.insert(0, HYPERSIGMA_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from downstream_task_head.utils.result_manager import ResultManager, AnomalyDetectionMetrics

# Default data directory (project benchmark data)
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'benchmark', 'anomaly_detection', 'Pavia')

from hypersigma.models.task_heads import AnomalyDetectionHead, SSAnomalyDetectionHead
from hypersigma.utils.metrics import compute_auc_scores
from hypersigma.mmcv_custom import LayerDecayOptimizerConstructor_ViT

# Import mmengine for optimizer
from mmengine.optim import build_optim_wrapper


def standard(x: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] range."""
    max_value = np.max(x)
    min_value = np.min(x)
    if max_value == min_value:
        return x
    return (x - min_value) / (max_value - min_value)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_data(dataset: str, data_dir: str):
    """Load anomaly detection dataset."""
    dataset_lower = dataset.lower()

    if dataset_lower == 'pavia':
        # Try benchmark path first
        benchmark_path = os.path.join(data_dir, 'gt_had', 'pavia.mat')
        old_path = os.path.join(data_dir, 'Pavia_150_150_102.mat')

        if os.path.exists(benchmark_path):
            mat = scio.loadmat(benchmark_path)
            # Benchmark format: 'data' and 'map'
            if 'data' in mat:
                hsi = mat['data']
            else:
                hsi = mat['hsi']
            if 'map' in mat:
                hsi_gt = mat['map']
            else:
                hsi_gt = mat['hsi_gt']
        elif os.path.exists(old_path):
            mat = scio.loadmat(old_path)
            hsi = mat['hsi']
            hsi_gt = mat['hsi_gt']
        else:
            raise FileNotFoundError(f"Pavia data not found in {data_dir}")

        # Load or create coarse detection map
        coarse_path = os.path.join(data_dir, 'Pavia_coarse_det_map.mat')
        if os.path.exists(coarse_path):
            coarse_det_dict = scio.loadmat(coarse_path)
            coarse_det = coarse_det_dict['show']
        else:
            # Generate coarse detection using RX detector approximation
            print("Generating coarse detection map...")
            from sklearn.covariance import EmpiricalCovariance
            r, c, d = hsi.shape
            X = hsi.reshape(-1, d).astype(np.float64)
            cov = EmpiricalCovariance().fit(X)
            mahal = cov.mahalanobis(X)
            coarse_det = mahal.reshape(r, c)

    elif dataset_lower == 'cri':
        mat = scio.loadmat(os.path.join(data_dir, 'Cri.mat'))
        # Handle different key names
        if 'hsi' in mat:
            hsi = mat['hsi']
        elif 'data' in mat:
            hsi = mat['data']
        else:
            hsi = mat['Y']
        if 'hsi_gt' in mat:
            hsi_gt = mat['hsi_gt']
        elif 'map' in mat:
            hsi_gt = mat['map']
        else:
            hsi_gt = mat['groundT']

        # Load or create coarse detection map
        coarse_path = os.path.join(data_dir, 'Cri_coarse_det_map.mat')
        if os.path.exists(coarse_path):
            coarse_det_dict = scio.loadmat(coarse_path)
            coarse_det = coarse_det_dict['show']
        else:
            from sklearn.covariance import EmpiricalCovariance
            r, c, d = hsi.shape
            X = hsi.reshape(-1, d).astype(np.float64)
            cov = EmpiricalCovariance().fit(X)
            mahal = cov.mahalanobis(X)
            coarse_det = mahal.reshape(r, c)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"HSI shape: {hsi.shape}")
    print(f"GT shape: {hsi_gt.shape}")
    print(f"Anomaly count: {np.sum(hsi_gt)}")

    r, c, d = hsi.shape
    original = hsi.reshape(r * c, d)
    gt = hsi_gt.reshape(r * c, 1)

    rows = np.arange(gt.shape[0])
    all_data = np.c_[rows, original, gt]
    labeled_data = all_data[all_data[:, -1] != 0, :]
    rows_num = labeled_data[:, 0]

    return all_data, labeled_data, rows_num, coarse_det, r, c, dataset


def train_epoch(args, epoch, net, optimizer, scheduler, trn_loader, criterion):
    """Train for one epoch."""
    net.train()
    loss_meter = AverageMeter()

    for idx, (X_data, y_target) in enumerate(trn_loader):
        X_data = Variable(X_data.float()).cuda(non_blocking=True)
        y_target = Variable(y_target.long()).cuda(non_blocking=True)

        pred_prob = net.forward(X_data)
        loss = criterion(pred_prob, y_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = X_data.size(0)
        loss_meter.update(loss.item(), n)

        del X_data, y_target

        scheduler.step()

        if (idx + 1) % args.print_freq == 0:
            print(f'Epoch: [{epoch + 1}/{args.epochs}][{idx + 1}/{len(trn_loader)}], '
                  f'Batch Loss {loss_meter.val:.4f}')

    print(f'Training epoch [{epoch + 1}/{args.epochs}]: Loss {loss_meter.avg:.4f}')
    return loss_meter.avg


def validate(args, epoch, net, val_loader):
    """Validate the model."""
    print('>>>>>>>>>>>>>>>> Start Evaluation <<<<<<<<<<<<<<<<<<')
    net.eval()
    auc_meter = AverageMeter()

    for idx, (X_data, y_target) in enumerate(val_loader):
        with torch.no_grad():
            X_data = Variable(X_data.float()).cuda(non_blocking=True)
            y_target = Variable(y_target.float().long()).cuda(non_blocking=True)
            prob = net.forward(X_data)

            y_pred_prob = standard(prob[:, 1, :, :].cpu().numpy())
            y_pred_prob = np.clip(y_pred_prob, 0, 1)

        aucs = compute_auc_scores(y_pred_prob, y_target.cpu().numpy())
        n = X_data.size(0)
        auc_meter.update(aucs['auc_roc'], n)

        if (idx + 1) % args.print_freq == 0:
            print(f'Epoch: [{epoch + 1}/{args.epochs}][{idx + 1}/{len(val_loader)}], '
                  f'AUC {auc_meter.val:.4f}')

    print(f'Validation epoch [{epoch + 1}/{args.epochs}]: Avg AUC {auc_meter.avg:.4f}')
    print('>>>>>>>>>>>>>>>> End Evaluation <<<<<<<<<<<<<<<<<<')
    return auc_meter.avg


def main():
    parser = argparse.ArgumentParser(description="HyperSIGMA Anomaly Detection")

    # Dataset
    parser.add_argument('--dataset', type=str, default='pavia', choices=['cri', 'pavia'])
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)

    # Model
    parser.add_argument('--mode', type=str, default='ss', choices=['sa', 'ss'],
                        help='sa: spatial-only, ss: spectral-spatial')

    # Weights
    parser.add_argument('--spat_weights', type=str,
                        default='pretrained/spat-vit-base-ultra-checkpoint-1599.pth')
    parser.add_argument('--spec_weights', type=str,
                        default='pretrained/spec-vit-base-ultra-checkpoint-1599.pth')

    # Normalization
    parser.add_argument('--norm', type=str, default='std', choices=['std', 'norm'])
    parser.add_argument('--mi', type=int, default=-1)
    parser.add_argument('--ma', type=int, default=1)

    # Input
    parser.add_argument('--input_mode', type=str, default='part', choices=['whole', 'part'])
    parser.add_argument('--input_size', nargs='+', default=[32, 32], type=int)
    parser.add_argument('--overlap_size', type=int, default=16)

    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--wd', type=float, default=5e-4)

    # Experiment
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_freq', type=int, default=3)
    parser.add_argument('--val_freq', type=int, default=3)

    # Output
    parser.add_argument('--output_dir', type=str, default='results/anomaly_detection')

    # Misc
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--ignore_label', type=int, default=255)

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    save_path = os.path.join(args.output_dir, args.mode)
    os.makedirs(save_path, exist_ok=True)

    # Load data
    all_data, labeled_data, rows_num, coarse_det, r, c, flag = load_data(
        args.dataset, args.data_dir
    )
    print('Data loaded successfully!')

    # Normalize
    if args.norm == 'norm':
        scaler = MinMaxScaler(feature_range=(args.mi, args.ma))
    else:
        scaler = StandardScaler()
    all_data_norm = scaler.fit_transform(all_data[:, 1:-1])
    print('Image normalization complete!')

    data_tube = all_data_norm.reshape(r, c, -1)
    gt_tube = all_data[:, -1].reshape(r, c)
    d = data_tube.shape[-1]

    # Prepare training map
    coarse_det_1d = coarse_det.reshape(-1)
    y_trn_map = (255 * np.ones([r, c])).astype('uint8')
    all_idxs = np.argsort(coarse_det_1d)
    bg_idx = all_idxs[:int(0.3 * r * c)]
    tg_idx = all_idxs[-int(0.0015 * r * c):]

    bg_idx_2d = np.zeros([bg_idx.shape[0], 2]).astype(int)
    bg_idx_2d[:, 0] = bg_idx // c
    bg_idx_2d[:, 1] = bg_idx % c
    for i in range(bg_idx.shape[0]):
        y_trn_map[bg_idx_2d[i, 0], bg_idx_2d[i, 1]] = 0

    tg_idx_2d = np.zeros([tg_idx.shape[0], 2]).astype(int)
    tg_idx_2d[:, 0] = tg_idx // c
    tg_idx_2d[:, 1] = tg_idx % c
    for i in range(tg_idx.shape[0]):
        y_trn_map[tg_idx_2d[i, 0], tg_idx_2d[i, 1]] = 1

    y_val_map = gt_tube

    # Data preparation
    if args.input_mode == 'whole':
        X_data = all_data_norm.reshape(1, r, c, -1)
        args.print_freq = 1
        input_size = (r, c)
    else:
        image_size = (r, c)
        input_size = args.input_size
        LyEnd, LxEnd = np.subtract(image_size, input_size)

        Lx = np.linspace(0, LxEnd, int(np.ceil(LxEnd / float(input_size[1] - args.overlap_size))) + 1,
                         endpoint=True).astype('int')
        Ly = np.linspace(0, LyEnd, int(np.ceil(LyEnd / float(input_size[0] - args.overlap_size))) + 1,
                         endpoint=True).astype('int')

        image_3D = all_data_norm.reshape(r, c, -1)
        N = len(Ly) * len(Lx)
        X_data = np.zeros([N, input_size[0], input_size[1], image_3D.shape[-1]])

        i = 0
        for j in range(len(Ly)):
            for k in range(len(Lx)):
                rStart, cStart = (Ly[j], Lx[k])
                rEnd, cEnd = (rStart + input_size[0], cStart + input_size[1])
                X_data[i] = image_3D[rStart:rEnd, cStart:cEnd, :]
                i += 1

    img_size = input_size[0]
    print(f'{args.dataset} image preparation finished! Data: {X_data.shape}')

    X_data = torch.from_numpy(X_data.transpose(0, 3, 1, 2))

    # Run experiments
    results = np.zeros([7, args.num_runs + 2])
    best_auc = 0

    for run in range(args.num_runs):
        print(f'\n===== Run {run + 1}/{args.num_runs} =====')

        # Prepare training labels
        if args.input_mode == 'whole':
            y_trn_data = y_trn_map.reshape(1, r, c)
        else:
            y_trn_data = np.zeros([N, input_size[0], input_size[1]], dtype=np.int32)
            i = 0
            for j in range(len(Ly)):
                for k in range(len(Lx)):
                    rStart, cStart = Ly[j], Lx[k]
                    rEnd, cEnd = rStart + input_size[0], cStart + input_size[1]
                    y_trn_data[i] = y_trn_map[rStart:rEnd, cStart:cEnd]
                    i += 1

        y_trn_data = torch.from_numpy(y_trn_data)

        # Prepare validation labels
        if args.input_mode == 'whole':
            y_val_data = y_val_map.reshape(1, r, c)
        else:
            y_val_data = np.zeros([N, input_size[0], input_size[1]])
            i = 0
            for j in range(len(Ly)):
                for k in range(len(Lx)):
                    rStart, cStart = (Ly[j], Lx[k])
                    rEnd, cEnd = (rStart + input_size[0], cStart + input_size[1])
                    y_val_data[i, :, :] = y_val_map[rStart:rEnd, cStart:cEnd]
                    i += 1

        y_val_data = torch.from_numpy(y_val_data)

        # Create data loaders
        torch.cuda.empty_cache()
        trn_dataset = TensorDataset(X_data, y_trn_data)
        trn_loader = DataLoader(
            trn_dataset, batch_size=args.batch_size, num_workers=args.workers,
            shuffle=True, drop_last=True, pin_memory=True
        )

        val_dataset = TensorDataset(X_data, y_val_data)
        val_loader = DataLoader(
            val_dataset, batch_size=args.val_batch_size, shuffle=False, pin_memory=True
        )

        # Create model
        if args.mode == 'sa':
            net = AnomalyDetectionHead(
                img_size=img_size,
                in_channels=X_data.shape[1],
                spat_weights=args.spat_weights,
            )
        else:
            net = SSAnomalyDetectionHead(
                img_size=img_size,
                in_channels=X_data.shape[1],
                spat_weights=args.spat_weights,
                spec_weights=args.spec_weights,
            )

        # Optimizer
        optim_wrapper = dict(
            optimizer=dict(type='AdamW', lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd),
            constructor='LayerDecayOptimizerConstructor_ViT',
            paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9)
        )
        optimizer = build_optim_wrapper(net, optim_wrapper)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer.optimizer, args.epochs, eta_min=0, last_epoch=-1
        )
        criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)

        net = torch.nn.DataParallel(net.cuda())

        # Training
        trn_time = 0
        best_val_auc = 0

        for epoch in range(args.epochs):
            trn_time1 = time.time()
            train_epoch(args, epoch, net, optimizer, scheduler, trn_loader, criterion)
            trn_time2 = time.time()
            trn_time += trn_time2 - trn_time1

            if (epoch + 1) % args.val_freq == 0:
                val_auc = validate(args, epoch, net, val_loader)
                if val_auc >= best_val_auc:
                    filename = os.path.join(save_path, f'HAD_{flag}_valbest_tmp.pth')
                    torch.save(net, filename)
                    best_val_auc = val_auc

        # Save final model if not saved during validation
        filename = os.path.join(save_path, f'HAD_{flag}_valbest_tmp.pth')
        if not os.path.exists(filename):
            torch.save(net, filename)

        # Testing
        net = torch.load(filename, map_location='cpu', weights_only=False)
        net = net.cuda()
        net.eval()

        tes_time1 = time.time()

        if args.input_mode == 'whole':
            with torch.no_grad():
                pred = net(X_data.float().cuda())
                y_tes_pred = pred[:, 1, :, :].cpu().numpy()
                y_tes_pred = standard(y_tes_pred)
                y_tes_pred = np.clip(y_tes_pred, 0, 1)
        else:
            img = torch.from_numpy(image_3D).permute(2, 0, 1)
            y_tes_pred = np.zeros([r, c])

            for j in range(len(Ly)):
                for k in range(len(Lx)):
                    rStart, cStart = (Ly[j], Lx[k])
                    rEnd, cEnd = (rStart + input_size[0], cStart + input_size[1])
                    img_part = img[:, rStart:rEnd, cStart:cEnd].unsqueeze(0)

                    with torch.no_grad():
                        pred = net(img_part.float().cuda())
                    part_pred = pred[0, 1, :, :].cpu().numpy()

                    if j == 0 and k == 0:
                        y_tes_pred[rStart:rEnd, cStart:cEnd] = part_pred
                    elif j == 0 and k > 0:
                        y_tes_pred[rStart:rEnd, cStart + int(args.overlap_size / 2):cEnd] = \
                            part_pred[:, int(args.overlap_size / 2):]
                    elif j > 0 and k == 0:
                        y_tes_pred[rStart + int(args.overlap_size / 2):rEnd, cStart:cEnd] = \
                            part_pred[int(args.overlap_size / 2):, :]
                    else:
                        y_tes_pred[rStart + int(args.overlap_size / 2):rEnd,
                                   cStart + int(args.overlap_size / 2):cEnd] = \
                            part_pred[int(args.overlap_size / 2):, int(args.overlap_size / 2):]

            y_tes_pred = standard(y_tes_pred)
            y_tes_pred = np.clip(y_tes_pred, 0, 1)

        tes_time2 = time.time()

        # Compute metrics
        y_tes_data = y_val_map.reshape(r, c)
        aucs = compute_auc_scores(y_tes_pred, y_tes_data)

        print(f'\n===== Run {run + 1} Results =====')
        for key, val in aucs.items():
            print(f'{key}: {val:.4f}')

        results[0, run] = aucs['auc_roc']
        results[1, run] = aucs['auc_fpr']
        results[2, run] = aucs['auc_tpr']
        results[3, run] = aucs['auc_combined']
        results[4, run] = aucs['auc_ratio']
        results[5, run] = trn_time
        results[6, run] = tes_time2 - tes_time1

        if aucs['auc_roc'] >= best_auc:
            best_auc = aucs['auc_roc']

    # Summary
    results[:, -2] = np.mean(results[:, 0:-2], axis=1)
    results[:, -1] = np.std(results[:, 0:-2], axis=1)

    print('\n===== Final Results =====')
    print(f'AUC-ROC: {results[0, -2]:.4f} +/- {results[0, -1]:.4f}')

    # Save results using ResultManager
    results_dir = os.path.join(HYPERSIGMA_ROOT, 'results')
    manager = ResultManager(
        task="anomaly_detection",
        dataset=args.dataset,
        checkpoint_path=args.spat_weights,
        output_base_dir=results_dir,
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
        'norm': args.norm,
        'input_mode': args.input_mode,
        'input_size': args.input_size,
        'model': 'HyperSIGMA',
    })

    # Note: For anomaly detection, AUC-ROC is the primary metric
    # F1/precision/recall require thresholding which isn't standard for AD
    # Store AUC metrics with correct labels
    manager.log_run(
        seed=args.seed,
        metrics=AnomalyDetectionMetrics(
            AUC_ROC=float(results[0, -2]),  # Mean across internal runs
            F1=0.0,  # Not computed for threshold-free AD
            precision=0.0,  # Not computed for threshold-free AD
            recall=0.0,  # Not computed for threshold-free AD
        ),
        extra={
            'auc_fpr': float(results[1, -2]),
            'auc_tpr': float(results[2, -2]),
            'auc_combined': float(results[3, -2]),
            'auc_ratio': float(results[4, -2]),
        }
    )
    manager.try_auto_aggregate()

    print(f'\nResults saved via ResultManager')


if __name__ == '__main__':
    main()
