#!/usr/bin/env python
"""
Unified benchmark runner for HyperSIGMA.

Usage:
    python scripts/run_benchmark.py --task anomaly --dataset pavia --mode ss
    python scripts/run_benchmark.py --task classification --dataset indian_pines --samples 10
"""

import argparse
import os
import subprocess
import sys


def run_anomaly_detection(args):
    """Run anomaly detection benchmark."""
    cmd = [
        sys.executable,
        'tasks/anomaly_detection/train.py',
        '--dataset', args.dataset,
        '--mode', args.mode,
        '--epochs', str(args.epochs),
        '--num_runs', str(args.num_runs),
        '--seed', str(args.seed),
    ]

    if args.output_dir:
        cmd.extend(['--output_dir', args.output_dir])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_classification(args):
    """Run classification benchmark."""
    cmd = [
        sys.executable,
        'tasks/classification/train.py',
        '--dataset', args.dataset,
        '--samples_per_class', str(args.samples),
        '--epochs', str(args.epochs),
        '--num_runs', str(args.num_runs),
        '--seed', str(args.seed),
    ]

    if args.output_dir:
        cmd.extend(['--output_dir', args.output_dir])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="HyperSIGMA Benchmark Runner")

    parser.add_argument('--task', type=str, required=True,
                        choices=['anomaly', 'classification', 'change_detection',
                                 'denoising', 'unmixing'],
                        help='Task to benchmark')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name')

    # Task-specific args
    parser.add_argument('--mode', type=str, default='ss',
                        choices=['sa', 'ss'],
                        help='Model mode for anomaly detection')
    parser.add_argument('--samples', type=int, default=10,
                        help='Samples per class for classification')

    # Common args
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of experiment runs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')

    args = parser.parse_args()

    # Change to repo root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    os.chdir(repo_dir)

    if args.task == 'anomaly':
        run_anomaly_detection(args)
    elif args.task == 'classification':
        run_classification(args)
    else:
        print(f"Task '{args.task}' not yet implemented")
        sys.exit(1)


if __name__ == '__main__':
    main()
