#!/bin/bash
# Full HyperSIGMA Benchmark Evaluation on GPU 0
# Runs all 6 tasks with all 10 standard seeds

set -e  # Exit on error

cd /home/jovyan/gnew-b200-2-datavol-1/chlee/Domain-independent-SpectralFM/hypersigma-benchmark

export CUDA_VISIBLE_DEVICES=0

# Standard seeds
SEEDS=(42 123 456 789 2024 1234 5678 9999 1111 7777)

echo "=============================================="
echo "HyperSIGMA Benchmark Evaluation"
echo "GPU: 0"
echo "Seeds: ${SEEDS[@]}"
echo "=============================================="

# 1. Denoising - IndianPines (50 epochs)
echo ""
echo "=== TASK 1: DENOISING ==="
for seed in "${SEEDS[@]}"; do
    echo "Running Denoising with seed=$seed..."
    python tasks/denoising/train.py \
        --dataset IndianPines \
        --epochs 50 \
        --seed $seed \
        --batch_size 8 \
        --patch_size 64
done
echo "Denoising complete!"

# 2. Change Detection - Hermiston (50 epochs)
echo ""
echo "=== TASK 2: CHANGE DETECTION ==="
for seed in "${SEEDS[@]}"; do
    echo "Running Change Detection with seed=$seed..."
    python tasks/change_detection/train.py \
        --dataset Hermiston \
        --mode ss \
        --epochs 50 \
        --seed $seed
done
echo "Change Detection complete!"

# 3. Target Detection - Sandiego (50 epochs)
echo ""
echo "=== TASK 3: TARGET DETECTION ==="
for seed in "${SEEDS[@]}"; do
    echo "Running Target Detection with seed=$seed..."
    python tasks/target_detection/train.py \
        --dataset Sandiego \
        --mode ss \
        --epochs 50 \
        --seed $seed
done
echo "Target Detection complete!"

# 4. Unmixing - Urban4 (50 epochs)
echo ""
echo "=== TASK 4: UNMIXING ==="
for seed in "${SEEDS[@]}"; do
    echo "Running Unmixing with seed=$seed..."
    python tasks/unmixing/train.py \
        --dataset Urban4 \
        --mode ss \
        --epochs 50 \
        --seed $seed
done
echo "Unmixing complete!"

# 5. Anomaly Detection - Pavia (10 epochs)
echo ""
echo "=== TASK 5: ANOMALY DETECTION ==="
for seed in "${SEEDS[@]}"; do
    echo "Running Anomaly Detection with seed=$seed..."
    python tasks/anomaly_detection/train.py \
        --dataset pavia \
        --mode ss \
        --epochs 10 \
        --seed $seed
done
echo "Anomaly Detection complete!"

# 6. Classification - IndianPines (100 epochs)
echo ""
echo "=== TASK 6: CLASSIFICATION ==="
for seed in "${SEEDS[@]}"; do
    echo "Running Classification with seed=$seed..."
    python tasks/classification/train.py \
        --dataset indian_pines \
        --samples_per_class 50 \
        --epochs 100 \
        --seed $seed \
        --num_runs 1
done
echo "Classification complete!"

echo ""
echo "=============================================="
echo "ALL BENCHMARKS COMPLETE!"
echo "=============================================="
echo "Results saved in: hypersigma-benchmark/results/"
