#!/bin/bash
# HyperSIGMA Full Benchmark Evaluation Script
# Run all downstream tasks with 10 seeds for statistical validation

# Configuration
GPU=${1:-0}  # Default to GPU 0
export CUDA_VISIBLE_DEVICES=$GPU

# Seeds for multi-run validation
SEEDS=(42 123 456 789 2024 1234 5678 9999 1111 7777)

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS_DIR="${SCRIPT_DIR}/tasks"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/benchmark_${TIMESTAMP}.log"

echo "========================================"
echo "HyperSIGMA Full Benchmark Evaluation"
echo "========================================"
echo "GPU: $GPU"
echo "Seeds: ${SEEDS[*]}"
echo "Log: $MAIN_LOG"
echo "========================================"
echo ""

# Function to run experiments with progress tracking
run_task() {
    local task=$1
    local dataset=$2
    local epochs=$3
    local extra_args=$4

    echo "----------------------------------------"
    echo "Task: $task | Dataset: $dataset"
    echo "----------------------------------------"

    for seed in "${SEEDS[@]}"; do
        echo "  Running seed $seed..."
        local log_file="${LOG_DIR}/${task}_${dataset}_seed${seed}_${TIMESTAMP}.log"

        case $task in
            classification)
                python "${TASKS_DIR}/classification/train.py" \
                    --dataset "$dataset" \
                    --samples_per_class 50 \
                    --epochs "$epochs" \
                    --seed "$seed" \
                    --num_runs 1 \
                    $extra_args \
                    >> "$log_file" 2>&1
                ;;
            denoising)
                python "${TASKS_DIR}/denoising/train.py" \
                    --dataset "$dataset" \
                    --epochs "$epochs" \
                    --seed "$seed" \
                    $extra_args \
                    >> "$log_file" 2>&1
                ;;
            anomaly_detection)
                python "${TASKS_DIR}/anomaly_detection/train.py" \
                    --dataset "$dataset" \
                    --epochs "$epochs" \
                    --seed "$seed" \
                    --num_runs 1 \
                    $extra_args \
                    >> "$log_file" 2>&1
                ;;
            target_detection)
                python "${TASKS_DIR}/target_detection/train.py" \
                    --dataset "$dataset" \
                    --epochs "$epochs" \
                    --seed "$seed" \
                    $extra_args \
                    >> "$log_file" 2>&1
                ;;
            change_detection)
                python "${TASKS_DIR}/change_detection/train.py" \
                    --dataset "$dataset" \
                    --epochs "$epochs" \
                    --seed "$seed" \
                    $extra_args \
                    >> "$log_file" 2>&1
                ;;
        esac

        if [ $? -eq 0 ]; then
            echo "    ✓ Seed $seed completed"
        else
            echo "    ✗ Seed $seed FAILED (see $log_file)"
        fi
    done
    echo ""
}

# Start time
START_TIME=$(date +%s)

# 1. Classification (3 datasets × 10 seeds = 30 runs)
echo "========================================" | tee -a "$MAIN_LOG"
echo "1. CLASSIFICATION" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
run_task "classification" "indian_pines" 100 "" 2>&1 | tee -a "$MAIN_LOG"
run_task "classification" "paviau" 100 "" 2>&1 | tee -a "$MAIN_LOG"
run_task "classification" "houston" 100 "" 2>&1 | tee -a "$MAIN_LOG"

# 2. Denoising (1 dataset × 10 seeds = 10 runs)
echo "========================================" | tee -a "$MAIN_LOG"
echo "2. DENOISING" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
run_task "denoising" "WDC" 50 "" 2>&1 | tee -a "$MAIN_LOG"

# 3. Anomaly Detection (2 datasets × 10 seeds = 20 runs)
echo "========================================" | tee -a "$MAIN_LOG"
echo "3. ANOMALY DETECTION" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
run_task "anomaly_detection" "pavia" 10 "" 2>&1 | tee -a "$MAIN_LOG"
run_task "anomaly_detection" "cri" 10 "" 2>&1 | tee -a "$MAIN_LOG"

# 4. Target Detection (1 dataset × 10 seeds = 10 runs)
echo "========================================" | tee -a "$MAIN_LOG"
echo "4. TARGET DETECTION" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
run_task "target_detection" "Sandiego" 50 "" 2>&1 | tee -a "$MAIN_LOG"

# 5. Change Detection (3 datasets × 10 seeds = 30 runs)
echo "========================================" | tee -a "$MAIN_LOG"
echo "5. CHANGE DETECTION" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
run_task "change_detection" "Hermiston" 50 "" 2>&1 | tee -a "$MAIN_LOG"
run_task "change_detection" "BayArea" 50 "" 2>&1 | tee -a "$MAIN_LOG"
run_task "change_detection" "SantaBarbara" 50 "" 2>&1 | tee -a "$MAIN_LOG"

# End time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "========================================" | tee -a "$MAIN_LOG"
echo "BENCHMARK COMPLETE" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s" | tee -a "$MAIN_LOG"
echo "Results saved to: ${SCRIPT_DIR}/results/" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "Total runs: 100" | tee -a "$MAIN_LOG"
echo "  - Classification: 30 (IndianPines, PaviaU, Houston)" | tee -a "$MAIN_LOG"
echo "  - Denoising: 10 (WDC)" | tee -a "$MAIN_LOG"
echo "  - Anomaly Detection: 20 (Pavia, CRI)" | tee -a "$MAIN_LOG"
echo "  - Target Detection: 10 (Sandiego)" | tee -a "$MAIN_LOG"
echo "  - Change Detection: 30 (Hermiston, BayArea, SantaBarbara)" | tee -a "$MAIN_LOG"
