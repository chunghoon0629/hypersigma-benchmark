#!/bin/bash
# HyperSIGMA Full 10-Seed Benchmark
# Runs ALL experiments with ALL 10 seeds

cd "$(dirname "$0")/.."
BENCHMARK_ROOT=$(pwd)

# All 10 seeds
SEEDS=(42 123 456 789 2024 1234 5678 9999 1111 7777)

# All 8 GPUs available
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

LOG_DIR="$BENCHMARK_ROOT/logs/full_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "HyperSIGMA Full 10-Seed Benchmark"
echo "=============================================="
echo "Seeds: ${SEEDS[*]}"
echo "GPUs: ${GPUS[*]}"
echo "Log dir: $LOG_DIR"
echo "=============================================="

# Backup and recreate results directory
if [ -d "results" ]; then
    BACKUP_DIR="results_backup_$(date +%Y%m%d_%H%M%S)"
    mv results "$BACKUP_DIR"
    echo "Backed up existing results to $BACKUP_DIR"
fi

mkdir -p results/{classification,anomaly_detection/ss,change_detection/ss,target_detection/{sa,ss},denoising,unmixing/{sa,ss}}

# GPU assignment
get_gpu() {
    echo ${GPUS[$(($1 % NUM_GPUS))]}
}

job_idx=0

# ============================================
# 1. Classification (20 runs: 2 datasets × 10 seeds)
#    Note: Classification runs all 10 seeds internally with --num_runs 10
#    So we just need 2 runs total (one per dataset)
# ============================================
echo ""
echo ">>> Classification (2 jobs: IndianPines + PaviaU, each with 10 internal runs)"

GPU=$(get_gpu $job_idx)
echo "GPU $GPU: Classification IndianPines"
CUDA_VISIBLE_DEVICES=$GPU python tasks/classification/train.py \
    --dataset indian_pines --samples_per_class 50 --eval_segmentation --num_runs 10 \
    > "$LOG_DIR/cls_indianpines.log" 2>&1 &
job_idx=$((job_idx + 1))

GPU=$(get_gpu $job_idx)
echo "GPU $GPU: Classification PaviaU"
CUDA_VISIBLE_DEVICES=$GPU python tasks/classification/train.py \
    --dataset pavia_university --samples_per_class 50 --eval_segmentation --num_runs 10 \
    > "$LOG_DIR/cls_paviau.log" 2>&1 &
job_idx=$((job_idx + 1))

# ============================================
# 2. Anomaly Detection (1 run with 10 internal runs)
# ============================================
echo ""
echo ">>> Anomaly Detection (1 job: Pavia with 10 internal runs)"

GPU=$(get_gpu $job_idx)
echo "GPU $GPU: Anomaly Detection Pavia"
CUDA_VISIBLE_DEVICES=$GPU python tasks/anomaly_detection/train.py \
    --dataset pavia --mode ss --num_runs 10 \
    > "$LOG_DIR/ad_pavia.log" 2>&1 &
job_idx=$((job_idx + 1))

# ============================================
# 3. Change Detection (20 runs: 2 datasets × 10 seeds)
# ============================================
echo ""
echo ">>> Change Detection (20 jobs: 2 datasets × 10 seeds)"

for SEED in "${SEEDS[@]}"; do
    GPU=$(get_gpu $job_idx)
    echo "GPU $GPU: CD Hermiston seed $SEED"
    CUDA_VISIBLE_DEVICES=$GPU python tasks/change_detection/train.py \
        --dataset Hermiston --mode ss --seed $SEED \
        > "$LOG_DIR/cd_hermiston_seed${SEED}.log" 2>&1 &
    job_idx=$((job_idx + 1))

    GPU=$(get_gpu $job_idx)
    echo "GPU $GPU: CD BayArea seed $SEED"
    CUDA_VISIBLE_DEVICES=$GPU python tasks/change_detection/train.py \
        --dataset BayArea --mode ss --seed $SEED \
        > "$LOG_DIR/cd_bayarea_seed${SEED}.log" 2>&1 &
    job_idx=$((job_idx + 1))
done

# ============================================
# 4. Target Detection (20 runs: 2 modes × 10 seeds)
# ============================================
echo ""
echo ">>> Target Detection (20 jobs: 2 modes × 10 seeds)"

for SEED in "${SEEDS[@]}"; do
    GPU=$(get_gpu $job_idx)
    echo "GPU $GPU: TD SanDiego SA seed $SEED"
    CUDA_VISIBLE_DEVICES=$GPU python tasks/target_detection/train.py \
        --dataset Sandiego --mode sa --seed $SEED \
        > "$LOG_DIR/td_sa_seed${SEED}.log" 2>&1 &
    job_idx=$((job_idx + 1))

    GPU=$(get_gpu $job_idx)
    echo "GPU $GPU: TD SanDiego SS seed $SEED"
    CUDA_VISIBLE_DEVICES=$GPU python tasks/target_detection/train.py \
        --dataset Sandiego --mode ss --seed $SEED \
        > "$LOG_DIR/td_ss_seed${SEED}.log" 2>&1 &
    job_idx=$((job_idx + 1))
done

# ============================================
# 5. Denoising (10 runs: 1 dataset × 10 seeds)
# ============================================
echo ""
echo ">>> Denoising (10 jobs: WDC × 10 seeds)"

for SEED in "${SEEDS[@]}"; do
    GPU=$(get_gpu $job_idx)
    echo "GPU $GPU: Denoising WDC seed $SEED"
    CUDA_VISIBLE_DEVICES=$GPU python tasks/denoising/train.py \
        --dataset WDC --seed $SEED \
        > "$LOG_DIR/dn_wdc_seed${SEED}.log" 2>&1 &
    job_idx=$((job_idx + 1))
done

# ============================================
# 6. Unmixing (20 runs: 2 modes × 10 seeds)
# ============================================
echo ""
echo ">>> Unmixing (20 jobs: 2 modes × 10 seeds)"

for SEED in "${SEEDS[@]}"; do
    GPU=$(get_gpu $job_idx)
    echo "GPU $GPU: Unmixing Urban4 SA seed $SEED"
    CUDA_VISIBLE_DEVICES=$GPU python tasks/unmixing/train.py \
        --dataset Urban4 --mode sa --seed $SEED --augment \
        > "$LOG_DIR/um_sa_seed${SEED}.log" 2>&1 &
    job_idx=$((job_idx + 1))

    GPU=$(get_gpu $job_idx)
    echo "GPU $GPU: Unmixing Urban4 SS seed $SEED"
    CUDA_VISIBLE_DEVICES=$GPU python tasks/unmixing/train.py \
        --dataset Urban4 --mode ss --seed $SEED --augment \
        > "$LOG_DIR/um_ss_seed${SEED}.log" 2>&1 &
    job_idx=$((job_idx + 1))
done

echo ""
echo "=============================================="
echo "Started $job_idx background jobs"
echo "=============================================="
echo ""
echo "Monitor progress:"
echo "  watch 'ps aux | grep python | grep -E \"train.py\" | wc -l'"
echo "  tail -f $LOG_DIR/*.log"
echo ""
echo "When complete, aggregate results:"
echo "  python scripts/aggregate_results.py --markdown"
