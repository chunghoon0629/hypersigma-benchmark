#!/bin/bash
# HyperSIGMA 10-Seed Benchmark Runner
# Runs remaining experiments for tasks that need 9 more seeds
#
# Status:
# - Classification: ✅ Complete (10 runs)
# - Anomaly Detection: ✅ Complete (10 runs)
# - Change Detection: ❌ Needs 9 more (seeds 123-7777)
# - Target Detection: ❌ Needs 9 more (seeds 123-7777)
# - Denoising: ❌ Needs 9 more (seeds 123-7777)
# - Unmixing: ❌ Needs 9 more (seeds 123-7777)

# Don't exit on error since ((job_idx++)) can return 1 when job_idx is 0
# set -e

# Configuration
SEEDS=(123 456 789 2024 1234 5678 9999 1111 7777)  # Seed 42 already done
GPUS=(0 1 2 3 4 6 7)  # Available GPUs (skip 5 if reserved)
NUM_GPUS=${#GPUS[@]}

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(dirname "$SCRIPT_DIR")"
TASKS_DIR="$BENCHMARK_ROOT/tasks"
LOG_DIR="$BENCHMARK_ROOT/logs/10seed_$(date +%Y%m%d_%H%M)"

mkdir -p "$LOG_DIR"

echo "=============================================="
echo "HyperSIGMA 10-Seed Benchmark"
echo "=============================================="
echo "Benchmark root: $BENCHMARK_ROOT"
echo "Log directory: $LOG_DIR"
echo "Seeds to run: ${SEEDS[*]}"
echo "GPUs available: ${GPUS[*]}"
echo "=============================================="

cd "$BENCHMARK_ROOT"

# GPU assignment function
get_gpu() {
    local idx=$1
    echo ${GPUS[$((idx % NUM_GPUS))]}
}

# Parse arguments
RUN_CHANGE=true
RUN_TARGET=true
RUN_DENOISING=true
RUN_UNMIXING=true
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --only-change)
            RUN_TARGET=false; RUN_DENOISING=false; RUN_UNMIXING=false
            shift ;;
        --only-target)
            RUN_CHANGE=false; RUN_DENOISING=false; RUN_UNMIXING=false
            shift ;;
        --only-denoising)
            RUN_CHANGE=false; RUN_TARGET=false; RUN_UNMIXING=false
            shift ;;
        --only-unmixing)
            RUN_CHANGE=false; RUN_TARGET=false; RUN_DENOISING=false
            shift ;;
        --dry-run)
            DRY_RUN=true
            shift ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
done

# Job counter for GPU assignment
job_idx=0

# Function to run a job
run_job() {
    local cmd="$1"
    local log_file="$2"
    local gpu=$(get_gpu $job_idx)

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] GPU $gpu: $cmd > $log_file"
    else
        echo "Starting on GPU $gpu: $log_file"
        CUDA_VISIBLE_DEVICES=$gpu $cmd > "$log_file" 2>&1 &
    fi
    job_idx=$((job_idx + 1))
}

# ============================================
# Change Detection (Hermiston + BayArea, ss mode)
# ============================================
if [ "$RUN_CHANGE" = true ]; then
    echo ""
    echo ">>> Change Detection (18 jobs: 2 datasets × 9 seeds)"

    for SEED in "${SEEDS[@]}"; do
        # Hermiston
        run_job "python tasks/change_detection/train.py --dataset Hermiston --mode ss --seed $SEED" \
                "$LOG_DIR/cd_hermiston_seed${SEED}.log"

        # BayArea
        run_job "python tasks/change_detection/train.py --dataset BayArea --mode ss --seed $SEED" \
                "$LOG_DIR/cd_bayarea_seed${SEED}.log"
    done
fi

# ============================================
# Target Detection (SanDiego, sa + ss modes)
# ============================================
if [ "$RUN_TARGET" = true ]; then
    echo ""
    echo ">>> Target Detection (18 jobs: 2 modes × 9 seeds)"

    for SEED in "${SEEDS[@]}"; do
        # SA mode
        run_job "python tasks/target_detection/train.py --dataset Sandiego --mode sa --seed $SEED" \
                "$LOG_DIR/td_sandiego_sa_seed${SEED}.log"

        # SS mode
        run_job "python tasks/target_detection/train.py --dataset Sandiego --mode ss --seed $SEED" \
                "$LOG_DIR/td_sandiego_ss_seed${SEED}.log"
    done
fi

# ============================================
# Denoising (WDC)
# ============================================
if [ "$RUN_DENOISING" = true ]; then
    echo ""
    echo ">>> Denoising (9 jobs: 1 dataset × 9 seeds)"

    for SEED in "${SEEDS[@]}"; do
        run_job "python tasks/denoising/train.py --dataset WDC --seed $SEED" \
                "$LOG_DIR/dn_wdc_seed${SEED}.log"
    done
fi

# ============================================
# Unmixing (Urban4, sa + ss modes with VCA + augmentation)
# ============================================
if [ "$RUN_UNMIXING" = true ]; then
    echo ""
    echo ">>> Unmixing (18 jobs: 2 modes × 9 seeds)"

    for SEED in "${SEEDS[@]}"; do
        # SA mode
        run_job "python tasks/unmixing/train.py --dataset Urban4 --mode sa --seed $SEED --augment" \
                "$LOG_DIR/um_urban4_sa_seed${SEED}.log"

        # SS mode
        run_job "python tasks/unmixing/train.py --dataset Urban4 --mode ss --seed $SEED --augment" \
                "$LOG_DIR/um_urban4_ss_seed${SEED}.log"
    done
fi

echo ""
echo "=============================================="
echo "Started $job_idx jobs"
echo "=============================================="

if [ "$DRY_RUN" = false ]; then
    echo ""
    echo "Monitor progress with:"
    echo "  watch 'ps aux | grep python | grep -E \"train.py\" | wc -l'"
    echo ""
    echo "Check logs:"
    echo "  ls -la $LOG_DIR/"
    echo "  tail -f $LOG_DIR/*.log"
    echo ""
    echo "When complete, aggregate results with:"
    echo "  python scripts/aggregate_results.py"
fi
