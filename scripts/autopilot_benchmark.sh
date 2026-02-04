#!/bin/bash
# Autopilot Benchmark Script for HyperSIGMA
# Automatically keeps all GPUs busy with benchmark experiments
# Runs 10 experiments per task for statistical validation

cd /home/jovyan/gnew-b200-2-datavol-1/chlee/Domain-independent-SpectralFM/hypersigma-benchmark

LOG_DIR="/tmp/hypersigma_benchmark_logs"
mkdir -p $LOG_DIR

# Track completed experiments
COMPLETED_FILE="$LOG_DIR/completed_experiments.txt"
touch $COMPLETED_FILE

# Seeds for 10-run validation
SEEDS=(42 123 456 789 2024 1234 5678 9999 1111 7777)

# Define all experiments to run
declare -a EXPERIMENTS=(
    # Classification - IndianPines (10 runs) - each has internal 10 runs
    "classification:indian_pines:50:42"
    # Classification - PaviaU (10 runs) - each has internal 10 runs
    "classification:paviau:50:42"
    # Anomaly Detection (10 runs internally)
    "anomaly_detection:pavia:ss:42"
    # Change Detection - Hermiston (10 seeds for 10 runs)
    "change_detection:Hermiston:ss:42"
    "change_detection:Hermiston:ss:123"
    "change_detection:Hermiston:ss:456"
    "change_detection:Hermiston:ss:789"
    "change_detection:Hermiston:ss:2024"
    "change_detection:Hermiston:ss:1234"
    "change_detection:Hermiston:ss:5678"
    "change_detection:Hermiston:ss:9999"
    "change_detection:Hermiston:ss:1111"
    "change_detection:Hermiston:ss:7777"
    # Change Detection - BayArea (10 seeds)
    "change_detection:BayArea:ss:42"
    "change_detection:BayArea:ss:123"
    "change_detection:BayArea:ss:456"
    "change_detection:BayArea:ss:789"
    "change_detection:BayArea:ss:2024"
    "change_detection:BayArea:ss:1234"
    "change_detection:BayArea:ss:5678"
    "change_detection:BayArea:ss:9999"
    "change_detection:BayArea:ss:1111"
    "change_detection:BayArea:ss:7777"
    # Denoising (5 runs)
    "denoising:IndianPines:50:42"
    "denoising:IndianPines:50:123"
    "denoising:IndianPines:50:456"
    "denoising:IndianPines:50:789"
    "denoising:IndianPines:50:2024"
    # Target Detection - ss mode (10 seeds)
    "target_detection:SanDiego:ss:42"
    "target_detection:SanDiego:ss:123"
    "target_detection:SanDiego:ss:456"
    "target_detection:SanDiego:ss:789"
    "target_detection:SanDiego:ss:2024"
    "target_detection:SanDiego:ss:1234"
    "target_detection:SanDiego:ss:5678"
    "target_detection:SanDiego:ss:9999"
    "target_detection:SanDiego:ss:1111"
    "target_detection:SanDiego:ss:7777"
    # Target Detection - sa mode (10 seeds)
    "target_detection:SanDiego:sa:42"
    "target_detection:SanDiego:sa:123"
    "target_detection:SanDiego:sa:456"
    "target_detection:SanDiego:sa:789"
    "target_detection:SanDiego:sa:2024"
    "target_detection:SanDiego:sa:1234"
    "target_detection:SanDiego:sa:5678"
    "target_detection:SanDiego:sa:9999"
    "target_detection:SanDiego:sa:1111"
    "target_detection:SanDiego:sa:7777"
    # Unmixing - sa mode (5 seeds)
    "unmixing:Urban4:sa:42"
    "unmixing:Urban4:sa:123"
    "unmixing:Urban4:sa:456"
    "unmixing:Urban4:sa:789"
    "unmixing:Urban4:sa:2024"
    # Unmixing - ss mode (5 seeds)
    "unmixing:Urban4:ss:42"
    "unmixing:Urban4:ss:123"
    "unmixing:Urban4:ss:456"
    "unmixing:Urban4:ss:789"
    "unmixing:Urban4:ss:2024"
)

# Function to check if GPU is free (less than 1GB memory used)
is_gpu_free() {
    local gpu_id=$1
    local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
    if [ "$mem_used" -lt 1000 ]; then
        return 0  # Free
    else
        return 1  # Busy
    fi
}

# Function to check if experiment is completed
is_completed() {
    local exp=$1
    grep -q "^$exp$" $COMPLETED_FILE
}

# Function to mark experiment as completed
mark_completed() {
    local exp=$1
    echo "$exp" >> $COMPLETED_FILE
}

# Function to start an experiment on a specific GPU
start_experiment() {
    local gpu_id=$1
    local exp=$2

    IFS=':' read -r task dataset param1 param2 <<< "$exp"

    local log_file="$LOG_DIR/${task}_${dataset}_${param1}_${param2}.log"

    echo "[$(date)] Starting $task on GPU $gpu_id: $exp"

    case $task in
        "classification")
            local samples=$param1
            local seed=$param2
            CUDA_VISIBLE_DEVICES=$gpu_id nohup python tasks/classification/train.py \
                --dataset $dataset \
                --data_dir ../data/benchmark/classification/$(echo $dataset | sed 's/indian_pines/IndianPines/;s/paviau/PaviaU/') \
                --samples_per_class $samples \
                --epochs 200 \
                --lr 1e-4 \
                --batch_size 64 \
                --num_runs 10 \
                --seed $seed \
                --gpus 0 \
                > $log_file 2>&1 &
            ;;
        "anomaly_detection")
            local mode=$param1
            local seed=$param2
            CUDA_VISIBLE_DEVICES=$gpu_id nohup python tasks/anomaly_detection/train.py \
                --dataset $dataset \
                --data_dir ../data/benchmark/anomaly_detection/Pavia \
                --mode $mode \
                --epochs 10 \
                --lr 6e-5 \
                --num_runs 10 \
                --seed $seed \
                > $log_file 2>&1 &
            ;;
        "change_detection")
            local mode=$param1
            local seed=$param2
            CUDA_VISIBLE_DEVICES=$gpu_id nohup python tasks/change_detection/train.py \
                --dataset $dataset \
                --data_dir ../data/benchmark/change_detection \
                --mode $mode \
                --epochs 10 \
                --lr 6e-5 \
                --seed $seed \
                > $log_file 2>&1 &
            ;;
        "denoising")
            local epochs=$param1
            local seed=$param2
            CUDA_VISIBLE_DEVICES=$gpu_id nohup python tasks/denoising/train.py \
                --dataset $dataset \
                --data_dir ../data/benchmark/classification/IndianPines \
                --epochs $epochs \
                --batch_size 8 \
                --lr 1e-4 \
                --seed $seed \
                > $log_file 2>&1 &
            ;;
        "target_detection")
            local mode=$param1
            local seed=$param2
            CUDA_VISIBLE_DEVICES=$gpu_id nohup python tasks/target_detection/train.py \
                --dataset $dataset \
                --data_dir ../data/benchmark/target_detection/SanDiego \
                --mode $mode \
                --epochs 50 \
                --batch_size 32 \
                --lr 1e-4 \
                --seed $seed \
                > $log_file 2>&1 &
            ;;
        "unmixing")
            local mode=$param1
            local seed=$param2
            CUDA_VISIBLE_DEVICES=$gpu_id nohup python tasks/unmixing/train.py \
                --dataset $dataset \
                --data_dir ../data/benchmark/unmixing/Urban \
                --mode $mode \
                --epochs 50 \
                --batch_size 64 \
                --lr 1e-4 \
                --seed $seed \
                > $log_file 2>&1 &
            ;;
    esac

    echo "[$(date)] Started PID: $!"
}

# Function to get next pending experiment
get_next_experiment() {
    for exp in "${EXPERIMENTS[@]}"; do
        if ! is_completed "$exp"; then
            echo "$exp"
            return 0
        fi
    done
    echo ""
    return 1
}

# Main loop
echo "========================================"
echo "HyperSIGMA Autopilot Benchmark Started"
echo "========================================"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "Log directory: $LOG_DIR"
echo ""

# Skip GPU 5 (already has a long-running process)
AVAILABLE_GPUS=(0 1 2 3 4 6 7)

while true; do
    # Check if all experiments are done
    all_done=true
    for exp in "${EXPERIMENTS[@]}"; do
        if ! is_completed "$exp"; then
            all_done=false
            break
        fi
    done

    if $all_done; then
        echo "[$(date)] All experiments completed!"
        break
    fi

    # Check each GPU
    for gpu_id in "${AVAILABLE_GPUS[@]}"; do
        if is_gpu_free $gpu_id; then
            next_exp=$(get_next_experiment)
            if [ -n "$next_exp" ]; then
                mark_completed "$next_exp"
                start_experiment $gpu_id "$next_exp"
                sleep 5  # Wait a bit before checking next GPU
            fi
        fi
    done

    # Status update every 60 seconds
    completed_count=$(wc -l < $COMPLETED_FILE)
    total_count=${#EXPERIMENTS[@]}
    echo "[$(date)] Progress: $completed_count/$total_count experiments started"

    # Show GPU status
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    echo ""

    sleep 60  # Check every minute
done

echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo "Results saved to: results/"
ls -la results/
