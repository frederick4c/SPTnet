#!/bin/bash
#SBATCH -J SPTnet_bench
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --output=sptnet_bench_%j.out
#SBATCH --error=sptnet_bench_%j.err
#SBATCH --export=ALL

set -euo pipefail

# ---------------------------------------------------------------------
# Module loading and environment setup
# ---------------------------------------------------------------------

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.11.0-icl
source ~/envs/SPTnet/bin/activate

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SPT_NUM_WORKERS="${SPT_NUM_WORKERS:-2}"

# With many .mat files, h5py/PyTorch can keep many file handles open.
ulimit -n 65536

# ---------------------------------------------------------------------
# Benchmark settings
# Override these with sbatch --export=ALL,NAME=value,...
# ---------------------------------------------------------------------

export SPT_TRAIN_SCRIPT="${SPT_TRAIN_SCRIPT:-SPTnet_training_cli.py}"
export SPT_TRAIN_DATA="${SPT_TRAIN_DATA:-./TrainData/*.mat}"
export SPT_RUN_NAME="${SPT_RUN_NAME:-benchmark}"
export SPT_MODEL_BASE_DIR="${SPT_MODEL_BASE_DIR:-./Trained_models/benchmarks}"
export SPT_MODEL_DIR="${SPT_MODEL_DIR:-$SPT_MODEL_BASE_DIR/$SPT_RUN_NAME}"

export SPT_BATCH_SIZE="${SPT_BATCH_SIZE:-16}"
export SPT_QUERY="${SPT_QUERY:-20}"
export SPT_LEARNING_RATE="${SPT_LEARNING_RATE:-0.0001}"
export SPT_MAX_DC="${SPT_MAX_DC:-0.5}"
export SPT_MAX_EPOCHS="${SPT_MAX_EPOCHS:-0}"
export SPT_GPUS="${SPT_GPUS:-1}"

if [[ ! -f "$SPT_TRAIN_SCRIPT" ]]; then
    echo "ERROR: training script '$SPT_TRAIN_SCRIPT' was not found in $(pwd)"
    exit 1
fi

match_count=$(
    python - "$SPT_TRAIN_DATA" <<'PY'
import glob
import sys

matches = sorted(glob.glob(sys.argv[1]))
print(len(matches))
PY
)

if [[ "$match_count" -eq 0 ]]; then
    echo "ERROR: training data pattern matched no files: $SPT_TRAIN_DATA"
    exit 1
fi

mkdir -p "$SPT_MODEL_DIR"
benchmark_log="$SPT_MODEL_DIR/slurm_benchmark_${SLURM_JOB_ID:-manual}.log"
benchmark_metrics="$SPT_MODEL_DIR/slurm_benchmark_${SLURM_JOB_ID:-manual}_metrics.txt"

echo "Starting SPTnet benchmark on CSD3 Ampere node..."
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Training script: $SPT_TRAIN_SCRIPT"
echo "Training data pattern: $SPT_TRAIN_DATA"
echo "Matched training files: $match_count"
echo "Run name: $SPT_RUN_NAME"
echo "Model dir: $SPT_MODEL_DIR"
echo "Batch size: $SPT_BATCH_SIZE"
echo "Query count: $SPT_QUERY"
echo "Learning rate: $SPT_LEARNING_RATE"
echo "Max diffusion coefficient: $SPT_MAX_DC"
echo "Max epochs: $SPT_MAX_EPOCHS"
echo "GPUs: $SPT_GPUS"
echo "DataLoader workers: $SPT_NUM_WORKERS"
echo "Python: $(which python)"
echo "Benchmark log: $benchmark_log"
echo "Benchmark metrics: $benchmark_metrics"

wall_start_epoch=$(date +%s)
wall_start_human=$(date '+%Y-%m-%d %H:%M:%S %Z')

echo "Benchmark wall-clock start: $wall_start_human"

set +e
/usr/bin/time -v ~/envs/SPTnet/bin/python "$SPT_TRAIN_SCRIPT" \
    --model-dir "$SPT_MODEL_DIR" \
    --data "$SPT_TRAIN_DATA" \
    --gpus "$SPT_GPUS" \
    --batch-size "$SPT_BATCH_SIZE" \
    --learning-rate "$SPT_LEARNING_RATE" \
    --query "$SPT_QUERY" \
    --max_dc "$SPT_MAX_DC" \
    --max-epochs "$SPT_MAX_EPOCHS" 2>&1 | tee "$benchmark_log"
train_status=${PIPESTATUS[0]}
set -e

wall_end_epoch=$(date +%s)
wall_end_human=$(date '+%Y-%m-%d %H:%M:%S %Z')
total_wall_seconds=$((wall_end_epoch - wall_start_epoch))

training_seconds=$(
    sed -n 's/.*Training takes \([0-9][0-9]*\) s.*/\1/p' "$benchmark_log" | tail -1
)
if [[ -n "$training_seconds" ]]; then
    startup_and_data_seconds=$((total_wall_seconds - training_seconds))
else
    training_seconds="unknown"
    startup_and_data_seconds="unknown"
fi

{
    echo "run_name=$SPT_RUN_NAME"
    echo "job_id=${SLURM_JOB_ID:-unknown}"
    echo "train_script=$SPT_TRAIN_SCRIPT"
    echo "train_data=$SPT_TRAIN_DATA"
    echo "matched_training_files=$match_count"
    echo "model_dir=$SPT_MODEL_DIR"
    echo "wall_start=$wall_start_human"
    echo "wall_end=$wall_end_human"
    echo "total_wall_seconds=$total_wall_seconds"
    echo "training_seconds=$training_seconds"
    echo "startup_plus_data_loading_seconds=$startup_and_data_seconds"
    echo "exit_status=$train_status"
} | tee "$benchmark_metrics"

echo "Benchmark wall-clock end: $wall_end_human"
echo "Total script wall time, including startup, imports, data loading, validation and training: ${total_wall_seconds}s"
echo "Training-loop time reported by Python script: ${training_seconds}s"
echo "Estimated startup plus data-loading/pre-training overhead: ${startup_and_data_seconds}s"

if [[ "$train_status" -ne 0 ]]; then
    echo "ERROR: training command failed with exit status $train_status"
    exit "$train_status"
fi

echo "Benchmark job completed."
