#!/bin/bash
#SBATCH -J SPTnet_train_full         # Job name
#SBATCH -A MPHIL-DIS-SL2-GPU         # Project / Account
#SBATCH -p ampere                    # Partition (queue) name for Ampere A100 nodes
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of scripts to run
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --gres=gpu:1                 # Number of GPUs per node (A100)
#SBATCH --time=24:00:00              # Time limit hrs:min:sec (24 hours for 1000 files)
#SBATCH --output=sptnet_train_full_%j.out # Standard output log (%j = job ID)
#SBATCH --error=sptnet_train_full_%j.err  # Standard error log
#SBATCH --export=ALL                 # Export all environment variables to the job

# ---------------------------------------------------------------------
# Module Loading & Environment Setup 
# ---------------------------------------------------------------------

# Purge existing modules and load standard Ampere environment for Rocky/RHEL 8
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

# Load the standard Python 3 module (adjust if CSD3 uses a specific python/3.x module)
module load python/3.11.0-icl
source ~/envs/SPTnet/bin/activate

# ---------------------------------------------------------------------
# Run the application
# ---------------------------------------------------------------------

# Go to the directory where the job was submitted from
cd $SLURM_SUBMIT_DIR

# Prevent standard python multiprocessing from taking all CPU cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Increase open file limit. With 1000 .mat files, PyTorch's ConcatDataset 
# might try to keep many h5py file handles open simultaneously. 
ulimit -n 65536

export SPT_MODEL_DIR="${SPT_MODEL_DIR:-./Trained_models/full_run/}"
export SPT_TRAIN_DATA="${SPT_TRAIN_DATA:-./TrainData/*.mat}"
export SPT_BATCH_SIZE="${SPT_BATCH_SIZE:-16}"
export SPT_QUERY="${SPT_QUERY:-20}"
export SPT_LEARNING_RATE="${SPT_LEARNING_RATE:-0.0001}"
export SPT_VAL_EVERY="${SPT_VAL_EVERY:-1}"
export SPT_PATIENCE="${SPT_PATIENCE:-6}"
export SPT_MAX_EPOCHS="${SPT_MAX_EPOCHS:-0}"
export SPT_MAX_DC="${SPT_MAX_DC:-0.5}"
export SPT_GRAD_CLIP="${SPT_GRAD_CLIP:-1.0}"

echo "Starting SPTnet training on CSD3 Ampere node..."
echo "Model dir: $SPT_MODEL_DIR"
echo "Training data pattern: $SPT_TRAIN_DATA"
echo "Batch size: $SPT_BATCH_SIZE"
echo "Query count: $SPT_QUERY"
echo "Learning rate: $SPT_LEARNING_RATE"
echo "Validation every: $SPT_VAL_EVERY"
echo "Patience: $SPT_PATIENCE"
echo "Max epochs: $SPT_MAX_EPOCHS"
echo "Gradient clip: $SPT_GRAD_CLIP"

mkdir -p "$SPT_MODEL_DIR"

~/envs/SPTnet/bin/python SPTnet_training_cli.py \
    --model-dir "$SPT_MODEL_DIR" \
    --data "$SPT_TRAIN_DATA" \
    --gpus 1 \
    --batch-size "$SPT_BATCH_SIZE" \
    --learning-rate "$SPT_LEARNING_RATE" \
    --query "$SPT_QUERY" \
    --max_dc "$SPT_MAX_DC" \
    --val-every "$SPT_VAL_EVERY" \
    --patience "$SPT_PATIENCE" \
    --max-epochs "$SPT_MAX_EPOCHS" \
    --grad-clip "$SPT_GRAD_CLIP"

echo "Training job completed."
