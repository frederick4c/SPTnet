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

echo "Starting FULL 1000-file SPTnet training on CSD3 Ampere node..."

# Ensure we are saving to a distinct folder so we don't overwrite the 5-file test run
mkdir -p ./Trained_models/full_run/

~/envs/SPTnet/bin/python SPTnet_training_cli.py \
    --model-dir ./Trained_models/full_run/ \
    --data "./TrainData/*.mat" \
    --gpus 1 \
    --batch-size 16

echo "Training job completed."
