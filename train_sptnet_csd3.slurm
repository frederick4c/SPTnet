#!/bin/bash
#SBATCH -J SPTnet_train              # Job name
#SBATCH -A MPHIL-DIS-SL2-GPU         # Project / Account
#SBATCH -p ampere                    # Partition (queue) name for Ampere A100 nodes
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of scripts to run
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --gres=gpu:1                 # Number of GPUs per node (A100)
#SBATCH --time=00:30:00              # Time limit hrs:min:sec
#SBATCH --output=sptnet_train_%j.out # Standard output log (%j = job ID)
#SBATCH --error=sptnet_train_%j.err  # Standard error log

# ---------------------------------------------------------------------
# Module Loading & Environment Setup (Adjust to your actual CSD3 setup)
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

echo "Starting SPTnet training on CSD3 Ampere node..."

python SPTnet_training_colab.py \
    --model-dir ./Trained_models/ \
    --data "./TestData/eg_trainingvideos_*.mat" \
    --gpus 1

echo "Training job completed."
