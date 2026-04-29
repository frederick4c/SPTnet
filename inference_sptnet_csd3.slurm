#!/bin/bash
#SBATCH -J SPTnet_infer_par           # Job name
#SBATCH -A MPHIL-DIS-SL2-GPU         # Project / Account
#SBATCH -p ampere                    # Partition (queue) name for Ampere A100 nodes
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of scripts to run
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --gres=gpu:1                 # Number of GPUs per node (A100)
#SBATCH --time=00:10:00              # Time limit hrs:min:sec
#SBATCH --output=sptnet_infer_par_%j.out # Standard output log (%j = job ID)
#SBATCH --error=sptnet_infer_par_%j.err  # Standard error log
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

echo "Diagnostic info for virtual environment:"
which python
pip list
python -c "import sys; print(sys.path)"

echo "Starting Parallel SPTnet inference on CSD3 Ampere node..."

~/envs/SPTnet/bin/python SPTnet_inference_parallel_cli.py \
    -m ./Trained_models/Pretrained_model_for_Ti2microscope \
    -d "./TestData/tiff_output/*.tif" \
    -b 8

echo "Inference job completed."
