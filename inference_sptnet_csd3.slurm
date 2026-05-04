#!/bin/bash
#SBATCH -J SPTnet_infer_par           # Job name
#SBATCH -A MPHIL-DIS-SL2-GPU         # Project / Account
#SBATCH -p ampere                    # Partition (queue) name for Ampere A100 nodes
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of scripts to run
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --gres=gpu:1                 # Number of GPUs per node (A100)
#SBATCH --time=00:30:00              # Time limit hrs:min:sec
#SBATCH --output=sptnet_infer_par_%j.out # Standard output log (%j = job ID)
#SBATCH --error=sptnet_infer_par_%j.err  # Standard error log
#SBATCH --export=ALL                 # Export all environment variables to the job

set -euo pipefail

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

log "Job starting: id=${SLURM_JOB_ID:-unknown} host=$(hostname) pwd=$(pwd)"
log "Submit dir: ${SLURM_SUBMIT_DIR:-unset}"
touch "sptnet_infer_par_${SLURM_JOB_ID:-manual}.start"

# ---------------------------------------------------------------------
# Module Loading & Environment Setup 
# ---------------------------------------------------------------------

# Purge existing modules and load standard Ampere environment for Rocky/RHEL 8
log "Loading modules..."
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

# Load the standard Python 3 module (adjust if CSD3 uses a specific python/3.x module)
module load python/3.11.0-icl
log "Activating virtual environment..."
source ~/envs/SPTnet/bin/activate

# ---------------------------------------------------------------------
# Run the application
# ---------------------------------------------------------------------

# Go to the directory where the job was submitted from
cd $SLURM_SUBMIT_DIR
log "Changed to submit dir: $(pwd)"

# Prevent standard python multiprocessing from taking all CPU cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

log "Diagnostic info for virtual environment:"
which python
python -c "import sys; print(sys.path)"
python -c "import torch; print('torch.cuda.is_available=', torch.cuda.is_available())"

log "Starting Parallel SPTnet inference on CSD3 Ampere node..."

~/envs/SPTnet/bin/python -u SPTnet_inference_parallel_cli.py \
    -m ./Trained_models/full_run/trained_model \
    -d "./TestData/tiff_output/testdata_*.tif" \
    -b 8

log "Inference job completed."
