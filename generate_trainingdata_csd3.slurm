#!/bin/bash
#SBATCH -J sptnet_gen
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=sptnet_gen_%j.out
#SBATCH --error=sptnet_gen_%j.err
#SBATCH --export=ALL

set -euo pipefail

module purge
module load rhel8/global matlab/r2024a

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Override these at submit time if needed, e.g.:
#   sbatch --export=ALL,SPT_NUM_FILES=50,SPT_VIDEOS_PER_FILE=100 generate_trainingdata_csd3.slurm
export SPT_NUM_FILES="${SPT_NUM_FILES:-10}"
export SPT_VIDEOS_PER_FILE="${SPT_VIDEOS_PER_FILE:-100}"
export SPT_OUTPUT_DIR="${SPT_OUTPUT_DIR:-$SLURM_SUBMIT_DIR/TestData/training_data}"

# If submitted as an array, each task writes a separate contiguous block.
# Example: --array=1-100 with SPT_NUM_FILES=10 writes files 1-1000.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -z "${SPT_FILE_START:-}" ]]; then
    export SPT_FILE_START=$(( (SLURM_ARRAY_TASK_ID - 1) * SPT_NUM_FILES + 1 ))
fi

mkdir -p "$SPT_OUTPUT_DIR"

echo "Starting SPTnet training-data generation on $(hostname)"
echo "Output dir: $SPT_OUTPUT_DIR"
echo "Files: $SPT_NUM_FILES, videos/file: $SPT_VIDEOS_PER_FILE"
echo "File start: ${SPT_FILE_START:-1}"

matlab -nodisplay -nojvm -r "try, SPTnet_trainingdata_generator_csd3; catch ME, disp(getReport(ME,'extended')); exit(1); end; exit(0);"

echo "Generation job completed."
