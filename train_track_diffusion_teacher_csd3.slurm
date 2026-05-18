#!/bin/bash
#SBATCH -J SPT_D_teacher_mix
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=track_diffusion_teacher_%j.out
#SBATCH --error=track_diffusion_teacher_%j.err
#SBATCH --export=ALL

# ---------------------------------------------------------------------
# Module Loading & Environment Setup
# ---------------------------------------------------------------------

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.11.0-icl
source ~/envs/SPTnet/bin/activate

# ---------------------------------------------------------------------
# Run from the directory where sbatch was submitted
# Expected layout:
#   ./experiments/
#   ./DenseTrain/*.mat
#   ./TrainData/*.mat
# ---------------------------------------------------------------------

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
ulimit -n 65536

export TEACHER_DENSE_DATA="${TEACHER_DENSE_DATA:-./DenseTrain/*.mat}"
export TEACHER_SPARSE_DATA="${TEACHER_SPARSE_DATA:-./TrainData/*.mat}"
export TEACHER_OUT_DIR="${TEACHER_OUT_DIR:-./experiments/csd3_mixed_teacher}"
export TEACHER_EPOCHS="${TEACHER_EPOCHS:-80}"
export TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-512}"
export TEACHER_LR="${TEACHER_LR:-0.001}"
export TEACHER_MAX_DIFF="${TEACHER_MAX_DIFF:-0.5}"
export TEACHER_NOISE_PX="${TEACHER_NOISE_PX:-0.25}"
export TEACHER_FRAME_DROP_PROB="${TEACHER_FRAME_DROP_PROB:-0.05}"
export TEACHER_TRUNCATE_MIN_FRAMES="${TEACHER_TRUNCATE_MIN_FRAMES:-10}"
export TEACHER_TRUNCATE_MAX_FRAMES="${TEACHER_TRUNCATE_MAX_FRAMES:-30}"
export TEACHER_FEATURE_SET="${TEACHER_FEATURE_SET:-physics_v1}"

mkdir -p "$TEACHER_OUT_DIR"

TEACHER_CHECKPOINT="$TEACHER_OUT_DIR/track_diffusion_teacher_mixed.pt"
TEACHER_METRICS="$TEACHER_OUT_DIR/track_diffusion_teacher_mixed_metrics.csv"
TEACHER_DENSE_EVAL="$TEACHER_OUT_DIR/track_diffusion_teacher_eval_dense.csv"
TEACHER_SPARSE_EVAL="$TEACHER_OUT_DIR/track_diffusion_teacher_eval_sparse.csv"

echo "Starting mixed track-diffusion teacher training on CSD3..."
echo "Submit dir:          $SLURM_SUBMIT_DIR"
echo "Dense data:          $TEACHER_DENSE_DATA"
echo "Sparse data:         $TEACHER_SPARSE_DATA"
echo "Output dir:          $TEACHER_OUT_DIR"
echo "Checkpoint:          $TEACHER_CHECKPOINT"
echo "Epochs:              $TEACHER_EPOCHS"
echo "Batch size:          $TEACHER_BATCH_SIZE"
echo "Learning rate:       $TEACHER_LR"
echo "Max diffusion:       $TEACHER_MAX_DIFF"
echo "Feature set:         $TEACHER_FEATURE_SET"
echo "Noise px:            $TEACHER_NOISE_PX"
echo "Frame drop prob:     $TEACHER_FRAME_DROP_PROB"
echo "Truncate frames:     $TEACHER_TRUNCATE_MIN_FRAMES-$TEACHER_TRUNCATE_MAX_FRAMES"
echo "Python:              $(which python)"

if [[ ! -d experiments ]]; then
    echo "ERROR: ./experiments was not found. Upload the experiments folder to the CSD3 submit directory."
    exit 1
fi

if ! compgen -G "$TEACHER_DENSE_DATA" > /dev/null; then
    echo "ERROR: no dense files matched: $TEACHER_DENSE_DATA"
    exit 1
fi

if ! compgen -G "$TEACHER_SPARSE_DATA" > /dev/null; then
    echo "ERROR: no sparse files matched: $TEACHER_SPARSE_DATA"
    exit 1
fi

python experiments/train_track_diffusion_teacher.py \
    --data "$TEACHER_DENSE_DATA" "$TEACHER_SPARSE_DATA" \
    --output "$TEACHER_CHECKPOINT" \
    --metrics-csv "$TEACHER_METRICS" \
    --epochs "$TEACHER_EPOCHS" \
    --batch-size "$TEACHER_BATCH_SIZE" \
    --learning-rate "$TEACHER_LR" \
    --max-diff "$TEACHER_MAX_DIFF" \
    --feature-set "$TEACHER_FEATURE_SET" \
    --noise-px "$TEACHER_NOISE_PX" \
    --frame-drop-prob "$TEACHER_FRAME_DROP_PROB" \
    --truncate-min-frames "$TEACHER_TRUNCATE_MIN_FRAMES" \
    --truncate-max-frames "$TEACHER_TRUNCATE_MAX_FRAMES" \
    --device cuda

echo "Training complete. Evaluating best checkpoint on dense files..."

python experiments/eval_track_diffusion_teacher.py \
    --checkpoint "$TEACHER_CHECKPOINT" \
    --data "$TEACHER_DENSE_DATA" \
    --output-csv "$TEACHER_DENSE_EVAL" \
    --batch-size "$TEACHER_BATCH_SIZE" \
    --device cuda

echo "Evaluating best checkpoint on sparse files..."

python experiments/eval_track_diffusion_teacher.py \
    --checkpoint "$TEACHER_CHECKPOINT" \
    --data "$TEACHER_SPARSE_DATA" \
    --output-csv "$TEACHER_SPARSE_EVAL" \
    --batch-size "$TEACHER_BATCH_SIZE" \
    --device cuda

echo "Mixed track-diffusion teacher job completed."
echo "Checkpoint:    $TEACHER_CHECKPOINT"
echo "Train metrics: $TEACHER_METRICS"
echo "Dense eval:    $TEACHER_DENSE_EVAL"
echo "Sparse eval:   $TEACHER_SPARSE_EVAL"
