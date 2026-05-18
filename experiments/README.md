# Diffusion-first experiments

These scripts test the smallest useful version of the diffusion-first idea without
changing the main SPTnet training code.

## 1. Train a track-only diffusion teacher

```bash
python experiments/train_track_diffusion_teacher.py \
  --data "TestData/dense_test/trainingvideos_*.mat" \
  --output experiments/track_diffusion_teacher.pt \
  --metrics-csv experiments/track_diffusion_teacher_metrics.csv \
  --epochs 50 \
  --batch-size 256 \
  --max-diff 0.5
```

The model sees only simulated ground-truth `traceposition` tracks and learns to
predict `Clabel`. It does not see images. By default it uses physics-informed
per-step features:

```text
dx, dy, dx^2, dy^2, r^2, step_length, valid_step
```

where `r^2 = dx^2 + dy^2`, the quantity behind classical
mean-squared-displacement diffusion estimates. To reproduce the original
raw-displacement teacher, add `--feature-set basic`.

Useful stress-test augmentations:

```bash
python experiments/train_track_diffusion_teacher.py \
  --data "TestData/dense_test/trainingvideos_*.mat" \
  --output experiments/track_diffusion_teacher_noisy.pt \
  --noise-px 0.5 \
  --frame-drop-prob 0.1 \
  --truncate-min-frames 10 \
  --truncate-max-frames 30
```

## 2. Evaluate the teacher on simulated GT tracks

```bash
python experiments/eval_track_diffusion_teacher.py \
  --checkpoint experiments/track_diffusion_teacher.pt \
  --data "TestData/dense_test/trainingvideos_*.mat" \
  --output-csv experiments/track_diffusion_teacher_eval.csv
```

Short-track stress test:

```bash
python experiments/eval_track_diffusion_teacher.py \
  --checkpoint experiments/track_diffusion_teacher.pt \
  --data "TestData/dense_test/trainingvideos_*.mat" \
  --truncate-frames 10 \
  --noise-px 0.5
```

## 3. Score existing SPTnet inference tracks

Run normal SPTnet inference first, then score the saved `result_*.mat` files:

```bash
python experiments/eval_sptnet_tracks_with_teacher.py \
  --checkpoint experiments/track_diffusion_teacher.pt \
  --results "Trained_models/dense_full/inference_results/result_*.mat" \
  --output-csv experiments/sptnet_tracks_teacher_scores.csv \
  --obj-threshold 0.5 \
  --min-valid-frames 3
```

The saved SPTnet coordinates are normalized, so `--pred-coord-scale 1.0` is the
default. The CSV compares the frozen teacher's `D` inferred from predicted
tracks with the network's existing `estimation_C` head.
