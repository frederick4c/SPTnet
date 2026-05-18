#!/usr/bin/env python
"""Evaluate a trained track-diffusion teacher on simulated ground-truth tracks.

Use this script to answer whether the teacher has learned a useful trajectory
prior, and how that prior degrades under localization noise, missing frames, or
shorter observed tracks.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from track_diffusion_common import (  # noqa: E402
    FEATURE_SETS,
    SimulatedTrackDataset,
    build_step_features,
    collate_tracks,
    load_teacher_checkpoint,
    regression_metrics,
)


def parse_args():
    """Parse CLI options for teacher evaluation and stress tests."""

    parser = argparse.ArgumentParser(description="Evaluate a track-only diffusion teacher on simulated GT tracks.")
    parser.add_argument("--checkpoint", required=True, help="Path from train_track_diffusion_teacher.py.")
    parser.add_argument("--data", nargs="+", required=True, help="Simulated .mat files or glob patterns.")
    parser.add_argument("--output-csv", default="", help="Optional per-track prediction CSV.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--min-valid-frames", type=int, default=2)
    parser.add_argument("--coord-scale", type=float, default=0.0, help="Override coordinate scale.")
    parser.add_argument(
        "--feature-set",
        default="",
        choices=[""] + sorted(FEATURE_SETS),
        help="Override checkpoint feature set. Empty means use checkpoint metadata.",
    )
    parser.add_argument("--noise-px", type=float, default=0.0, help="Evaluation localization noise stress test.")
    parser.add_argument("--frame-drop-prob", type=float, default=0.0, help="Evaluation frame dropout stress test.")
    parser.add_argument("--truncate-frames", type=int, default=0, help="Evaluation crop length stress test.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    """Resolve `auto` to CUDA when available, otherwise CPU."""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main():
    """Load a checkpoint, score GT tracks, print metrics, and optionally write CSV."""

    args = parse_args()
    device = choose_device(args.device)
    model, checkpoint = load_teacher_checkpoint(args.checkpoint, device)
    max_diff = float(checkpoint["max_diff"])
    feature_set = args.feature_set or checkpoint.get("feature_set", "basic")
    if model.input_size != len(FEATURE_SETS[feature_set]):
        raise ValueError(
            f"Checkpoint expects {model.input_size} input features, but feature_set={feature_set!r} "
            f"has {len(FEATURE_SETS[feature_set])}. Use the checkpoint feature set or retrain."
        )

    dataset = SimulatedTrackDataset(args.data, min_valid_frames=args.min_valid_frames)
    coord_scale = args.coord_scale or float(checkpoint.get("coord_scale") or (float(dataset.image_size) / 2.0))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_tracks)

    rows = []
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            positions = batch["positions"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            features, step_mask = build_step_features(
                positions,
                valid_mask,
                coord_scale=coord_scale,
                feature_set=feature_set,
                noise_px=args.noise_px,
                frame_drop_prob=args.frame_drop_prob,
                truncate_min_frames=args.truncate_frames,
                truncate_max_frames=args.truncate_frames,
                training=bool(args.noise_px or args.frame_drop_prob or args.truncate_frames),
            )
            pred = model(features, step_mask).cpu().numpy() * max_diff
            target = batch["diffusion"].numpy()
            predictions.append(pred)
            targets.append(target)

            valid_steps = step_mask.sum(dim=1).cpu().numpy()
            for i in range(pred.shape[0]):
                rows.append(
                    {
                        "source": batch["source"][i],
                        "video_index": batch["video_index"][i],
                        "track_index": batch["track_index"][i],
                        "valid_steps": int(valid_steps[i]),
                        "target_D": float(target[i]),
                        "predicted_D": float(pred[i]),
                        "abs_error": float(abs(pred[i] - target[i])),
                    }
                )

    pred_np = np.concatenate(predictions)
    target_np = np.concatenate(targets)
    metrics = regression_metrics(pred_np, target_np)
    print(f"tracks={len(target_np)} mae={metrics['mae']:.6f} rmse={metrics['rmse']:.6f} r2={metrics['r2']:.4f}")
    print(f"target_D mean={target_np.mean():.6f} pred_D mean={pred_np.mean():.6f}")

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["source", "video_index", "track_index", "valid_steps", "target_D", "predicted_D", "abs_error"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
