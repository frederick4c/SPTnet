#!/usr/bin/env python
"""Train the diffusion-first track teacher.

This script tests the most isolated version of the idea: can a model infer the
diffusion constant from a clean simulated trajectory alone? It never reads image
pixels or SPTnet predictions. The resulting checkpoint can later be used as a
frozen physics critic for predicted tracks.
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from track_diffusion_common import (  # noqa: E402
    FEATURE_SETS,
    SimulatedTrackDataset,
    TrackDiffusionEstimator,
    build_step_features,
    collate_tracks,
    regression_metrics,
)


def parse_args():
    """Parse CLI options for teacher training and stress-test augmentation."""

    parser = argparse.ArgumentParser(
        description="Pretrain a track-only diffusion teacher from simulated ground-truth tracks."
    )
    parser.add_argument("--data", nargs="+", required=True, help="Simulated training .mat files or glob patterns.")
    parser.add_argument("--output", default="experiments/track_diffusion_teacher.pt", help="Checkpoint path.")
    parser.add_argument("--metrics-csv", default="", help="Optional CSV log path.")
    parser.add_argument("--max-diff", type=float, default=0.5, help="Maximum diffusion coefficient for target scaling.")
    parser.add_argument("--coord-scale", type=float, default=0.0, help="Position scale; default is image_size / 2.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=68)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--feature-set",
        default="physics_v1",
        choices=sorted(FEATURE_SETS),
        help="Track features given to the teacher. physics_v1 adds MSD-informed squared-step features.",
    )
    parser.add_argument("--min-valid-frames", type=int, default=2)
    parser.add_argument("--noise-px", type=float, default=0.0, help="Training-only localization noise augmentation.")
    parser.add_argument("--frame-drop-prob", type=float, default=0.0, help="Training-only random frame dropout.")
    parser.add_argument("--truncate-min-frames", type=int, default=0, help="Training-only random crop min length.")
    parser.add_argument("--truncate-max-frames", type=int, default=0, help="Training-only random crop max length.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    """Resolve `auto` to CUDA when available, otherwise CPU."""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable train/validation splits."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, args, device, coord_scale, training):
    """Run one train or validation epoch and return loss/MAE/RMSE/R2 metrics.

    Targets are scaled by `max_diff` before the loss because the teacher outputs
    a sigmoid-normalized value. Reported metrics are converted back into physical
    diffusion units.
    """

    model.train(training)
    losses = []
    predictions = []
    targets = []
    loss_fn = nn.SmoothL1Loss()

    iterator = tqdm(loader, desc="train" if training else "val", leave=False)
    for batch in iterator:
        positions = batch["positions"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        target = (batch["diffusion"].to(device) / args.max_diff).clamp(0.0, 1.0)

        features, step_mask = build_step_features(
            positions,
            valid_mask,
            coord_scale=coord_scale,
            feature_set=args.feature_set,
            noise_px=args.noise_px,
            frame_drop_prob=args.frame_drop_prob,
            truncate_min_frames=args.truncate_min_frames,
            truncate_max_frames=args.truncate_max_frames,
            training=training,
        )

        pred = model(features, step_mask)
        loss = loss_fn(pred, target)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        losses.append(float(loss.detach().cpu()))
        predictions.append(pred.detach().cpu().numpy() * args.max_diff)
        targets.append(batch["diffusion"].numpy())
        iterator.set_postfix(loss=f"{losses[-1]:.4f}")

    pred_np = np.concatenate(predictions)
    target_np = np.concatenate(targets)
    metrics = regression_metrics(pred_np, target_np)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def main():
    """Load tracks, train the teacher, and save the best validation checkpoint."""

    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    dataset = SimulatedTrackDataset(args.data, min_valid_frames=args.min_valid_frames)
    coord_scale = args.coord_scale if args.coord_scale > 0 else float(dataset.image_size) / 2.0

    indices = np.arange(len(dataset))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(len(indices) * args.val_fraction)))
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_tracks,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_tracks,
    )

    model_config = {
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "input_size": len(FEATURE_SETS[args.feature_set]),
    }
    model = TrackDiffusionEstimator(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.metrics_csv:
        Path(args.metrics_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["epoch", "train_loss", "train_mae", "train_rmse", "train_r2", "val_loss", "val_mae", "val_rmse", "val_r2"])

    print(f"Loaded {len(dataset)} GT tracks from {len(dataset.data_paths)} file(s).")
    print(f"Device: {device}; coord_scale: {coord_scale:.4g}; max_diff: {args.max_diff:.4g}")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, args, device, coord_scale, training=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, args, device, coord_scale, training=False)

        print(
            f"epoch {epoch:03d} "
            f"train_mae={train_metrics['mae']:.5f} val_mae={val_metrics['mae']:.5f} "
            f"val_rmse={val_metrics['rmse']:.5f} val_r2={val_metrics['r2']:.4f}"
        )

        if args.metrics_csv:
            with open(args.metrics_csv, "a", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    [
                        epoch,
                        train_metrics["loss"],
                        train_metrics["mae"],
                        train_metrics["rmse"],
                        train_metrics["r2"],
                        val_metrics["loss"],
                        val_metrics["mae"],
                        val_metrics["rmse"],
                        val_metrics["r2"],
                    ]
                )

        if val_metrics["mae"] < best_val:
            best_val = val_metrics["mae"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "max_diff": args.max_diff,
                    "coord_scale": coord_scale,
                    "feature_set": args.feature_set,
                    "feature_names": FEATURE_SETS[args.feature_set],
                    "best_val_mae": best_val,
                    "epoch": epoch,
                    "data": dataset.data_paths,
                },
                args.output,
            )
            print(f"saved best checkpoint to {args.output}")


if __name__ == "__main__":
    main()
