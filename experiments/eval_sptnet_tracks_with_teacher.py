#!/usr/bin/env python
"""Score saved SPTnet inference tracks with the frozen diffusion teacher.

This is a diagnostic bridge between the old SPTnet outputs and the
diffusion-first idea. It loads `result_*.mat` files, keeps sufficiently confident
predicted tracks, feeds their coordinate sequences to the teacher, and compares
the teacher-derived diffusion with SPTnet's own `estimation_C` head.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.io as sio
import torch

sys.path.append(os.path.dirname(__file__))
from track_diffusion_common import (  # noqa: E402
    FEATURE_SETS,
    build_step_features,
    collate_tracks,
    expand_paths,
    iter_batches,
    load_teacher_checkpoint,
)


def parse_args():
    """Parse CLI options for filtering SPTnet result tracks and writing scores."""

    parser = argparse.ArgumentParser(
        description="Run the frozen track-diffusion teacher on saved SPTnet inference tracks."
    )
    parser.add_argument("--checkpoint", required=True, help="Teacher checkpoint.")
    parser.add_argument("--results", nargs="+", required=True, help="SPTnet result .mat files or glob patterns.")
    parser.add_argument("--output-csv", default="experiments/sptnet_tracks_teacher_scores.csv")
    parser.add_argument("--obj-threshold", type=float, default=0.5)
    parser.add_argument("--min-valid-frames", type=int, default=3)
    parser.add_argument(
        "--pred-coord-scale",
        type=float,
        default=1.0,
        help="Scale for saved estimation_xy. Use 1.0 for normalized SPTnet coordinates.",
    )
    parser.add_argument(
        "--network-c-normalized",
        action="store_true",
        default=True,
        help="Treat saved estimation_C as C / max_diff, which is how the current SPTnet head is trained.",
    )
    parser.add_argument(
        "--feature-set",
        default="",
        choices=[""] + sorted(FEATURE_SETS),
        help="Override checkpoint feature set. Empty means use checkpoint metadata.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    """Resolve `auto` to CUDA when available, otherwise CPU."""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def normalize_result_arrays(mat: Dict[str, np.ndarray]):
    """Normalize MATLAB result arrays to `xy=[N,Q,T,2]`, `obj=[N,Q,T]`, `C=[N,Q]`.

    Different inference scripts save a singleton dimension in slightly different
    places. This helper removes the harmless singleton axes and validates the
    shapes used by the scoring path.
    """

    xy = np.asarray(mat["estimation_xy"], dtype=np.float32)
    obj = np.asarray(mat["obj_estimation"], dtype=np.float32)
    net_c = np.asarray(mat.get("estimation_C", np.full(xy.shape[:2], np.nan)), dtype=np.float32)

    if xy.ndim == 5 and xy.shape[1] == 1:
        xy = xy[:, 0]
    if xy.ndim != 4 or xy.shape[-1] != 2:
        raise ValueError(f"estimation_xy must resolve to [N,Q,T,2], got {xy.shape}")

    if obj.ndim == 4 and obj.shape[1] == 1:
        obj = obj[:, 0]
    if obj.ndim != 3:
        raise ValueError(f"obj_estimation must resolve to [N,Q,T], got {obj.shape}")

    net_c = np.squeeze(net_c)
    if net_c.ndim == 1 and xy.shape[0] == 1:
        net_c = net_c[None, :]
    if net_c.ndim == 3 and net_c.shape[-1] == 1:
        net_c = net_c[..., 0]
    if net_c.shape[:2] != xy.shape[:2]:
        net_c = np.full(xy.shape[:2], np.nan, dtype=np.float32)

    return xy, obj, net_c


def collect_tracks(path: str, obj_threshold: float, min_valid_frames: int, max_diff: float, network_c_normalized: bool):
    """Extract confident predicted tracks from one saved SPTnet result file."""

    mat = sio.loadmat(path)
    xy, obj, net_c = normalize_result_arrays(mat)
    records: List[Dict[str, object]] = []

    for sample_idx in range(xy.shape[0]):
        for query_idx in range(xy.shape[1]):
            valid = obj[sample_idx, query_idx] >= obj_threshold
            valid &= np.isfinite(xy[sample_idx, query_idx]).all(axis=1)
            if int(valid.sum()) < min_valid_frames:
                continue
            c_value = float(net_c[sample_idx, query_idx])
            if network_c_normalized and np.isfinite(c_value):
                c_value *= max_diff
            records.append(
                {
                    "positions": xy[sample_idx, query_idx],
                    "valid_mask": valid,
                    "diffusion": np.float32(0.0),
                    "hurst": np.float32(0.0),
                    "source": path,
                    "video_index": sample_idx,
                    "track_index": query_idx,
                    "network_D": c_value,
                    "valid_frames": int(valid.sum()),
                    "mean_obj": float(obj[sample_idx, query_idx, valid].mean()),
                }
            )
    return records


def summarize(values: np.ndarray) -> Dict[str, float]:
    """Return compact distribution statistics for finite numeric values."""

    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "p05": float("nan"), "p95": float("nan")}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def main():
    """Load the teacher, score all selected SPTnet tracks, and write a CSV report."""

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
    result_paths = expand_paths(args.results)

    all_records: List[Dict[str, object]] = []
    for path in result_paths:
        records = collect_tracks(
            path,
            obj_threshold=args.obj_threshold,
            min_valid_frames=args.min_valid_frames,
            max_diff=max_diff,
            network_c_normalized=args.network_c_normalized,
        )
        all_records.extend(records)
        print(f"{path}: kept {len(records)} predicted tracks")

    if not all_records:
        raise RuntimeError("No predicted tracks passed the objectness/min-frame filters.")

    rows = []
    with torch.no_grad():
        for batch_items in iter_batches(all_records, args.batch_size):
            batch = collate_tracks(batch_items)
            positions = batch["positions"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            features, step_mask = build_step_features(
                positions,
                valid_mask,
                coord_scale=args.pred_coord_scale,
                feature_set=feature_set,
                training=False,
            )
            teacher_d = model(features, step_mask).cpu().numpy() * max_diff
            valid_steps = step_mask.sum(dim=1).cpu().numpy()

            for i, item in enumerate(batch_items):
                rows.append(
                    {
                        "source": item["source"],
                        "sample_index": item["video_index"],
                        "query_index": item["track_index"],
                        "valid_frames": item["valid_frames"],
                        "valid_steps": int(valid_steps[i]),
                        "mean_obj": item["mean_obj"],
                        "teacher_D": float(teacher_d[i]),
                        "network_D": float(item["network_D"]),
                        "teacher_network_abs_diff": float(abs(teacher_d[i] - item["network_D"]))
                        if np.isfinite(item["network_D"])
                        else float("nan"),
                    }
                )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "source",
                "sample_index",
                "query_index",
                "valid_frames",
                "valid_steps",
                "mean_obj",
                "teacher_D",
                "network_D",
                "teacher_network_abs_diff",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    teacher_values = np.asarray([row["teacher_D"] for row in rows], dtype=np.float64)
    network_values = np.asarray([row["network_D"] for row in rows], dtype=np.float64)
    diff_values = np.asarray([row["teacher_network_abs_diff"] for row in rows], dtype=np.float64)
    print(f"wrote {args.output_csv}")
    print(f"teacher_D summary: {summarize(teacher_values)}")
    print(f"network_D summary: {summarize(network_values)}")
    print(f"abs difference summary: {summarize(diff_values)}")


if __name__ == "__main__":
    main()
