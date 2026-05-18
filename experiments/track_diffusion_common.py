"""Shared utilities for diffusion-first SPTnet experiments.

The experiment scripts deliberately avoid importing the full SPTnet training
stack. This module extracts simulated ground-truth tracks from the original
MATLAB/HDF5 files, turns absolute positions into displacement sequences, and
defines the small track-only teacher model used to predict diffusion constants.
"""

import glob
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn


@dataclass
class TrackRecord:
    """One ground-truth trajectory and its scalar simulation labels."""

    positions: np.ndarray
    valid_mask: np.ndarray
    diffusion: float
    hurst: float
    source: str
    video_index: int
    track_index: int


def expand_paths(patterns: Sequence[str]) -> List[str]:
    """Expand CLI path/glob arguments and fail early if any path is missing."""

    paths: List[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])
    paths = sorted(os.path.abspath(p) for p in paths)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {missing[:5]}")
    return paths


def _read_scalar_ref(dataset: h5py.File, ref) -> float:
    """Read a scalar value from a MATLAB object reference, returning 0 for empty refs."""

    if ref is None:
        return 0.0
    try:
        if not ref:
            return 0.0
    except TypeError:
        pass
    value = np.asarray(dataset[ref]).reshape(-1)
    if value.size == 0:
        return 0.0
    return float(value[0])


def _read_position_ref(dataset: h5py.File, ref) -> Optional[np.ndarray]:
    """Read a referenced track array and normalize it to shape ``[T, 2]``."""

    try:
        if not ref:
            return None
    except TypeError:
        pass
    pos = np.asarray(dataset[ref])
    if pos.ndim != 2:
        return None
    if pos.shape[-1] == 2:
        return pos.astype(np.float32, copy=False)
    if pos.shape[0] == 2:
        return pos.T.astype(np.float32, copy=False)
    return None


class SimulatedTrackDataset(torch.utils.data.Dataset):
    """Track-only dataset extracted from the original simulated MATLAB files.

    Each item is a single molecule trajectory, not a movie. The dataset reads
    `traceposition`, `Clabel`, and `Hlabel` references from the MATLAB v7.3 HDF5
    files and keeps only tracks with enough finite positions for displacement
    statistics.
    """

    def __init__(self, data_paths: Sequence[str], min_valid_frames: int = 2):
        self.data_paths = expand_paths(data_paths)
        self.records: List[TrackRecord] = []
        self.image_size: Optional[int] = None
        self.max_frames = 0

        for path in self.data_paths:
            self._load_file(path, min_valid_frames=min_valid_frames)

        if not self.records:
            raise RuntimeError("No valid tracks found in the provided simulated data.")

    def _load_file(self, path: str, min_valid_frames: int) -> None:
        """Append valid track records from one simulated `.mat` file."""

        with h5py.File(path, "r") as dataset:
            for required in ("Hlabel", "Clabel", "traceposition", "timelapsedata"):
                if required not in dataset:
                    raise KeyError(f"{path} is missing required variable {required!r}")

            timelapsedata = dataset["timelapsedata"]
            self.image_size = int(timelapsedata.shape[-1])
            n_tracks, n_videos = dataset["Hlabel"].shape

            for video_idx in range(n_videos):
                for track_idx in range(n_tracks):
                    h_ref = dataset["Hlabel"][track_idx, video_idx]
                    c_ref = dataset["Clabel"][track_idx, video_idx]
                    p_ref = dataset["traceposition"][track_idx, video_idx]

                    hurst = _read_scalar_ref(dataset, h_ref)
                    diffusion = _read_scalar_ref(dataset, c_ref)
                    if hurst == 0.0 or diffusion <= 0.0:
                        continue

                    positions = _read_position_ref(dataset, p_ref)
                    if positions is None:
                        continue
                    valid_mask = np.isfinite(positions).all(axis=1)
                    if int(valid_mask.sum()) < min_valid_frames:
                        continue

                    self.max_frames = max(self.max_frames, positions.shape[0])
                    self.records.append(
                        TrackRecord(
                            positions=positions,
                            valid_mask=valid_mask.astype(np.bool_),
                            diffusion=diffusion,
                            hurst=hurst,
                            source=path,
                            video_index=video_idx,
                            track_index=track_idx,
                        )
                    )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        return {
            "positions": record.positions,
            "valid_mask": record.valid_mask,
            "diffusion": np.float32(record.diffusion),
            "hurst": np.float32(record.hurst),
            "source": record.source,
            "video_index": record.video_index,
            "track_index": record.track_index,
        }


def collate_tracks(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Pad variable-length tracks into tensors for a PyTorch DataLoader batch."""

    max_len = max(np.asarray(item["positions"]).shape[0] for item in batch)
    positions = np.full((len(batch), max_len, 2), np.nan, dtype=np.float32)
    valid_mask = np.zeros((len(batch), max_len), dtype=np.bool_)
    diffusion = np.zeros((len(batch),), dtype=np.float32)
    hurst = np.zeros((len(batch),), dtype=np.float32)

    for i, item in enumerate(batch):
        pos = np.asarray(item["positions"], dtype=np.float32)
        mask = np.asarray(item["valid_mask"], dtype=np.bool_)
        positions[i, : pos.shape[0]] = pos
        valid_mask[i, : mask.shape[0]] = mask
        diffusion[i] = float(item["diffusion"])
        hurst[i] = float(item["hurst"])

    return {
        "positions": torch.from_numpy(positions),
        "valid_mask": torch.from_numpy(valid_mask),
        "diffusion": torch.from_numpy(diffusion),
        "hurst": torch.from_numpy(hurst),
        "source": [str(item["source"]) for item in batch],
        "video_index": [int(item["video_index"]) for item in batch],
        "track_index": [int(item["track_index"]) for item in batch],
    }


def build_step_features(
    positions: torch.Tensor,
    valid_mask: torch.Tensor,
    coord_scale: float,
    noise_px: float = 0.0,
    frame_drop_prob: float = 0.0,
    truncate_min_frames: int = 0,
    truncate_max_frames: int = 0,
    training: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert absolute positions to displacement features plus a valid-step mask.

    Parameters use pixel units before `coord_scale`. The returned feature tensor
    has shape `[B, T-1, 3]`: normalized `dx`, normalized `dy`, and a binary flag
    indicating whether that step is valid. Optional noise, frame dropout, and
    truncation are used to stress-test how robustly a teacher can infer diffusion
    from imperfect tracks.
    """

    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError(f"positions must have shape [B,T,2], got {tuple(positions.shape)}")

    pos = torch.nan_to_num(positions.float(), nan=0.0)
    mask = valid_mask.bool().clone()

    if training and noise_px > 0:
        pos = pos + torch.randn_like(pos) * float(noise_px)

    if training and frame_drop_prob > 0:
        keep = torch.rand_like(mask.float()) >= float(frame_drop_prob)
        mask = mask & keep

    if training and truncate_max_frames > 0:
        max_frames = min(int(truncate_max_frames), mask.shape[1])
        min_frames = max(2, int(truncate_min_frames) if truncate_min_frames else max_frames)
        if min_frames > max_frames:
            min_frames = max_frames
        for i in range(mask.shape[0]):
            length = int(torch.randint(min_frames, max_frames + 1, (1,), device=mask.device).item())
            if length >= mask.shape[1]:
                continue
            start = int(torch.randint(0, mask.shape[1] - length + 1, (1,), device=mask.device).item())
            crop_mask = torch.zeros_like(mask[i])
            crop_mask[start : start + length] = True
            mask[i] = mask[i] & crop_mask

    step_mask = mask[:, 1:] & mask[:, :-1]
    displacement = pos[:, 1:] - pos[:, :-1]
    displacement = displacement / float(coord_scale)
    displacement = torch.where(step_mask.unsqueeze(-1), displacement, torch.zeros_like(displacement))
    features = torch.cat([displacement, step_mask.float().unsqueeze(-1)], dim=-1)
    return features, step_mask


class TrackDiffusionEstimator(nn.Module):
    """Small GRU teacher that maps displacement sequences to normalized diffusion."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.gru = nn.GRU(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, step_features: torch.Tensor, step_mask: torch.Tensor) -> torch.Tensor:
        """Return one normalized diffusion estimate per track in the batch."""

        encoded, _ = self.gru(step_features)
        weights = step_mask.float().unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute simple regression metrics in physical diffusion units."""

    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    err = pred - target
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum((target - np.mean(target)) ** 2))
    r2 = 1.0 - float(np.sum(err**2)) / denom if denom > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def load_teacher_checkpoint(path: str, device: torch.device) -> Tuple[TrackDiffusionEstimator, Dict[str, object]]:
    """Load a saved teacher checkpoint and return the model plus metadata."""

    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("model_config", {})
    model = TrackDiffusionEstimator(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def iter_batches(items: Sequence[Dict[str, object]], batch_size: int) -> Iterable[Sequence[Dict[str, object]]]:
    """Yield fixed-size chunks from a Python sequence without making a DataLoader."""

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
