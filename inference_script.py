"""
Visualize SPTnet Outputs — Python equivalent of Visualize_SPTnet_Outputs.m

Designed to run inside a Jupyter notebook. Import and call `show_video()`.

Usage in a notebook cell:
    from inference_script import show_video
    from IPython.display import HTML

    ani = show_video(
        test_data_path='TestData/Example_testdata.mat',
        results_path='result_Example_testdata.mat',
        video_idx=0,
        threshold=0.50,
    )
    HTML(ani.to_jshtml())

Or loop over several videos:
    for i in range(5):
        ani = show_video(..., video_idx=i)
        display(HTML(ani.to_jshtml()))
"""

import os
import numpy as np
import h5py
import scipy.io as sio

import matplotlib
matplotlib.use('Agg')   # headless — safe both in notebooks and on the cluster
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap


# ─── Parula-like colormap (approximates MATLAB's default) ────────────────────
_parula_data = [
    (0.2422, 0.1504, 0.6603),
    (0.2810, 0.3228, 0.9579),
    (0.1786, 0.5289, 0.9682),
    (0.0689, 0.6948, 0.8394),
    (0.2161, 0.7843, 0.5923),
    (0.6720, 0.7793, 0.2227),
    (0.9970, 0.7659, 0.2199),
    (0.9769, 0.5834, 0.0805),
    (0.9769, 0.3480, 0.0549),
]
parula_cmap = LinearSegmentedColormap.from_list('parula', _parula_data, N=256)


# ─── Data loading ─────────────────────────────────────────────────────────────
def _coerce_video_array(td_array):
    """
    Convert MATLAB v7.3 `timelapsedata` loaded via h5py to (N, T, H, W).

    h5py exposes MATLAB arrays with reversed axis order, so a MATLAB movie
    (H, W, T, N) appears as (N, T, W, H).
    """
    arr = np.array(td_array)
    if arr.ndim == 4:
        return np.transpose(arr, (0, 1, 3, 2))  # (N,T,W,H) -> (N,T,H,W)
    if arr.ndim == 3:
        return np.transpose(arr, (0, 2, 1))[np.newaxis]  # (T,W,H) -> (1,T,H,W)
    raise ValueError(f"Unexpected timelapsedata shape: {arr.shape}")


def _coerce_trace_to_tx2(raw_trace):
    """
    Convert one MATLAB cell trace payload to shape (T,2), preserving column
    order used by MATLAB visualizers.
    """
    arr = np.array(raw_trace)
    if arr.ndim != 2:
        return None
    if arr.shape[0] == 2 and arr.shape[1] >= 2:
        return arr.T
    if arr.shape[1] == 2 and arr.shape[0] >= 2:
        return arr
    return None


def load_test_data(mat_path):
    """
    Load original training/test .mat file (HDF5 v7.3 format).

    Returns
    -------
    videos       : np.ndarray (N, T, H, W)
    gt_positions : list[list[ndarray|None]]  — (T,2) per particle, or None
    gt_H         : list[list[float|None]]
    gt_C         : list[list[float|None]]
    f_handle     : open h5py.File  — caller must close()
    """
    f = h5py.File(mat_path, 'r')

    td = f['timelapsedata']
    videos = _coerce_video_array(td)

    N = videos.shape[0]
    has_gt = all(k in f for k in ['traceposition', 'Hlabel', 'Clabel'])

    gt_positions, gt_H, gt_C = [], [], []

    if has_gt:
        tp_refs = f['traceposition']   # (max_particles, N)
        hl_refs = f['Hlabel']
        cl_refs = f['Clabel']
        max_particles = tp_refs.shape[0]

        for vid_idx in range(N):
            pos_v, h_v, c_v = [], [], []
            for pi in range(max_particles):
                try:
                    h_val = float(np.array(f[hl_refs[pi, vid_idx]][0]).item())
                except Exception:
                    h_val = 0.0

                if h_val == 0:
                    pos_v.append(None); h_v.append(None); c_v.append(None)
                    continue

                try:
                    c_val = float(np.array(f[cl_refs[pi, vid_idx]][0]).item())
                except Exception:
                    c_val = 0.0

                try:
                    raw = np.array(f[tp_refs[pi, vid_idx]])
                    pos = _coerce_trace_to_tx2(raw)
                except Exception:
                    pos = None

                pos_v.append(pos); h_v.append(h_val); c_v.append(c_val)

            gt_positions.append(pos_v)
            gt_H.append(h_v)
            gt_C.append(c_v)
    else:
        for _ in range(N):
            gt_positions.append([]); gt_H.append([]); gt_C.append([])

    return videos, gt_positions, gt_H, gt_C, f


def load_inference_results(mat_path):
    """
    Load SPTnet inference output .mat file and apply the same transforms
    as Visualize_SPTnet_Outputs.m (lines 25-28).

    Returns (all N-indexed, ready to slice by video index)
    -------
    obj_est : (N, T, Q)       — detection confidence per frame per query
    xy_est  : (N, T, Q, 2)   — predicted pixel coords in [0, 64]
    est_H   : (N, Q)          — Hurst exponent
    est_C   : (N, Q)          — diffusion coefficient (scaled by 0.5)
    """
    data = sio.loadmat(mat_path)

    obj_raw = data['obj_estimation']   # (N, 1, Q, T)  from our inference script
    xy_raw  = data['estimation_xy']    # (N, Q, T, 2)
    est_H   = np.squeeze(data['estimation_H'])   # → (N, Q) or (Q,)
    est_C   = np.squeeze(data['estimation_C'])   # → (N, Q) or (Q,)

    # MATLAB line 25: estimation_xy_scale = estimation_xy*32+32
    xy_scaled = xy_raw * 32 + 32                 # pixel coords [0, 64]

    # MATLAB line 26: estimation_C = estimation_C*0.5
    est_C = est_C * 0.5

    # MATLAB line 27: permute([1,3,2,4])  →  swap Q and T axes  →  (N,T,Q,2)
    xy_perm = np.transpose(xy_scaled, (0, 2, 1, 3))

    # MATLAB line 28: squeeze(permute([1,4,3,2]))  →  (N,T,Q,1) → (N,T,Q)
    obj_perm = np.squeeze(np.transpose(obj_raw, (0, 3, 2, 1)))

    # Fix single-video case where squeeze collapses N dim
    if obj_perm.ndim == 2:
        obj_perm = obj_perm[np.newaxis]   # (1, T, Q)
    if est_H.ndim == 1:
        est_H = est_H[np.newaxis]         # (1, Q)
        est_C = est_C[np.newaxis]         # (1, Q)

    return obj_perm, xy_perm, est_H, est_C


# ─── Core animation builder ───────────────────────────────────────────────────

def build_animation(video_frames, gt_pos_list, gt_h_list, gt_c_list,
                    obj_est, xy_est, est_H, est_C,
                    threshold=0.90, min_track_len=5, num_queries=20,
                    interval=200):
    """
    Build a matplotlib FuncAnimation for one video.

    Parameters
    ----------
    video_frames : (T, H, W)
    gt_pos_list  : list of (T,2) arrays or None, one per GT particle
    gt_h_list    : list of float or None
    gt_c_list    : list of float or None
    obj_est      : (T, Q)
    xy_est       : (T, Q, 2)  — pixel coords already scaled to [0,64]
    est_H        : (Q,)
    est_C        : (Q,)  — already * 0.5
    threshold    : detection threshold
    min_track_len: minimum frames above threshold to show a track
    num_queries  : number of query slots
    interval     : ms between frames in the animation

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    T, H, W = video_frames.shape
    cropsize = 5
    d = cropsize / 2.0
    frmlist = np.arange(T)

    # Per-video normalisation (MATLAB lines 44-46)
    vmin, vmax = video_frames.min(), video_frames.max()
    denom = max(vmax - vmin, 1.0)

    # Precompute threshold mask and track lengths
    predict = obj_est > threshold           # (T, Q)
    track_lengths = predict.sum(axis=0)     # (Q,)

    # Parula colours per query
    cmap_vals = [parula_cmap(i / max(num_queries - 1, 1)) for i in range(num_queries)]

    # ── Set up figure with a single axes ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

    # Initialise image artist with frame 0
    frm0 = ((video_frames[0] - vmin) / denom)
    rgb0 = np.stack([frm0, frm0, frm0], axis=-1)
    im = ax.imshow(rgb0, cmap='gray', origin='upper',
                   extent=[0, W, H, 0], vmin=0, vmax=1)

    # We will collect all dynamic artists and clear them each frame
    dynamic_artists = []

    def _draw_frame(frame):
        # Remove previous dynamic artists
        for art in dynamic_artists:
            try:
                art.remove()
            except Exception:
                pass
        dynamic_artists.clear()

        # Update background image
        frm = ((video_frames[frame] - vmin) / denom)
        rgb = np.stack([frm, frm, frm], axis=-1)
        im.set_data(rgb)

        # ── Ground truth (red) ────────────────────────────────────────────
        for pi, pos in enumerate(gt_pos_list):
            if pos is None or pos.shape[0] != T:
                continue
            if np.isnan(pos[frame, 0]):
                continue

            gt_x = pos[:, 0] + 32   # [-32,32] → [0,64]
            gt_y = pos[:, 1] + 32

            # Current position (X marker)
            sc = ax.scatter(gt_x[frame], gt_y[frame], s=100, c='red',
                            marker='x', linewidths=2, zorder=10)
            dynamic_artists.append(sc)

            # Full trajectory (all valid frames)
            valid = ~np.isnan(pos[:, 0])
            vf = frmlist[valid]
            ln, = ax.plot(gt_x[vf], gt_y[vf], '-o', color='red',
                          markersize=2, linewidth=1.5, markerfacecolor='red', zorder=9)
            dynamic_artists.append(ln)

            # H and D labels
            h_val = gt_h_list[pi] if gt_h_list[pi] is not None else 0
            c_val = gt_c_list[pi] if gt_c_list[pi] is not None else 0
            lx, ly = gt_x[frame] - 0.5 * d, gt_y[frame] + 2.5 * d
            t1 = ax.text(lx,       ly, f'H={h_val:.2f},', color='red',
                         fontsize=7, fontweight='bold', zorder=11)
            t2 = ax.text(lx + 2*d, ly, f'D={c_val:.2f}',  color='red',
                         fontsize=7, fontweight='bold', zorder=11)
            dynamic_artists.extend([t1, t2])

        # ── Predictions (parula-coloured per query) ───────────────────────
        for qi in range(num_queries):
            if not predict[frame, qi]:
                continue
            if track_lengths[qi] < min_track_len:
                continue

            color = cmap_vals[qi]
            active_frames = frmlist[predict[:, qi]]

            # Trajectory line
            tx = xy_est[active_frames, qi, 0]
            ty = xy_est[active_frames, qi, 1]
            ln, = ax.plot(tx, ty, '-o', color=color, markersize=2,
                          linewidth=1.5, markerfacecolor=color, zorder=5)
            dynamic_artists.append(ln)

            # Current position circle
            cx, cy = xy_est[frame, qi, 0], xy_est[frame, qi, 1]
            sc = ax.scatter(cx, cy, s=100, facecolors='none',
                            edgecolors=color, linewidths=2, zorder=6)
            dynamic_artists.append(sc)

            # Bounding box
            rect = patches.Rectangle((cx - d, cy - d), cropsize, cropsize,
                                      linewidth=2, edgecolor=color,
                                      facecolor='none', zorder=6)
            ax.add_patch(rect)
            dynamic_artists.append(rect)

            # H and D labels
            t1 = ax.text(cx - 0.5*d, cy - 0.5*d, f'H={est_H[qi]:.4f},',
                         color=color, fontsize=7, fontweight='bold', zorder=7)
            t2 = ax.text(cx + 1.5*d, cy - 0.5*d, f'D={est_C[qi]:.4f}',
                         color=color, fontsize=7, fontweight='bold', zorder=7)
            dynamic_artists.extend([t1, t2])

        return [im] + dynamic_artists

    ani = animation.FuncAnimation(
        fig, _draw_frame, frames=T, interval=interval, blit=True
    )
    return ani


# ─── Public notebook API ──────────────────────────────────────────────────────

def show_video(test_data_path, results_path,
               video_idx=0, threshold=0.90, min_track_len=5, interval=200):
    """
    Load data and return a FuncAnimation for one video.

    Call from a notebook like:
        from inference_script import show_video
        from IPython.display import HTML

        ani = show_video('TestData/Example_testdata.mat',
                         'result_Example_testdata.mat',
                         video_idx=0, threshold=0.50)
        HTML(ani.to_jshtml())

    Parameters
    ----------
    test_data_path : str  — path to original .mat file (video + ground truth)
    results_path   : str  — path to SPTnet inference output .mat file
    video_idx      : int  — which video (0-based)
    threshold      : float — detection confidence threshold
    min_track_len  : int  — min frames active to display a predicted track
    interval       : int  — ms between frames in the animation

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    print(f"Loading test data from {test_data_path}...")
    videos, gt_pos, gt_H, gt_C, f_handle = load_test_data(test_data_path)
    N = videos.shape[0]
    print(f"  {N} videos, shape per video: {videos.shape[1:]}")

    print(f"Loading inference results from {results_path}...")
    obj_est, xy_est, est_H, est_C = load_inference_results(results_path)
    num_queries = obj_est.shape[2]
    print(f"  obj_estimation: {obj_est.shape}  (N, T, Q)")
    print(f"  estimation_xy:  {xy_est.shape}  (N, T, Q, 2)")

    if N != obj_est.shape[0]:
        raise ValueError(
            "Test data and inference results do not match: "
            f"test videos={N}, result videos={obj_est.shape[0]}. "
            "Use the same test .mat that was used to generate this result_*.mat."
        )
    if videos.shape[1] != obj_est.shape[1]:
        raise ValueError(
            "Frame count mismatch between test data and inference results: "
            f"test frames={videos.shape[1]}, result frames={obj_est.shape[1]}."
        )

    idx = video_idx
    if idx >= obj_est.shape[0]:
        raise IndexError(
            f"video_idx={idx} but results only cover {obj_est.shape[0]} videos.")
    if idx >= N:
        raise IndexError(
            f"video_idx={idx} but test data only has {N} videos.")

    h_row = np.asarray(est_H[idx], dtype=float).ravel()
    c_row = np.asarray(est_C[idx], dtype=float).ravel()
    print(
        "  parameter ranges: "
        f"H[{np.nanmin(h_row):.4f}, {np.nanmax(h_row):.4f}]  "
        f"D[{np.nanmin(c_row):.4f}, {np.nanmax(c_row):.4f}]"
    )

    print(f"Building animation for video {idx}  (threshold={threshold})...")
    ani = build_animation(
        video_frames=videos[idx],
        gt_pos_list=gt_pos[idx] if idx < len(gt_pos) else [],
        gt_h_list=gt_H[idx]     if idx < len(gt_H)  else [],
        gt_c_list=gt_C[idx]     if idx < len(gt_C)  else [],
        obj_est=obj_est[idx],
        xy_est=xy_est[idx],
        est_H=est_H[idx],
        est_C=est_C[idx],
        threshold=threshold,
        min_track_len=min_track_len,
        num_queries=num_queries,
        interval=interval,
    )

    f_handle.close()
    return ani
