"""
Visualize SPTnet Outputs — Python equivalent of Visualize_SPTnet_Outputs.m

Designed to run inside a Jupyter notebook. Import and call `show_video()`.

Usage in a notebook cell (MAT + GT):
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

Usage for per-TIFF CSD3 inference outputs (no GT):
    ani = show_video(
        test_data_path='TestData/tiff_output/Example_testdata_000.tif',
        results_path='Trained_models/inference_results/result_Example_testdata_000.mat',
        threshold=0.50,
    )
    HTML(ani.to_jshtml())

Or auto-match TIFF/result pairs by index:
    from inference_script import show_tiff_result_by_index
    ani = show_tiff_result_by_index(pair_index=0, threshold=0.50)
    HTML(ani.to_jshtml())

To overlay ground truth for TIFF input:
    ani = show_video(
        test_data_path='TestData/tiff_output/Example_testdata_000.tif',
        results_path='Trained_models/inference_results/result_Example_testdata_000.mat',
        gt_data_path='TestData/Example_testdata.mat',  # contains traceposition/Hlabel/Clabel
        # gt_video_idx defaults to suffix in TIFF name (_000 -> 0)
        threshold=0.50,
    )
    HTML(ani.to_jshtml())

Or visualize split result directly on MAT (no TIFF in visualization path):
    from inference_script import show_mat_with_single_result
    ani = show_mat_with_single_result(
        test_data_mat_path='TestData/Example_testdata.mat',
        result_mat_path='Trained_models/inference_results/result_Example_testdata_000.mat',
        # video_idx defaults to suffix in result filename (_000 -> 0)
        threshold=0.50,
    )
    HTML(ani.to_jshtml())
"""

import os
import glob
import re
import numpy as np
import h5py
import scipy.io as sio
import tifffile

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


def _swap_xy_for_track_list(track_list):
    """
    Swap x/y columns for every (T,2) track in a per-video GT list.
    """
    out = []
    for tr in track_list:
        if tr is None:
            out.append(None)
        else:
            out.append(tr[:, [1, 0]])
    return out


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


def load_tiff_data(tiff_path):
    """
    Load a TIFF stack and return as (N, T, H, W) with N=1.
    """
    arr = np.array(tifffile.imread(tiff_path))
    if arr.ndim == 2:
        raise ValueError(f"{tiff_path} contains only one frame; need a time series.")
    if arr.ndim != 3:
        raise ValueError(f"Unexpected TIFF shape {arr.shape} for {tiff_path}.")
    return arr[np.newaxis, ...]


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
               video_idx=0, threshold=0.90, min_track_len=5, interval=200,
               gt_data_path=None, gt_video_idx=None, swap_gt_xy_for_tiff=True):
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
    test_data_path : str
        Path to source video:
        - `.mat` training/test file (with optional GT), or
        - `.tif/.tiff` movie stack (no GT overlay).
    results_path   : str  — path to SPTnet inference output .mat file
    video_idx      : int  — which video (0-based)
    threshold      : float — detection confidence threshold
    min_track_len  : int  — min frames active to display a predicted track
    interval       : int  — ms between frames in the animation
    gt_data_path   : str|None
        Optional GT `.mat` when `test_data_path` is TIFF.
    gt_video_idx   : int|None
        Which sample in `gt_data_path` to use. If None and TIFF filename ends
        with `_<number>.tif`, that number is used.
    swap_gt_xy_for_tiff : bool
        For TIFF input with external GT MAT, swap GT x/y columns to match TIFF
        orientation produced by this repository's `mat_to_tiff.py`.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    test_ext = os.path.splitext(test_data_path)[1].lower()
    f_handle = None

    print(f"Loading test data from {test_data_path}...")
    if test_ext in ['.tif', '.tiff']:
        videos = load_tiff_data(test_data_path)
        gt_pos, gt_H, gt_C = [[]], [[]], [[]]
        N = 1
        print(f"  TIFF stack loaded, shape per video: {videos.shape[1:]}")

        if gt_data_path is not None:
            print(f"Loading GT overlays from {gt_data_path}...")
            _, gt_pos_all, gt_H_all, gt_C_all, gt_handle = load_test_data(gt_data_path)
            num_gt = len(gt_pos_all)

            resolved_gt_idx = gt_video_idx
            if resolved_gt_idx is None:
                stem = os.path.splitext(os.path.basename(test_data_path))[0]
                m = re.search(r'_(\d+)$', stem)
                if m:
                    resolved_gt_idx = int(m.group(1))
                else:
                    resolved_gt_idx = 0

            if resolved_gt_idx < 0 or resolved_gt_idx >= num_gt:
                gt_handle.close()
                raise IndexError(
                    f"gt_video_idx={resolved_gt_idx} out of range for {gt_data_path} "
                    f"(has {num_gt} videos)."
                )

            gt_pos = [gt_pos_all[resolved_gt_idx]]
            gt_H = [gt_H_all[resolved_gt_idx]]
            gt_C = [gt_C_all[resolved_gt_idx]]
            if swap_gt_xy_for_tiff:
                gt_pos = [_swap_xy_for_track_list(gt_pos[0])]
            gt_handle.close()
            print(f"  Using GT video index {resolved_gt_idx}.")
    else:
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

    if f_handle is not None:
        f_handle.close()
    return ani


def find_tiff_result_pairs(
    tiff_pattern='TestData/tiff_output/*.tif',
    result_pattern='Trained_models/inference_results/result_*.mat',
):
    """
    Match TIFF stacks with per-file `result_*.mat` outputs by basename.
    Returns a sorted list of (tiff_path, result_path).
    """
    tiff_files = sorted(glob.glob(tiff_pattern))
    result_files = sorted(glob.glob(result_pattern))

    result_by_stem = {}
    for rp in result_files:
        stem = os.path.splitext(os.path.basename(rp))[0]
        if stem.startswith('result_'):
            stem = stem[len('result_'):]
        result_by_stem[stem] = rp

    pairs = []
    for tp in tiff_files:
        t_stem = os.path.splitext(os.path.basename(tp))[0]
        if t_stem in result_by_stem:
            pairs.append((tp, result_by_stem[t_stem]))
    return pairs


def show_tiff_result_by_index(
    pair_index=0,
    tiff_pattern='TestData/tiff_output/*.tif',
    result_pattern='Trained_models/inference_results/result_*.mat',
    gt_data_path=None,
    gt_video_idx=None,
    threshold=0.90,
    min_track_len=5,
    interval=200,
):
    """
    Convenience wrapper for per-file TIFF + result MAT workflows.
    """
    pairs = find_tiff_result_pairs(tiff_pattern=tiff_pattern, result_pattern=result_pattern)
    if not pairs:
        raise FileNotFoundError(
            f"No matched pairs found for TIFF pattern '{tiff_pattern}' and result pattern '{result_pattern}'."
        )
    if pair_index < 0 or pair_index >= len(pairs):
        raise IndexError(f"pair_index={pair_index} out of range [0, {len(pairs)-1}].")

    tiff_path, result_path = pairs[pair_index]
    print(f"Using pair {pair_index + 1}/{len(pairs)}:")
    print(f"  TIFF:   {tiff_path}")
    print(f"  Result: {result_path}")

    return show_video(
        test_data_path=tiff_path,
        results_path=result_path,
        video_idx=0,
        threshold=threshold,
        min_track_len=min_track_len,
        interval=interval,
        gt_data_path=gt_data_path,
        gt_video_idx=gt_video_idx,
    )


def show_mat_with_single_result(
    test_data_mat_path,
    result_mat_path,
    video_idx=None,
    threshold=0.90,
    min_track_len=5,
    interval=200,
):
    """
    Visualize one split `result_..._NNN.mat` directly on the source MAT video.

    This avoids TIFF loading during visualization. If `video_idx` is None, it
    is inferred from the trailing `_NNN` in `result_mat_path` (fallback: 0).
    """
    if video_idx is None:
        stem = os.path.splitext(os.path.basename(result_mat_path))[0]
        m = re.search(r'_(\d+)$', stem)
        video_idx = int(m.group(1)) if m else 0

    print(f"Loading test data from {test_data_mat_path}...")
    videos, gt_pos, gt_H, gt_C, f_handle = load_test_data(test_data_mat_path)
    N = videos.shape[0]
    print(f"  {N} videos, shape per video: {videos.shape[1:]}")

    if video_idx < 0 or video_idx >= N:
        f_handle.close()
        raise IndexError(f"video_idx={video_idx} out of range [0, {N-1}] for {test_data_mat_path}.")

    print(f"Loading split inference result from {result_mat_path}...")
    obj_est, xy_est, est_H, est_C = load_inference_results(result_mat_path)
    if obj_est.shape[0] != 1:
        f_handle.close()
        raise ValueError(
            f"Expected split result with N=1, got shape {obj_est.shape}. "
            "Use show_video(...) for multi-video result files."
        )

    num_queries = obj_est.shape[2]
    h_row = np.asarray(est_H[0], dtype=float).ravel()
    c_row = np.asarray(est_C[0], dtype=float).ravel()
    print(f"  Using MAT video index {video_idx} with split result index 0.")
    print(
        "  parameter ranges: "
        f"H[{np.nanmin(h_row):.4f}, {np.nanmax(h_row):.4f}]  "
        f"D[{np.nanmin(c_row):.4f}, {np.nanmax(c_row):.4f}]"
    )

    ani = build_animation(
        video_frames=videos[video_idx],
        gt_pos_list=gt_pos[video_idx] if video_idx < len(gt_pos) else [],
        gt_h_list=gt_H[video_idx]     if video_idx < len(gt_H) else [],
        gt_c_list=gt_C[video_idx]     if video_idx < len(gt_C) else [],
        obj_est=obj_est[0],
        xy_est=xy_est[0],
        est_H=est_H[0],
        est_C=est_C[0],
        threshold=threshold,
        min_track_len=min_track_len,
        num_queries=num_queries,
        interval=interval,
    )
    f_handle.close()
    return ani
