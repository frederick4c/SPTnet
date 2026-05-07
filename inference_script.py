"""
Visualize SPTnet Outputs — Python equivalent of Visualize_SPTnet_Outputs.m

Designed to run inside a Jupyter notebook. Import and call `show_video()`.

Usage in a notebook cell (MAT + GT; same N in test/result):
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
        results_path='Trained_models/full_run/inference_results/result_Example_testdata_000.mat',
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
        results_path='Trained_models/full_run/inference_results/result_Example_testdata_000.mat',
        gt_data_path='TestData/Example_testdata.mat',  # contains traceposition/Hlabel/Clabel
        # by default, GT video index is auto-matched by pixel similarity
        threshold=0.50,
    )
    HTML(ani.to_jshtml())

Recommended for CSD3 MAT-first-clip inference outputs:
    from inference_script import show_mat_result_by_index
    ani = show_mat_result_by_index(
        pair_index=0,  # testdata_000 + result_testdata_000
        mat_clip_index=0,  # first clip in timelapsedata
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


# ─── Directory defaults ──────────────────────────────────────────────────────
# These can be overridden in a notebook with:
#     set_inference_dirs(test_data_dir='TestData/dense_test',
#                        results_dir='Trained_models/.../inference_results')
TEST_DATA_DIR = os.environ.get('SPT_TEST_DATA_DIR', 'TestData')
RESULTS_DIR = os.environ.get(
    'SPT_RESULTS_DIR',
    'Trained_models/full_run/inference_results',
)


def set_inference_dirs(test_data_dir=None, results_dir=None):
    """
    Set default directories used by the result-pair helpers.

    This is useful in notebooks, where you can configure once:

        set_inference_dirs(
            test_data_dir='TestData/dense_test',
            results_dir='Trained_models/dense_test/inference_results',
        )

    and then call `show_mat_result_by_index(...)` without repeatedly passing
    glob patterns.
    """
    global TEST_DATA_DIR, RESULTS_DIR
    if test_data_dir is not None:
        TEST_DATA_DIR = test_data_dir
    if results_dir is not None:
        RESULTS_DIR = results_dir
    print(f"Test data directory: {TEST_DATA_DIR}")
    print(f"Results directory:   {RESULTS_DIR}")


def get_inference_dirs():
    """Return the currently configured default test/result directories."""
    return TEST_DATA_DIR, RESULTS_DIR


def _pattern_from_dir_or_pattern(path_or_pattern, default_pattern):
    """
    Accept either a directory or a glob/file pattern.
    """
    if path_or_pattern is None:
        return default_pattern
    if glob.has_magic(path_or_pattern):
        return path_or_pattern
    _, ext = os.path.splitext(os.path.basename(path_or_pattern.rstrip(os.sep)))
    if os.path.isdir(path_or_pattern) or ext == '':
        return os.path.join(path_or_pattern, default_pattern)
    return path_or_pattern


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


def _normalize_video_stack(videos):
    """
    Per-video min-max normalization for robust clip matching.
    Accepts (T,H,W) or (N,T,H,W), returns float32 with same shape.
    """
    arr = np.asarray(videos, dtype=np.float32)
    if arr.ndim == 3:
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
        return (arr - vmin) / (vmax - vmin + 1e-8)
    if arr.ndim == 4:
        flat = arr.reshape(arr.shape[0], -1)
        vmin = np.nanmin(flat, axis=1)[:, None, None, None]
        vmax = np.nanmax(flat, axis=1)[:, None, None, None]
        return (arr - vmin) / (vmax - vmin + 1e-8)
    raise ValueError(f"Expected 3D or 4D video array, got shape {arr.shape}.")


def _find_best_matching_video_index(single_video, candidate_videos):
    """
    Find candidate index whose video best matches `single_video` by MAE after
    per-video normalization.
    single_video: (T,H,W), candidate_videos: (N,T,H,W)
    Returns (best_idx, best_mae).
    """
    sv = _normalize_video_stack(single_video)
    cv = _normalize_video_stack(candidate_videos)
    if cv.ndim != 4 or sv.ndim != 3:
        raise ValueError("Unexpected input shapes for video matching.")
    if cv.shape[1:] != sv.shape:
        raise ValueError(
            f"Shape mismatch for matching: candidates {cv.shape[1:]} vs single {sv.shape}"
        )
    mae = np.mean(np.abs(cv - sv[np.newaxis, ...]), axis=(1, 2, 3))
    best_idx = int(np.argmin(mae))
    return best_idx, float(mae[best_idx])


def _prediction_alignment_score(video_frames, obj_est, xy_est, threshold=0.5, min_track_len=1):
    """
    Score prediction alignment by mean intensity sampled at predicted active points.
    Higher is better.
    """
    predict = obj_est > threshold  # (T,Q)
    if min_track_len > 1:
        keep_q = predict.sum(axis=0) >= min_track_len
        predict = predict & keep_q[np.newaxis, :]

    ts, qs = np.where(predict)
    if ts.size == 0:
        return float("-inf")

    x = xy_est[ts, qs, 0]
    y = xy_est[ts, qs, 1]
    xi = np.rint(x).astype(int)
    yi = np.rint(y).astype(int)
    H, W = video_frames.shape[1], video_frames.shape[2]
    valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    if not np.any(valid):
        return float("-inf")

    return float(np.mean(video_frames[ts[valid], yi[valid], xi[valid]]))


def _calibrate_prediction_xy(video_frames, obj_est, xy_est, threshold=0.5, min_track_len=1):
    """
    Try (swap/no-swap) x (delta {-1,0,+1}) and pick best prediction alignment.
    Returns (xy_best, best_swap, best_delta, best_score).
    """
    best_score = float("-inf")
    best_swap = False
    best_delta = 0
    best_xy = xy_est

    for swap in (False, True):
        base = xy_est[..., [1, 0]] if swap else xy_est
        for delta in (-1.0, 0.0, 1.0):
            cand = base + delta
            score = _prediction_alignment_score(
                video_frames, obj_est, cand, threshold=threshold, min_track_len=min_track_len
            )
            if score > best_score:
                best_score = score
                best_swap = swap
                best_delta = delta
                best_xy = cand

    return best_xy, best_swap, best_delta, best_score


def _gt_alignment_score(video_frames, gt_pos_list, offset=32.0, swap_xy=False):
    """
    Score GT alignment by mean intensity sampled at GT points after mapping:
    (x,y) -> (x+offset, y+offset), optionally swapping x/y.
    """
    vals = []
    H, W = video_frames.shape[1], video_frames.shape[2]

    for pos in gt_pos_list:
        if pos is None or pos.shape[0] != video_frames.shape[0]:
            continue
        x = pos[:, 1] if swap_xy else pos[:, 0]
        y = pos[:, 0] if swap_xy else pos[:, 1]
        x = x + offset
        y = y + offset
        xi = np.rint(x).astype(int)
        yi = np.rint(y).astype(int)
        t = np.arange(video_frames.shape[0])
        valid = (~np.isnan(x)) & (~np.isnan(y)) & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        if np.any(valid):
            vals.extend(video_frames[t[valid], yi[valid], xi[valid]])

    if not vals:
        return float("-inf")
    return float(np.mean(vals))


def _best_gt_transform_for_idx(video_frames, gt_pos_list, offsets=(32.0, 33.0, 1.0)):
    """
    Pick best (offset, swap_xy) for one GT video index.
    Returns (offset, swap_xy, score).
    """
    best_score = float("-inf")
    best_offset = 32.0
    best_swap = False
    for off in offsets:
        for sw in (False, True):
            sc = _gt_alignment_score(video_frames, gt_pos_list, offset=off, swap_xy=sw)
            if sc > best_score:
                best_score = sc
                best_offset = off
                best_swap = sw
    return best_offset, best_swap, best_score


def _find_best_gt_video_and_transform(video_frames, gt_pos_all, offsets=(32.0, 33.0, 1.0)):
    """
    Search all GT video indices and transforms; return the best match.
    Returns (best_idx, best_offset, best_swap, best_score).
    """
    best = (0, 32.0, False, float("-inf"))
    for idx, pos_list in enumerate(gt_pos_all):
        off, sw, sc = _best_gt_transform_for_idx(video_frames, pos_list, offsets=offsets)
        if sc > best[3]:
            best = (idx, off, sw, sc)
    return best


def _apply_gt_transform_for_builder(gt_pos_list, offset=32.0, swap_xy=False):
    """
    Build transformed GT tracks so downstream builder (which adds +32) is correct.
    """
    out = []
    delta = float(offset) - 32.0
    for pos in gt_pos_list:
        if pos is None:
            out.append(None)
            continue
        p = pos[:, [1, 0]] if swap_xy else pos.copy()
        if abs(delta) > 1e-12:
            p = p + delta
        out.append(p)
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
               gt_data_path=None, gt_video_idx=None, swap_gt_xy_for_tiff=True,
               auto_match_gt_video=True):
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
        Legacy override for TIFF + external GT mode when auto-matching is off.
    auto_match_gt_video : bool
        For TIFF input with external GT MAT and unspecified `gt_video_idx`,
        auto-match the TIFF clip to the best GT video and GT transform
        (offset/swap) by trajectory intensity score.

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
            gt_videos_all, gt_pos_all, gt_H_all, gt_C_all, gt_handle = load_test_data(gt_data_path)
            num_gt = len(gt_pos_all)

            resolved_gt_idx = gt_video_idx
            resolved_gt_offset = 32.0
            resolved_gt_swap = bool(swap_gt_xy_for_tiff)

            if auto_match_gt_video:
                if resolved_gt_idx is None and num_gt > 1:
                    resolved_gt_idx, resolved_gt_offset, resolved_gt_swap, gt_score = _find_best_gt_video_and_transform(
                        videos[0], gt_pos_all
                    )
                    print(
                        f"  Auto-matched GT index {resolved_gt_idx} with "
                        f"offset={resolved_gt_offset:.1f}, swap_xy={resolved_gt_swap} "
                        f"(trajectory score={gt_score:.3f})."
                    )
                elif resolved_gt_idx is None:
                    resolved_gt_idx = 0
                    resolved_gt_offset, resolved_gt_swap, gt_score = _best_gt_transform_for_idx(
                        videos[0], gt_pos_all[resolved_gt_idx]
                    )
                    print(
                        f"  Single GT video; using index 0 with "
                        f"offset={resolved_gt_offset:.1f}, swap_xy={resolved_gt_swap} "
                        f"(trajectory score={gt_score:.3f})."
                    )
                else:
                    resolved_gt_offset, resolved_gt_swap, gt_score = _best_gt_transform_for_idx(
                        videos[0], gt_pos_all[resolved_gt_idx]
                    )
                    print(
                        f"  Using provided gt_video_idx={resolved_gt_idx} with "
                        f"best offset={resolved_gt_offset:.1f}, swap_xy={resolved_gt_swap} "
                        f"(trajectory score={gt_score:.3f})."
                    )
            else:
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

            gt_pos_one = gt_pos_all[resolved_gt_idx]
            gt_H = [gt_H_all[resolved_gt_idx]]
            gt_C = [gt_C_all[resolved_gt_idx]]
            gt_pos = [_apply_gt_transform_for_builder(
                gt_pos_one, offset=resolved_gt_offset, swap_xy=resolved_gt_swap
            )]
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

    result_N = obj_est.shape[0]
    if videos.shape[1] != obj_est.shape[1]:
        raise ValueError(
            "Frame count mismatch between test data and inference results: "
            f"test frames={videos.shape[1]}, result frames={obj_est.shape[1]}."
        )

    src_idx = int(video_idx)
    if src_idx < 0 or src_idx >= N:
        raise IndexError(
            f"video_idx={src_idx} but test data only has {N} videos."
        )

    # Support "split result" mode from MAT-per-file inference where each
    # result_*.mat contains only one inferred clip (N=1), while the source MAT
    # can still contain many clips.
    if result_N == N:
        res_idx = src_idx
    elif result_N == 1 and N >= 1:
        res_idx = 0
        print(
            f"  Split-result mode detected: test videos={N}, result videos=1. "
            f"Using test clip {src_idx} with result index 0."
        )
    else:
        raise ValueError(
            "Test data and inference results do not match: "
            f"test videos={N}, result videos={result_N}. "
            "Use matching files, or pass a split result (N=1) with the source MAT."
        )

    gt_pos_view = gt_pos[src_idx] if src_idx < len(gt_pos) else []
    gt_h_view = gt_H[src_idx] if src_idx < len(gt_H) else []
    gt_c_view = gt_C[src_idx] if src_idx < len(gt_C) else []

    # For MAT input, calibrate GT transform similarly to TIFF+GT mode to avoid
    # x/y swap or offset inconsistencies from MATLAB/Python conventions.
    if test_ext not in ['.tif', '.tiff'] and gt_pos_view:
        gt_off, gt_sw, gt_sc = _best_gt_transform_for_idx(videos[src_idx], gt_pos_view)
        gt_pos_view = _apply_gt_transform_for_builder(gt_pos_view, offset=gt_off, swap_xy=gt_sw)
        print(
            f"  GT mapping: offset={gt_off:.1f}, swap_xy={gt_sw} "
            f"(alignment score={gt_sc:.3f})"
        )

    h_row = np.asarray(est_H[res_idx], dtype=float).ravel()
    c_row = np.asarray(est_C[res_idx], dtype=float).ravel()
    print(
        "  parameter ranges: "
        f"H[{np.nanmin(h_row):.4f}, {np.nanmax(h_row):.4f}]  "
        f"D[{np.nanmin(c_row):.4f}, {np.nanmax(c_row):.4f}]"
    )

    xy_view, pred_swap, pred_delta, pred_score = _calibrate_prediction_xy(
        videos[src_idx], obj_est[res_idx], xy_est[res_idx],
        threshold=threshold, min_track_len=min_track_len
    )
    print(
        f"  Prediction mapping: swap_xy={pred_swap}, delta={pred_delta:+.1f} px "
        f"(alignment score={pred_score:.3f})"
    )

    print(
        f"Building animation for test clip {src_idx} "
        f"(result index {res_idx}, threshold={threshold})..."
    )
    ani = build_animation(
        video_frames=videos[src_idx],
        gt_pos_list=gt_pos_view,
        gt_h_list=gt_h_view,
        gt_c_list=gt_c_view,
        obj_est=obj_est[res_idx],
        xy_est=xy_view,
        est_H=est_H[res_idx],
        est_C=est_C[res_idx],
        threshold=threshold,
        min_track_len=min_track_len,
        num_queries=num_queries,
        interval=interval,
    )

    if f_handle is not None:
        f_handle.close()
    return ani


def find_tiff_result_pairs(
    tiff_pattern=None,
    result_pattern=None,
    test_data_dir=None,
    results_dir=None,
):
    """
    Match TIFF stacks with per-file `result_*.mat` outputs by basename.
    Returns a sorted list of (tiff_path, result_path).
    """
    if tiff_pattern is None:
        base_test_dir = TEST_DATA_DIR if test_data_dir is None else test_data_dir
        tiff_pattern = os.path.join(base_test_dir, '*.tif')
    else:
        tiff_pattern = _pattern_from_dir_or_pattern(tiff_pattern, '*.tif')

    if result_pattern is None:
        base_results_dir = RESULTS_DIR if results_dir is None else results_dir
        result_pattern = os.path.join(base_results_dir, 'result_*.mat')
    else:
        result_pattern = _pattern_from_dir_or_pattern(result_pattern, 'result_*.mat')

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
    tiff_pattern=None,
    result_pattern=None,
    test_data_dir=None,
    results_dir=None,
    gt_data_path=None,
    gt_video_idx=None,
    auto_match_gt_video=True,
    threshold=0.90,
    min_track_len=5,
    interval=200,
):
    """
    Convenience wrapper for per-file TIFF + result MAT workflows.
    """
    pairs = find_tiff_result_pairs(
        tiff_pattern=tiff_pattern,
        result_pattern=result_pattern,
        test_data_dir=test_data_dir,
        results_dir=results_dir,
    )
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
        auto_match_gt_video=auto_match_gt_video,
    )


def show_mat_with_single_result(
    test_data_mat_path,
    result_mat_path,
    video_idx=None,
    auto_match_video_idx=True,
    threshold=0.90,
    min_track_len=5,
    interval=200,
):
    """
    Backward-compatible wrapper for split MAT result visualization.
    Prefer `show_mat_result_by_index(...)` for new workflows.
    """
    if video_idx is None:
        if auto_match_video_idx:
            print(
                "video_idx not provided; using clip 0 by default for split MAT results."
            )
        video_idx = 0

    return show_video(
        test_data_path=test_data_mat_path,
        results_path=result_mat_path,
        video_idx=video_idx,
        threshold=threshold,
        min_track_len=min_track_len,
        interval=interval,
    )


def find_mat_result_pairs(
    test_mat_pattern=None,
    result_pattern=None,
    test_data_dir=None,
    results_dir=None,
):
    """
    Match source MAT files with split result MAT files by basename.
    Example: testdata_000.mat <-> result_testdata_000.mat
    """
    if test_mat_pattern is None:
        base_test_dir = TEST_DATA_DIR if test_data_dir is None else test_data_dir
        test_mat_pattern = os.path.join(base_test_dir, '*.mat')
    else:
        test_mat_pattern = _pattern_from_dir_or_pattern(test_mat_pattern, '*.mat')

    if result_pattern is None:
        base_results_dir = RESULTS_DIR if results_dir is None else results_dir
        result_pattern = os.path.join(base_results_dir, 'result_*.mat')
    else:
        result_pattern = _pattern_from_dir_or_pattern(result_pattern, 'result_*.mat')

    test_files = sorted(glob.glob(test_mat_pattern))
    result_files = sorted(glob.glob(result_pattern))

    result_by_stem = {}
    for rp in result_files:
        stem = os.path.splitext(os.path.basename(rp))[0]
        if stem.startswith('result_'):
            stem = stem[len('result_'):]
        result_by_stem[stem] = rp

    pairs = []
    for tp in test_files:
        t_stem = os.path.splitext(os.path.basename(tp))[0]
        if t_stem in result_by_stem:
            pairs.append((tp, result_by_stem[t_stem]))
    return pairs


def show_mat_result_by_index(
    pair_index=0,
    test_mat_pattern=None,
    result_pattern=None,
    test_data_dir=None,
    results_dir=None,
    mat_clip_index=0,
    threshold=0.90,
    min_track_len=5,
    interval=200,
):
    """
    Convenience wrapper for MAT-only split inference results.
    Uses one source MAT file and its matched `result_*.mat`.
    """
    pairs = find_mat_result_pairs(
        test_mat_pattern=test_mat_pattern,
        result_pattern=result_pattern,
        test_data_dir=test_data_dir,
        results_dir=results_dir,
    )
    if not pairs:
        raise FileNotFoundError(
            f"No matched MAT/result pairs found for '{test_mat_pattern}' and '{result_pattern}'."
        )
    if pair_index < 0 or pair_index >= len(pairs):
        raise IndexError(f"pair_index={pair_index} out of range [0, {len(pairs)-1}].")

    mat_path, result_path = pairs[pair_index]
    print(f"Using MAT/result pair {pair_index + 1}/{len(pairs)}:")
    print(f"  MAT:    {mat_path}")
    print(f"  Result: {result_path}")
    print(f"  MAT clip index: {mat_clip_index}")

    return show_video(
        test_data_path=mat_path,
        results_path=result_path,
        video_idx=mat_clip_index,
        threshold=threshold,
        min_track_len=min_track_len,
        interval=interval,
    )
