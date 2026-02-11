import os
import argparse
import glob
if 'MPLBACKEND' in os.environ:
    del os.environ['MPLBACKEND']
import matplotlib
matplotlib.use('Agg')
from SPTnet_toolbox import *
from tqdm import tqdm
# from tkinter import Tk
# from tkinter.filedialog import askopenfilename
# from tkinter.filedialog import askdirectory
from scipy.optimize import linear_sum_assignment
from torch.autograd import Variable
import torch.profiler
from transformer import *
from transformer3d import *
import sys
import time
import platform
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.cuda.empty_cache()

# Environment detection and info
print("Running on:", platform.platform())
print("Python:", platform.python_version())
if torch.cuda.is_available():
    print(f"CUDA available. GPUs: {torch.cuda.device_count()}")
    try:
        import subprocess
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], timeout=10)
        print("GPU info:", result.decode().strip())
    except:
        pass
else:
    print("No CUDA detected, using CPU.")

"""
Colab setup (run in notebook before script):
!pip install torch torchvision torchaudio h5py scipy tqdm matplotlib
from google.colab import drive
drive.mount('/content/drive')
python SPTnet_training_grok.py --data "/content/drive/MyDrive/data/**/*.mat" --batch-size 8

CSD3 SLURM example (save as train.sbatch, sbatch train.sbatch):
#!/bin/bash
#SBATCH --job-name=SPTnet_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=train_%j.out

module purge
module load rhel8 default-modules python3 cuda/12.1 pytorch-gpu

python /path/to/SPTnet_training_grok.py --batch-size 32 --gpus 4 --data "/rds/project/YOUR/data/**/*.mat"
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_folder = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    p = argparse.ArgumentParser(description="Optimized SPTnet training for Colab/CSD3")
    p.add_argument('-b', '--batch-size',  type=int,   default=16,     help="training batch size")
    p.add_argument('-g', '--gpus',        type=int,   default=None,   help="number of GPUs (None=auto)")
    p.add_argument('-lr','--learning-rate',type=float, default=0.0001, help="initial learning rate")
    p.add_argument('-m','--model-dir',    type=str,   default='models', help="where to save/load model (subdir)")
    p.add_argument('-q', '--query', type=int, default=20, help="number of query")
    p.add_argument('-dc', '--max_dc', type=float, default=0.5, help="the maximum diffusion coefficient")
    p.add_argument('-d', '--data', type=str, nargs='+', default=[], help="Path to training data .mat files")
    p.add_argument('--resume', type=str, default=None, help="Path to model.pt for resume")
    return p.parse_args()

def main():
    args = parse_args()

    # Auto-detect GPUs
    if args.gpus is None:
        args.gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    args.gpus = min(args.gpus, torch.cuda.device_count() or 0)
    print(f"Using {args.gpus} GPUs")

    model_dir = os.path.join(current_folder, args.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    model_name = "SPTnet_trained_model.pt"
    full_path = os.path.join(model_dir, model_name)
    optimizer_path = full_path.replace('.pt', '_optimizer.pt')
    print(f"Model will be saved to: {os.path.abspath(full_path)}")

    # Write test
    try:
        test_file = full_path + "_test.tmp"
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print("✅ Model dir writable")
    except Exception as e:
        raise RuntimeError(f"Model dir not writable: {e}")

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpus > 0 else 'cpu')

    if args.data:
        filename_train = []
        for pattern in args.data:
            filename_train.extend(glob.glob(pattern, recursive=True))
    else:
        raise RuntimeError("No --data provided!")

    if not filename_train:
        raise RuntimeError("No training files found!")

    # Get data shape from first file
    with h5py.File(filename_train[0], 'r') as f:
        data = f['timelapsedata'][()]
        n_videos, n_frames, H, W = data.shape

    print(f"Data shape: {n_videos} videos, {n_frames} frames, {H}x{W}")

    spt = SPTnet_toolbox(
        path_saved_model=full_path,
        momentum=0.9,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        use_gpu=args.gpus > 0,
        image_size=H,
        number_of_frame=n_frames,
        num_queries=args.query,
        diff_max=args.max_dc
    )

    print(f"Loading {len(filename_train)} data files...")
    data_train = []
    for i, fp in enumerate(tqdm(filename_train)):
        datafile = spt.Transformer_mat2python(SPTnet_toolbox=spt, dataset_path=fp)
        data_train = torch.utils.data.ConcatDataset([data_train, datafile])
    print(f"Loaded {len(data_train)} samples")

    spt.data_loader(data_train, [int(0.8 * len(data_train)), int(0.2 * len(data_train))])

    crlb_path = os.path.join(os.path.dirname(__file__), 'CRLB_H_D_frame.mat')
    if not os.path.exists(crlb_path):
        raise FileNotFoundError(crlb_path)
    with h5py.File(crlb_path, 'r') as f:
        CRLB_matrix = f['CRLB_matrix_HD_frame'][()]

    def hungarian_matched_loss(pred_classes, pred_positions, pred_H, pred_C, gt_classes, gt_positions, gt_H, gt_C):
        # [same as original]
        num_batches, num_queries, num_frames = pred_classes.shape
        loss_pb = 0
        total_class_pb = 0
        total_coordi_pb = 0
        total_hurst_pb = 0
        total_diffusion_pb = 0
        fullindex = np.arange(num_queries)
        gt_positions = gt_positions.permute(0,2,1,3)
        if spt.num_queries <= gt_H.shape[1]:
            raise ValueError(f"Queries {spt.num_queries} <= GT particles {gt_H.shape[1]}")
        zeros_pd = torch.zeros(spt.batch_size, spt.num_queries - gt_H.shape[1], device=gt_H.device)
        gt_H = torch.cat((gt_H, zeros_pd), dim=1)
        gt_C = torch.cat((gt_C, zeros_pd), dim=1)
        for b in range(num_batches):
            total_loss = 0
            track_flag = (gt_classes[b].sum(dim=0) >= 2)
            num_tracks = int(track_flag.sum())
            if num_tracks > 0:
                gt_pos_track = gt_positions[b][track_flag].unsqueeze(0).repeat(num_queries, 1, 1, 1)
                gt_classes_pm = gt_classes[b][:, track_flag].permute(1, 0)
                class_loss_matrix = F.binary_cross_entropy(pred_classes[b].view(num_queries,1,num_frames).repeat(1, num_tracks,1), gt_classes_pm.view(1,num_tracks,num_frames).repeat(num_queries,1,1), reduction='none')
                nan_mask = torch.isnan(gt_pos_track)
                gt_pos_track[nan_mask] = 0
                pred_masked = pred_positions[b].unsqueeze(1).repeat(1, num_tracks, 1, 1)
                pred_masked[nan_mask] = 0
                pos_loss_matrix = torch.norm(pred_masked - gt_pos_track, dim=-1).sum(dim=-1)
                cost_matrix_class_pf = class_loss_matrix.mean(dim=-1)
                duration = gt_classes[b].sum(dim=0)[track_flag]
                pos_loss_matrix_allfrm_pf = pos_loss_matrix / duration.unsqueeze(0)
                gt_H_nonzero = gt_H[b][track_flag]
                gt_C_nonzero = gt_C[b][track_flag]
                H_idx = torch.clamp((gt_H_nonzero * 100).round() - 1, 0, 98).cpu().numpy().astype(int)
                C_idx = torch.clamp((gt_C_nonzero * spt.diff_max * 100).round() - 1, 0, int(spt.diff_max * 100) - 1).cpu().numpy().astype(int)
                stepidx = (duration - 1).cpu().numpy().astype(int)
                CRLBweight_H = CRLB_matrix[0,0,C_idx,H_idx,stepidx] / CRLB_matrix[0,0,C_idx,H_idx,spt.number_of_frame-1]
                CRLBweight_C = CRLB_matrix[1,1,C_idx,H_idx,stepidx] / CRLB_matrix[1,1,C_idx,H_idx,spt.number_of_frame-1]
                H_loss_matrix = F.l1_loss(pred_H[b].unsqueeze(1).repeat(1, gt_H_nonzero.shape[-1]), gt_H_nonzero.unsqueeze(0).repeat(pred_H.shape[-1], 1), reduction='none') / torch.tensor(CRLBweight_H, device=device).unsqueeze(0).repeat(pred_H.shape[-1], 1)
                C_loss_matrix = F.l1_loss(pred_C[b].unsqueeze(1).repeat(1, gt_C_nonzero.shape[-1]), gt_C_nonzero.unsqueeze(0).repeat(pred_C.shape[-1], 1), reduction='none') / torch.tensor(CRLBweight_C, device=device).unsqueeze(0).repeat(pred_H.shape[-1], 1)
                cost_matrix_all_pf = (cost_matrix_class_pf + 2 * pos_loss_matrix_allfrm_pf + 0.5 * H_loss_matrix + 0.5 * C_loss_matrix).T
                row_ind, col_ind = linear_sum_assignment(cost_matrix_all_pf.detach().cpu().numpy())
                cost_matrix_all_pf = cost_matrix_all_pf[row_ind, col_ind].sum()
                total_class = cost_matrix_class_pf.T[row_ind, col_ind].sum().item() / num_tracks
                total_coordi = 2 * pos_loss_matrix_allfrm_pf.T[row_ind, col_ind].sum().item() / num_tracks
                total_hurst = 0.5 * H_loss_matrix.T[row_ind, col_ind].sum().item() / num_tracks
                total_diffusion = 0.5 * C_loss_matrix.T[row_ind, col_ind].sum().item() / num_tracks
                non_obj_pre = pred_classes[b][torch.tensor(list(set(fullindex) - set(col_ind)))]
                non_obj_loss = F.binary_cross_entropy(non_obj_pre, torch.zeros_like(non_obj_pre))
                loss_pv = (cost_matrix_all_pf / num_tracks) + non_obj_loss
                loss_pb += loss_pv
            else:
                non_obj_loss = F.binary_cross_entropy(gt_classes[b], torch.zeros_like(gt_classes[b]))
                loss_pb += non_obj_loss
                total_class = total_coordi = total_hurst = total_diffusion = 0
            total_class_pb += total_class
            total_coordi_pb += total_coordi
            total_hurst_pb += total_hurst
            total_diffusion_pb += total_diffusion
        return loss_pb / num_batches, total_class_pb / num_batches, total_coordi_pb / num_batches, total_hurst_pb / num_batches, total_diffusion_pb / num_batches

    def train_step(batch_idx, data):
        model.train()
        inputs, Hlabel, Clabel, position_label, class_label = data['video'], data['Hlabel'], data['Clabel'], data['position'], data['class_label']
        inputs = torch.unsqueeze(inputs, 1).float().to(device)
        for i in range(inputs.shape[0]):
            img_max = inputs[i, 0].max()
            img_min = inputs[i, 0].min()
            inputs[i, 0] = (inputs[i, 0] - img_min) / (img_max - img_min + 1e-8)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            class_out, center_out, H_out, C_out = model(inputs)
            class_out, H_out, C_out = class_out.squeeze(-1), H_out.squeeze(-1), C_out.squeeze(-1)
            class_label = class_label.float().to(device)
            position_label = (position_label / (spt.image_size / 2)).float().to(device)
            Hlabel = Hlabel.float().to(device)
            Clabel = (Clabel / spt.diff_max).float().to(device)
            t_loss, cl_ls, coor_ls, h_ls, diff_ls = hungarian_matched_loss(class_out, center_out, H_out, C_out, class_label, position_label, Hlabel, Clabel)
        optimizer.zero_grad()
        if scaler:
            scaler.scale(t_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            t_loss.backward()
            optimizer.step()
        return t_loss.item(), cl_ls, coor_ls, h_ls, diff_ls

    def val_step(batch_idx, data):
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=scaler is not None):
            inputs, Hlabel, Clabel, position_label, class_label = data['video'], data['Hlabel'], data['Clabel'], data['position'], data['class_label']
            inputs = torch.unsqueeze(inputs, 1).float().to(device)
            for i in range(inputs.shape[0]):
                img_max = inputs[i, 0].max()
                img_min = inputs[i, 0].min()
                inputs[i, 0] = (inputs[i, 0] - img_min) / (img_max - img_min + 1e-8)
            class_out, center_out, H_out, C_out = model(inputs)
            class_out, H_out, C_out = class_out.squeeze(-1), H_out.squeeze(-1), C_out.squeeze(-1)
            class_label = class_label.float().to(device)
            position_label = (position_label / (spt.image_size / 2)).float().to(device)
            Hlabel = Hlabel.float().to(device)
            Clabel = (Clabel / spt.diff_max).float().to(device)
            v_loss, cl_ls, coor_ls, h_ls, diff_ls = hungarian_matched_loss(class_out, center_out, H_out, C_out, class_label, position_label, Hlabel, Clabel)
        return v_loss.item(), cl_ls, coor_ls, h_ls, diff_ls

    torch.backends.cudnn.benchmark = True
    criterion_mae = nn.L1Loss(reduction='none').to(device)
    pdist = nn.PairwiseDistance(p=2)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() and args.gpus > 0 else None
    transformer3d = Transformer3d(d_model=256, dropout=0, nhead=8, dim_feedforward=1024, num_encoder_layers=6, num_decoder_layers=6, normalize_before=False)
    transformer = Transformer(d_model=256, dropout=0, nhead=8, dim_feedforward=1024, num_encoder_layers=6, num_decoder_layers=6, normalize_before=False)
    print("Initializing model...")
    model = spt.SPTnet(num_classes=1, num_queries=spt.num_queries, num_frames=spt.number_of_frame, spatial_t=transformer, temporal_t=transformer3d, input_channel=512).to(device)
    if args.gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(args.gpus)))
    optimizer = optim.AdamW(model.parameters(), lr=spt.learning_rate, weight_decay=0.01)
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt)
        opt_ckpt = torch.load(args.resume.replace('.pt', '_optimizer.pt'), map_location='cpu')
        optimizer.load_state_dict(opt_ckpt)
        print(f"Resumed training from {args.resume}")
        model = model.to(device)
        if args.gpus > 1:
            model = nn.DataParallel(model, device_ids=list(range(args.gpus)))

    # Full optimized training loop (AMP, device safe, plots, early stop)
    t_loss_append = []
    t_loss_epoch_cls_append = []
    t_loss_epoch_coor_append = []
    t_loss_epoch_hurst_append = []
    t_loss_epoch_diff_append = []
    t_loss_epoch_bg_append = []
    v_loss_append = []
    v_loss_epoch_cls_append = []
    v_loss_epoch_coor_append = []
    v_loss_epoch_hurst_append = []
    v_loss_epoch_diff_append = []
    v_loss_epoch_bg_append = []
    epoch_list = []
    no_improvement = 0
    min_v_loss = float('inf')
    max_num_of_epoch_without_improving = 6
    epoch = 1
    start = time.time()
    lr = []
    log_path = spt.path_saved_model + '_training_log.txt'
    modelrecord = open(log_path, 'w')
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
    while no_improvement < max_num_of_epoch_without_improving:
        epoch_list.append(epoch)
        t_loss_total = 0
        t_loss_total_cls = 0
        t_loss_total_coor = 0
        t_loss_total_hurst = 0
        t_loss_total_diff = 0
        t_loss_total_bg = 0
        pbar = tqdm(spt.train_dataloader, desc=f'Epoch {epoch}')
        for batch_idx, data in enumerate(spt.train_dataloader):
            t_loss, cl_ls, coor_ls, h_ls, diff_ls, bg_ls = train_step(batch_idx, data)
            t_loss_total += t_loss
            t_loss_total_cls += cl_ls
            t_loss_total_coor += coor_ls
            t_loss_total_hurst += h_ls
            t_loss_total_diff += diff_ls
            t_loss_total_bg += bg_ls
            pbar.set_postfix({'loss': f'{t_loss:.4f}'})
        pbar.close()
        t_loss_epoch = t_loss_total / len(spt.train_dataloader)
        t_loss_epoch_cls = t_loss_total_cls / len(spt.train_dataloader)
        t_loss_epoch_coor = t_loss_total_coor / len(spt.train_dataloader)
        t_loss_epoch_hurst = t_loss_total_hurst / len(spt.train_dataloader)
        t_loss_epoch_diff = t_loss_total_diff / len(spt.train_dataloader)
        t_loss_epoch_bg = t_loss_total_bg / len(spt.train_dataloader)
        # Validation
        v_loss_total = 0
        v_loss_total_cls = 0
        v_loss_total_coor = 0
        v_loss_total_hurst = 0
        v_loss_total_diff = 0
        v_loss_total_bg = 0
        for batch_idx, data in enumerate(spt.val_dataloader):
            v_loss, cl_ls, coor_ls, h_ls, diff_ls, bg_ls = val_step(batch_idx, data)
            v_loss_total += v_loss
            v_loss_total_cls += cl_ls
            v_loss_total_coor += coor_ls
            v_loss_total_hurst += h_ls
            v_loss_total_diff += diff_ls
            v_loss_total_bg += bg_ls
        v_loss_epoch = v_loss_total / len(spt.val_dataloader)
        v_loss_epoch_cls = v_loss_total_cls / len(spt.val_dataloader)
        v_loss_epoch_coor = v_loss_total_coor / len(spt.val_dataloader)
        v_loss_epoch_hurst = v_loss_total_hurst / len(spt.val_dataloader)
        v_loss_epoch_diff = v_loss_total_diff / len(spt.val_dataloader)
        v_loss_epoch_bg = v_loss_total_bg / len(spt.val_dataloader)
        print(f'Epoch {epoch}: Train {t_loss_epoch:.4f}, Val {v_loss_epoch:.4f}')
        if v_loss_epoch < min_v_loss:
            min_v_loss = v_loss_epoch
            no_improvement = 0
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), full_path)
            else:
                torch.save(model.state_dict(), full_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            print('==> New best model saved')
        else:
            no_improvement += 1
        lr.append(optimizer.param_groups[0]['lr'])
        print(f'LR: {lr[-1]:.2e}')
        # Update plots (total)
        t_loss_append.append(t_loss_epoch)
        v_loss_append.append(v_loss_epoch)
        ax[0,0].clear()
        ax[0,0].plot(epoch_list, t_loss_append, 'r-', lw=2, label='Train')
        ax[0,0].plot(epoch_list, v_loss_append, 'b-', lw=2, label='Val')
        ax[0,0].set_title('Total Loss')
        ax[0,0].legend()
        modelrecord.write(f'\nepoch {epoch}, t_loss: {t_loss_epoch}, v_loss: {v_loss_epoch}')
        # CLS
        t_loss_epoch_cls_append.append(t_loss_epoch_cls)
        v_loss_epoch_cls_append.append(v_loss_epoch_cls)
        ax[0,1].clear()
        ax[0,1].plot(epoch_list, t_loss_epoch_cls_append, 'r-', lw=2)
        ax[0,1].plot(epoch_list, v_loss_epoch_cls_append, 'b-', lw=2)
        ax[0,1].set_title('Class Loss')
        modelrecord.write(f', t_cls: {t_loss_epoch_cls}, v_cls: {v_loss_epoch_cls}')
        # Coor
        t_loss_epoch_coor_append.append(t_loss_epoch_coor)
        v_loss_epoch_coor_append.append(v_loss_epoch_coor)
        ax[0,2].clear()
        ax[0,2].plot(epoch_list, t_loss_epoch_coor_append, 'r-', lw=2)
        ax[0,2].plot(epoch_list, v_loss_epoch_coor_append, 'b-', lw=2)
        ax[0,2].set_title('Coord Loss')
        modelrecord.write(f', t_coor: {t_loss_epoch_coor}, v_coor: {v_loss_epoch_coor}')
        # Hurst
        t_loss_epoch_hurst_append.append(t_loss_epoch_hurst)
        v_loss_epoch_hurst_append.append(v_loss_epoch_hurst)
        ax[1,0].clear()
        ax[1,0].plot(epoch_list, t_loss_epoch_hurst_append, 'r-', lw=2)
        ax[1,0].plot(epoch_list, v_loss_epoch_hurst_append, 'b-', lw=2)
        ax[1,0].set_title('Hurst Loss')
        modelrecord.write(f', t_hurst: {t_loss_epoch_hurst}, v_hurst: {v_loss_epoch_hurst}')
        # Diff
        t_loss_epoch_diff_append.append(t_loss_epoch_diff)
        v_loss_epoch_diff_append.append(v_loss_epoch_diff)
        ax[1,1].clear()
        ax[1,1].plot(epoch_list, t_loss_epoch_diff_append, 'r-', lw=2)
        ax[1,1].plot(epoch_list, v_loss_epoch_diff_append, 'b-', lw=2)
        ax[1,1].set_title('Diffusion Loss')
        modelrecord.write(f', t_diff: {t_loss_epoch_diff}, v_diff: {v_loss_epoch_diff}')
        # BG
        t_loss_epoch_bg_append.append(t_loss_epoch_bg)
        v_loss_epoch_bg_append.append(v_loss_epoch_bg)
        ax[1,2].clear()
        ax[1,2].plot(epoch_list, t_loss_epoch_bg_append, 'r-', lw=2)
        ax[1,2].plot(epoch_list, v_loss_epoch_bg_append, 'b-', lw=2)
        ax[1,2].set_title('BG Loss')
        modelrecord.write(f', t_bg: {t_loss_epoch_bg}, v_bg: {v_loss_epoch_bg}')
        plt.tight_layout()
        plt.savefig(spt.path_saved_model + 'learning_curve.png')
        plt.close(fig)  # Prevent memory leak
        epoch += 1
    end = time.time()
    print("Training complete in {:.1f}s. Best val loss: {:.4f}".format(end - start, min_v_loss))
    modelrecord.write('\nTrained {} epochs. Min val loss: {:.4f}\n'.format(epoch, min_v_loss))
    modelrecord.close()

if __name__ == '__main__':
    main()
