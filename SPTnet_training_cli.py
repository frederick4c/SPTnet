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
import torch.profiler
from transformer import *
from transformer3d import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.cuda.empty_cache()
device = torch.device('cuda:0')

# --- Ampere / Turing GPU optimizations ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

current_folder = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    #define training parameters based user's inputs
    p = argparse.ArgumentParser(description="Training SPTnet with user defined parameters")
    p.add_argument('-b', '--batch-size',  type=int,   default=16,     help="training batch size")
    p.add_argument('-g', '--gpus',        type=int,   default=1,      help="number of GPUs to use")
    p.add_argument('-lr','--learning-rate',type=float, default=0.0001, help="initial learning rate")
    p.add_argument('-m','--model-dir',    type=str,   default='.',    help="where to save/load model")
    p.add_argument('-q', '--query', type=int, default=20, help="number of query")
    p.add_argument('-dc', '--max_dc', type=float, default=0.5, help="the maximum diffusion coefficient among all training data")
    p.add_argument('--val-every', type=int, default=1, help="run validation every N epochs")
    p.add_argument('--patience', type=int, default=6, help="early-stopping patience measured in validation checks")
    p.add_argument('--max-epochs', type=int, default=0, help="maximum epochs to run; 0 means no explicit cap")
    p.add_argument('-d', '--data', type=str, nargs='+', help="Path to training data .mat files")
    return p.parse_args()


def main():
    args = parse_args()
    
    model_name = "trained_model"
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    full_path = os.path.join(args.model_dir, model_name)
    print(f"Model will be saved to: {os.path.abspath(full_path)}")
    
    # Verify write permissions immediately
    try:
        test_file_path = full_path + "_write_test.tmp"
        with open(test_file_path, 'w') as f:
            f.write("Write test successful.")
        os.remove(test_file_path)
        print("✅ Write check passed: Output directory is writable.")
    except Exception as e:
        print(f"❌ Write check failed: Cannot write to {os.path.abspath(full_path)}")
        raise RuntimeError(f"Output directory is not writable: {e}")

    if args.gpus > 0 and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Tk().withdraw() # keep the root window from appearing
    # filename_train = askopenfilename(multiple=True, initialdir=current_folder, title='#######Please select all the Training Data File########') # show an "Open" dialog box and return the path to the selected file
    
    if args.data:
        filename_train = []
        for pattern in args.data:
            filename_train.extend(glob.glob(pattern))
    else:
        raise RuntimeError("No training data provided! Use --data argument.")

    data_train = []

    if not filename_train:
        raise RuntimeError("No training data selected!")

    # Normalize to a Python list
    if isinstance(filename_train, tuple):
        training_files = list(filename_train)
    else:
        # In this specific case, filename_train is already a list of strings
        training_files = filename_train
    # Only read the FIRST file to get image dimensions — reading all 1000 files just for H/W
    # is a huge bottleneck (~minutes) and completely unnecessary.
    with h5py.File(training_files[0], 'r') as f:
        data = f['timelapsedata']
        if data.ndim == 4:
            (n_videos, n_frames, H, W) = data.shape
        elif data.ndim == 3:
            (n_frames, H, W) = data.shape
        else:
            raise ValueError(f"Unexpected timelapsedata shape: {data.shape}")

    spt = SPTnet_toolbox(
        path_saved_model=full_path,
        momentum=0.9,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        use_gpu=(args.gpus > 0 and torch.cuda.is_available()),
        image_size=H,
        number_of_frame=n_frames,
        num_queries= args.query,
        diff_max = args.max_dc
    )

    print(f"Loading training data from {len(training_files)} files...")
    all_datasets = []
    for i, file in enumerate(training_files):
        print(f"Processing file {i+1}/{len(training_files)}: {file}")
        datafile = spt.Transformer_mat2python(SPTnet_toolbox=spt, dataset_path=file)
        all_datasets.append(datafile)
    # Build a FLAT ConcatDataset in one call. Nesting ConcatDataset inside ConcatDataset
    # on every iteration creates O(N) recursion depth, which overflows Python's limit at ~1000 files.
    data_train = torch.utils.data.ConcatDataset(all_datasets)
    print(f"Data loading complete. Total samples: {len(data_train)}")
    train_size = int(len(data_train) * 0.8)
    val_size = len(data_train) - train_size  # remainder goes to val, guarantees they sum correctly
    spt.data_loader(data_train, [train_size, val_size])
    file_path = os.path.join(os.path.dirname(__file__), 'CRLB_H_D_frame.mat')
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Calcualted CRLB matrix file is not found: {file_path}")
    # Otherwise load as usual
    with h5py.File(file_path, 'r') as f:
        CRLB_matrix = f['CRLB_matrix_HD_frame'][()]

    def hungarian_matched_loss(pred_classes, pred_positions, pred_H, pred_C, gt_classes, gt_positions, gt_H, gt_C):
        num_batches, num_queries, num_frames = pred_classes.shape
        # Clamp sigmoid outputs to valid BCE range — can be outside range due to floating point errors
        pred_classes = pred_classes.clamp(1e-6, 1 - 1e-6)
        loss_pb = 0
        total_class_pb = 0
        total_coordi_pb = 0
        total_hurst_pb = 0
        total_diffusion_pb = 0
        fullindex = np.arange(len(pred_classes[0,:,0]))
        gt_positions = gt_positions.permute(0,2,1,3)
        if spt.num_queries <= gt_H.shape[1]:
            raise ValueError(
                f"❌ Number of queries ({pred_H.shape[1]}) must be greater than number of particles ({gt_H.shape[1]}). "
                "Please increase `spt.num_queries` to be > max number of ground truth particles."
            )
        zeros_pd = torch.zeros(spt.batch_size, spt.num_queries-gt_H.shape[1]).cuda()
        gt_H = torch.cat((gt_H, zeros_pd), dim=1)
        gt_C = torch.cat((gt_C, zeros_pd), dim=1)
        for b in range(num_batches):
            total_loss = 0
            non_obj_loss_all = 0
            track_flag = sum(gt_classes[b,:])>=2
            num_tracks = int(sum(track_flag))
            if num_tracks != 0:
                    # Calculate the cost matrix for hungarian matching
                gt_pos_track = gt_positions[b,:, :, :][track_flag,:,:].unsqueeze(0).repeat(num_queries,1,1,1)
                gt_classes_pm = gt_classes[b,:][:,track_flag].permute(1,0)
                class_loss_matrix = F.binary_cross_entropy(pred_classes[b,:,:].view(num_queries,1,num_frames).repeat(1, num_tracks,1), gt_classes_pm.view(1,num_tracks,num_frames).repeat(num_queries, 1,1),reduction='none')
                nan_mask = torch.isnan(gt_pos_track)
                gt_pos_track[nan_mask] = 0
                pred_masked = pred_positions[b, :, :, :].unsqueeze(1).repeat(1,num_tracks,1,1)
                pred_masked[nan_mask] = 0
                pos_loss_matrix = pdist(pred_masked, gt_pos_track)
                pos_loss_matrix = torch.nansum(pos_loss_matrix,dim=2)
                cost_matrix_class_pf = torch.mean(class_loss_matrix,dim=2)
                duration = sum(gt_classes[b, :])[track_flag]
                pos_loss_matrix_allfrm_pf = pos_loss_matrix/duration
                gt_H_nonzero = gt_H[b][track_flag]
                gt_C_nonzero = gt_C[b][track_flag]
                H_idx = torch.clamp((gt_H_nonzero*100).round()-1,min=0,max=98).cpu().numpy().astype(int)
                C_idx = torch.clamp((gt_C_nonzero*spt.diff_max*100).round() - 1, min=0, max=spt.diff_max*100-1).cpu().numpy().astype(int)
                stepidx = duration.cpu().numpy().astype(int)-1
                CRLBweight_H = CRLB_matrix[0, 0, C_idx, H_idx, stepidx] / (CRLB_matrix[0, 0, C_idx, H_idx, spt.number_of_frame-1] + 1e-8)
                CRLBweight_C = CRLB_matrix[1, 1, C_idx, H_idx, stepidx] / (CRLB_matrix[1, 1, C_idx, H_idx, spt.number_of_frame-1] + 1e-8)
                H_loss_matrix = criterion_mae(pred_H[b].view(-1,1).repeat(1, gt_H_nonzero.shape[-1]),gt_H_nonzero.view(1,-1).repeat(pred_H.shape[-1],1)) / torch.tensor(CRLBweight_H).repeat(pred_H.shape[-1],1).cuda().clamp(min=1e-4)
                C_loss_matrix = criterion_mae(pred_C[b].view(-1, 1).repeat(1, gt_C_nonzero.shape[-1]),gt_C_nonzero.view(1, -1).repeat(pred_C.shape[-1], 1)) / torch.tensor(CRLBweight_C).repeat(pred_H.shape[-1],1).cuda().clamp(min=1e-4)
                cost_matrix_all_pf = (cost_matrix_class_pf + 2*pos_loss_matrix_allfrm_pf + 0.5*H_loss_matrix + 0.5*C_loss_matrix).t()
                
                # Safety clamp for SciPy linear_sum_assignment: Replace NaN/Inf with safe max values
                cost_matrix_safe = torch.nan_to_num(cost_matrix_all_pf, nan=1e4, posinf=1e4, neginf=-1e4)

                # Compute the optimal assignment
                row_indices, col_indices = linear_sum_assignment(cost_matrix_safe.cpu().detach().numpy())
                # Calculate the losses for the assigned pairs
                cost_matrix_all_pf = cost_matrix_all_pf[row_indices, col_indices].sum()

                total_class = (cost_matrix_class_pf.t().cpu().detach().numpy()[row_indices,col_indices].sum()) / num_tracks
                total_coordi = (2*pos_loss_matrix_allfrm_pf.t().cpu().detach().numpy()[row_indices,col_indices].sum()) / num_tracks
                total_hurst = (0.5*H_loss_matrix.t().cpu().detach().numpy()[row_indices,col_indices].sum()) / num_tracks
                total_diffusion = (0.5*C_loss_matrix.t().cpu().detach().numpy()[row_indices,col_indices].sum()) / num_tracks

                # Not matched trajectory loss
                non_obj_pre = pred_classes[b,:,:][np.setdiff1d(fullindex, col_indices),:]
                non_obj_loss = F.binary_cross_entropy(non_obj_pre, torch.zeros(non_obj_pre.shape).cuda(),reduction='mean')
                loss_pv = (cost_matrix_all_pf/num_tracks) + non_obj_loss
                loss_pb += loss_pv
                if torch.isnan(loss_pv):
                    print('Tracks', num_tracks)
                    continue
            else:
                non_obj_loss = F.binary_cross_entropy(gt_classes[b,:,:], torch.zeros(gt_classes[b,:,:].shape).cuda(),reduction='mean')
                loss_pb += non_obj_loss
                total_class = 0
                total_coordi = 0
                total_hurst = 0
                total_diffusion = 0
            non_obj_loss_all += non_obj_loss
            total_class_pb += total_class
            total_coordi_pb += total_coordi
            total_hurst_pb += total_hurst
            total_diffusion_pb += total_diffusion
        return loss_pb / num_batches, total_class_pb / num_batches, total_coordi_pb / num_batches, total_hurst_pb / num_batches, total_diffusion_pb / num_batches, non_obj_loss_all / num_batches



    def train_step(batch_idx, data):
        model.train()
        inputs, Hlabel, Clabel, position_label, class_label = data['video'], data['Hlabel'], data['Clabel'], data['position'], data['class_label']
        inputs = torch.unsqueeze(inputs, 1).float().cuda() # float64 is actually "double"
        # Vectorized per-sample min-max normalization across the batch
        img = inputs[:, 0]  # (B, T, H, W)
        mins = img.reshape(img.shape[0], -1).min(dim=1).values[:, None, None, None]
        maxs = img.reshape(img.shape[0], -1).max(dim=1).values[:, None, None, None]
        inputs[:, 0] = (img - mins) / (maxs - mins + 1e-8)

        class_label, position_label, Hlabel, Clabel = class_label.float().cuda(), (position_label/(spt.image_size/2)).float().cuda(), Hlabel.float().cuda(), (Clabel/spt.diff_max).float().cuda()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            class_out, center_out, H_out, C_out = model(inputs)
        # Cast back to float32 for loss computation (BCE is unsafe under autocast)
        class_out = class_out.float()
        center_out = center_out.float()
        H_out = H_out.float().squeeze(-1)  # (B, Q, 1) -> (B, Q)
        C_out = C_out.float().squeeze(-1)  # (B, Q, 1) -> (B, Q)
        t_loss, cl_ls, coor_ls, h_ls, diff_ls, bg_ls = hungarian_matched_loss(class_out, center_out, H_out, C_out, class_label, position_label, Hlabel, Clabel)
        scaler.scale(t_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        t_loss, cl_ls, coor_ls, h_ls, diff_ls, bg_ls = float(t_loss), float(cl_ls), float(coor_ls), float(h_ls), float(diff_ls), float(bg_ls)
        return t_loss, cl_ls, coor_ls, h_ls, diff_ls, bg_ls

    def val_step(batch_idx, data):
        model.eval()
        with torch.no_grad():
            inputs, Hlabel, Clabel, position_label, class_label = data['video'], data['Hlabel'], data['Clabel'], data['position'], data['class_label']
            inputs = torch.unsqueeze(inputs, 1).float().cuda() # float64 is actually "double"
            # Vectorized per-sample min-max normalization across the batch
            img = inputs[:, 0]  # (B, T, H, W)
            mins = img.reshape(img.shape[0], -1).min(dim=1).values[:, None, None, None]
            maxs = img.reshape(img.shape[0], -1).max(dim=1).values[:, None, None, None]
            inputs[:, 0] = (img - mins) / (maxs - mins + 1e-8)

            class_label, position_label, Hlabel, Clabel = class_label.float().cuda(), (position_label/(spt.image_size/2)).float().cuda(), Hlabel.float().cuda(), (Clabel/spt.diff_max).float().cuda()
            with torch.amp.autocast('cuda'):
                class_out, center_out, H_out, C_out = model(inputs)
            # Cast back to float32 for loss computation (BCE is unsafe under autocast)
            class_out = class_out.float()
            center_out = center_out.float()
            H_out = H_out.float().squeeze(-1)  # (B, Q, 1) -> (B, Q)
            C_out = C_out.float().squeeze(-1)  # (B, Q, 1) -> (B, Q)
            v_loss, cl_ls, coor_ls, h_ls, diff_ls, bg_ls = hungarian_matched_loss(class_out, center_out, H_out, C_out, class_label, position_label, Hlabel, Clabel)
            v_loss, cl_ls, coor_ls, h_ls, diff_ls, bg_ls = float(v_loss), float(cl_ls), float(coor_ls), float(h_ls), float(diff_ls), float(bg_ls)
        return v_loss, cl_ls, coor_ls, h_ls, diff_ls, bg_ls

    torch.backends.cudnn.benchmark = True  # use the fastest convolution methods when the inputs size are fixed improves performance
    #torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)
    criterion_mae = nn.L1Loss(reduction='none').to(device)
    pdist = nn.PairwiseDistance(p=2)
    transformer3d = Transformer3d(d_model=256,dropout=0,nhead=8,dim_feedforward=1024,num_encoder_layers=6,num_decoder_layers=6,normalize_before=False)
    transformer = Transformer(d_model=256,dropout=0,nhead=8,dim_feedforward=1024,num_encoder_layers=6,num_decoder_layers=6,normalize_before=False)
    print("Initializing model...")
    model = spt.SPTnet(num_classes=1, num_queries=spt.num_queries, num_frames=spt.number_of_frame, spatial_t=transformer,
                       temporal_t=transformer3d, input_channel=512).to(device)
    # torch.autograd.set_detect_anomaly(True)  # Disabled for performance; re-enable only for debugging
    scaler = torch.amp.GradScaler('cuda')  # AMP gradient scaler for mixed-precision training

    if args.gpus > 1:
        device_ids = list(range(args.gpus))
        model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=spt.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    # scheduler_rdpl = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=5, verbose=True,
    #                                                  threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
    #                                                  eps=1e-08)
    # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-6)
    ##############################################
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
    min_v_loss = 99999999999
    max_num_of_epoch_without_improving = args.patience
    epoch = 1
    #
    start = time.time()
    lr = []

    modelrecord = open(spt.path_saved_model + 'training_log.txt', 'a')
    # Lightweight CSV log — writing a matplotlib figure on every epoch is slow on a headless node.
    csv_log_path = spt.path_saved_model + 'loss_history.csv'
    if not os.path.exists(csv_log_path):
        with open(csv_log_path, 'w') as csv_f:
            csv_f.write('epoch,t_loss,v_loss,t_cls,v_cls,t_coor,v_coor,t_hurst,v_hurst,t_diff,v_diff,t_bg,v_bg\n')
    while no_improvement < max_num_of_epoch_without_improving and (args.max_epochs <= 0 or epoch <= args.max_epochs):
    # for epoch in range(n_epochs):
        print(f"Starting Epoch {epoch}...")
        epoch_list.append(epoch)
        t_loss_total = 0
        t_loss_total_cls = 0
        t_loss_total_coor = 0
        t_loss_total_hurst = 0
        t_loss_total_diff = 0
        t_loss_total_bg = 0
        v_loss_total = 0
        v_loss_total_cls = 0
        v_loss_total_coor = 0
        v_loss_total_hurst = 0
        v_loss_total_diff = 0
        v_loss_total_bg = 0
        pbar = tqdm(spt.train_dataloader, desc=f"Epoch {epoch} [train]")
        for batch_idx, data in enumerate(pbar):
            t_loss, cl_ls, coor_ls, h_ls, diff_ls, bg_ls = train_step(batch_idx, data)
            t_loss_total += t_loss
            t_loss_total_cls += cl_ls
            t_loss_total_coor += coor_ls
            t_loss_total_hurst += h_ls
            t_loss_total_diff += diff_ls
            t_loss_total_bg += bg_ls
            pbar.set_postfix({'loss': f'{t_loss:.4f}'})
        t_loss_epoch = t_loss_total/(batch_idx+1)
        t_loss_epoch_cls  = t_loss_total_cls/(batch_idx+1)
        t_loss_epoch_coor = t_loss_total_coor/(batch_idx+1)
        t_loss_epoch_hurst = t_loss_total_hurst/(batch_idx+1)
        t_loss_epoch_diff = t_loss_total_diff/(batch_idx+1)
        t_loss_epoch_bg = t_loss_total_bg / (batch_idx + 1)
            # lr.append(scheduler_rdpl.get_lr()[0])

        did_validate = (epoch == 1) or (args.val_every > 0 and epoch % args.val_every == 0)
        if did_validate:
            for batch_idx, data in enumerate(spt.val_dataloader):
                v_loss, cl_ls, coor_ls, h_ls, diff_ls, bg_ls = val_step(batch_idx, data)
                v_loss_total+=v_loss
                v_loss_total_cls += cl_ls
                v_loss_total_coor += coor_ls
                v_loss_total_hurst += h_ls
                v_loss_total_diff += diff_ls
                v_loss_total_bg += bg_ls
            v_loss_epoch = v_loss_total / (batch_idx + 1)
            v_loss_epoch_cls = v_loss_total_cls / (batch_idx + 1)
            v_loss_epoch_coor = v_loss_total_coor / (batch_idx + 1)
            v_loss_epoch_hurst = v_loss_total_hurst / (batch_idx + 1)
            v_loss_epoch_diff = v_loss_total_diff / (batch_idx + 1)
            v_loss_epoch_bg = v_loss_total_bg / (batch_idx + 1)
            print(f"Finished validation for Epoch {epoch}. Loss: {v_loss_epoch:.4f}")
        else:
            v_loss_epoch = float('nan')
            v_loss_epoch_cls = float('nan')
            v_loss_epoch_coor = float('nan')
            v_loss_epoch_hurst = float('nan')
            v_loss_epoch_diff = float('nan')
            v_loss_epoch_bg = float('nan')
            print(f"Skipping validation for Epoch {epoch}; next validation check is controlled by --val-every {args.val_every}.")

        if did_validate and v_loss_epoch < min_v_loss:
            min_v_loss = v_loss_epoch
            no_improvement = 0
            if args.gpus > 1:
                torch.save(model.module.state_dict(), spt.path_saved_model) #Save model.module.state_dict() in DP case!!!
            else:
                torch.save(model.state_dict(), spt.path_saved_model)
            torch.save(optimizer.state_dict(), spt.path_saved_model+'optimizer_stat')
            print('==> Saving a new best model')
        elif did_validate:
            no_improvement+=1
        else:
            print(f"Early stopping patience unchanged ({no_improvement}/{max_num_of_epoch_without_improving}) because validation was skipped.")
        lr.append(optimizer.param_groups[0]['lr'])
        # scheduler_rdpl.step(v_loss_epoch)
        print('learning rate is: %f' %lr[-1])

        # ---- Collect per-epoch metrics into lists ----
        t_loss_append.append(t_loss_epoch)
        v_loss_append.append(v_loss_epoch)
        t_loss_epoch_cls_append.append(t_loss_epoch_cls)
        v_loss_epoch_cls_append.append(v_loss_epoch_cls)
        t_loss_epoch_coor_append.append(t_loss_epoch_coor)
        v_loss_epoch_coor_append.append(v_loss_epoch_coor)
        t_loss_epoch_hurst_append.append(t_loss_epoch_hurst)
        v_loss_epoch_hurst_append.append(v_loss_epoch_hurst)
        t_loss_epoch_diff_append.append(t_loss_epoch_diff)
        v_loss_epoch_diff_append.append(v_loss_epoch_diff)
        t_loss_epoch_bg_append.append(t_loss_epoch_bg)
        v_loss_epoch_bg_append.append(v_loss_epoch_bg)

        # ---- Write to text log (identical format to original) ----
        modelrecord.write('\nepoch %d, t_loss: %s, v_loss: %s' % (epoch, t_loss_epoch, v_loss_epoch))
        modelrecord.write(', t_cls_loss: %s, v_cls_loss: %s' % (t_loss_epoch_cls, v_loss_epoch_cls))
        modelrecord.write(', t_coor_loss: %s, v_coor_loss: %s' % (t_loss_epoch_coor, v_loss_epoch_coor))
        modelrecord.write(', t_hurst_loss: %s, v_hurst_loss: %s' % (t_loss_epoch_hurst, v_loss_epoch_hurst))
        modelrecord.write(', t_diff_loss: %s, v_diff_loss: %s' % (t_loss_epoch_diff, v_loss_epoch_diff))
        modelrecord.write(', t_bg_loss: %s, v_bg_loss: %s' % (t_loss_epoch_bg, v_loss_epoch_bg))
        modelrecord.flush()  # ensure log is written even if the job is killed

        # ---- Write to CSV log (easy to load with pandas later) ----
        with open(csv_log_path, 'a') as csv_f:
            csv_f.write(f'{epoch},{t_loss_epoch},{v_loss_epoch},{t_loss_epoch_cls},{v_loss_epoch_cls},'
                        f'{t_loss_epoch_coor},{v_loss_epoch_coor},{t_loss_epoch_hurst},{v_loss_epoch_hurst},'
                        f'{t_loss_epoch_diff},{v_loss_epoch_diff},{t_loss_epoch_bg},{v_loss_epoch_bg}\n')

        # ---- Save learning curve plot every 5 epochs or when a new best model is saved ----
        if epoch % 5 == 0 or (did_validate and no_improvement == 0):
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
            ax[0,0].plot(epoch_list, t_loss_append, 'r', lw=2, label='train')
            ax[0,0].plot(epoch_list, v_loss_append, 'b', lw=2, label='val')
            ax[0,0].set_title('Total loss'); ax[0,0].legend()
            ax[0,1].plot(epoch_list, t_loss_epoch_cls_append, 'r', lw=2)
            ax[0,1].plot(epoch_list, v_loss_epoch_cls_append, 'b', lw=2)
            ax[0,1].set_title('cls loss')
            ax[0,2].plot(epoch_list, t_loss_epoch_coor_append, 'r', lw=2)
            ax[0,2].plot(epoch_list, v_loss_epoch_coor_append, 'b', lw=2)
            ax[0,2].set_title('coordinate loss')
            ax[1,0].plot(epoch_list, t_loss_epoch_hurst_append, 'r', lw=2)
            ax[1,0].plot(epoch_list, v_loss_epoch_hurst_append, 'b', lw=2)
            ax[1,0].set_title('hurst loss')
            ax[1,1].plot(epoch_list, t_loss_epoch_diff_append, 'r', lw=2)
            ax[1,1].plot(epoch_list, v_loss_epoch_diff_append, 'b', lw=2)
            ax[1,1].set_title('diffusion loss')
            ax[1,2].plot(epoch_list, t_loss_epoch_bg_append, 'r', lw=2)
            ax[1,2].plot(epoch_list, v_loss_epoch_bg_append, 'b', lw=2)
            ax[1,2].set_title('bg loss')
            plt.tight_layout()
            plt.savefig(spt.path_saved_model + 'learning_curve.png')
            plt.close('all')
        print("(""epoch", epoch, ")", "Training Loss", t_loss_epoch, "Validation Loss", v_loss_epoch)
        epoch+=1
    end = time.time()
    print("...Done Training...")
    print("...Training takes %d s..." % (end - start))

    modelrecord.write('\n...Training for %d epoch...\nThe minimal validation loss is %s\n' % (
    epoch, min_v_loss))
    modelrecord.close()

if __name__ == '__main__':
    main()
