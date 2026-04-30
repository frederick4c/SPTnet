import argparse
import sys
import os
import glob
from collections import defaultdict
from os.path import dirname, basename

import h5py
import numpy as np
import scipy.io as sio
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from SPTnet_toolbox import *
from transformer import *
from transformer3d import *

# Set up processing device
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.cuda.empty_cache()

# Device selection logic
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class FileSampleDataset(Dataset):
    """
    Flatten all selected files into individual inference samples while keeping
    file ownership, so we can batch many samples on the GPU and still save one
    output .mat per original file.
    """
    def __init__(self, file_list):
        self.records = []
        self.shape_groups = defaultdict(list)

        for file_path in file_list:
            ext = os.path.splitext(file_path)[1].lower()

            if ext in ['.tif', '.tiff']:
                video = tifffile.imread(file_path)
                if video.ndim == 2:
                    raise ValueError(f"{file_path} contains only one frame; need a time series.")
                shape_key = tuple(video.shape)
                record = {
                    'file_path': file_path,
                    'ext': ext,
                    'sample_idx': 0,
                    'shape_key': shape_key,
                }
                self.shape_groups[shape_key].append(len(self.records))
                self.records.append(record)

            else:
                with h5py.File(file_path, 'r') as f:
                    if 'timelapsedata' not in f:
                        raise KeyError(f"Missing variable 'timelapsedata' in dataset: {file_path}")
                    td = f['timelapsedata']
                    if td.ndim == 3:
                        shape_key = tuple(td.shape)
                        record = {
                            'file_path': file_path,
                            'ext': ext,
                            'sample_idx': 0,
                            'shape_key': shape_key,
                        }
                        self.shape_groups[shape_key].append(len(self.records))
                        self.records.append(record)
                    elif td.ndim == 4:
                        shape_key = tuple(td.shape[1:])
                        for i in range(td.shape[0]):
                            record = {
                                'file_path': file_path,
                                'ext': ext,
                                'sample_idx': i,
                                'shape_key': shape_key,
                            }
                            self.shape_groups[shape_key].append(len(self.records))
                            self.records.append(record)
                    else:
                        raise ValueError(f"'timelapsedata' must be 3D or 4D, got {td.shape} in {file_path}.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        file_path = rec['file_path']
        ext = rec['ext']
        sample_idx = rec['sample_idx']

        if ext in ['.tif', '.tiff']:
            video = tifffile.imread(file_path).astype(np.float32)
        else:
            with h5py.File(file_path, 'r') as f:
                td = f['timelapsedata']
                if td.ndim == 3:
                    video = np.array(td, dtype=np.float32)
                else:
                    video = np.array(td[sample_idx], dtype=np.float32)

        return {
            'video': video,
            'file_path': file_path,
            'sample_idx': sample_idx,
        }

class SubsetByIndices(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

def collate_inference(batch):
    videos = torch.stack([torch.from_numpy(item['video']) for item in batch], dim=0)
    file_paths = [item['file_path'] for item in batch]
    sample_indices = [item['sample_idx'] for item in batch]
    return {
        'video': videos,
        'file_path': file_paths,
        'sample_idx': sample_indices,
    }

def run_batched_inference(model, dataloader):
    results = []
    with torch.no_grad():
        for data in dataloader:
            inputs = data['video'].unsqueeze(1).float().to(device)  # [B,1,T,H,W]

            image_max = inputs.amax(dim=(2, 3, 4), keepdim=True)
            image_min = inputs.amin(dim=(2, 3, 4), keepdim=True)
            inputs = (inputs - image_min) / (image_max - image_min).clamp_min(1e-8)

            class_out, center_out, H_out, C_out = model(inputs)

            class_out = class_out.detach().cpu().numpy()   # [B, Q, T]
            center_out = center_out.detach().cpu().numpy() # [B, Q, T, 2]
            H_out = H_out.detach().cpu().numpy()           # [B, Q, 1]
            C_out = C_out.detach().cpu().numpy()           # [B, Q, 1]

            for i in range(class_out.shape[0]):
                results.append({
                    'file_path': data['file_path'][i],
                    'sample_idx': data['sample_idx'][i],
                    'obj_estimation': class_out[i:i+1],
                    'estimation_xy': center_out[i:i+1],
                    'estimation_H': H_out[i],
                    'estimation_C': C_out[i],
                })
    return results

def get_num_queries(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    # read query_embed size
    for k in sd:
        if "query_embed.weight" in k:
            return sd[k].shape[0]
    raise ValueError("query_embed.weight not found")

def parse_args():
    p = argparse.ArgumentParser(description="SPTnet Parallel Inference on Colab/CSD3")
    p.add_argument('-m', '--model-path', type=str, required=True, help="Path to the trained model file (e.g. .../trained_model)")
    p.add_argument('-d', '--data', type=str, nargs='+', required=True, help="Path(s) to test data files (.mat or .tif)")
    p.add_argument('-b', '--batch-size', type=int, default=8, help="Batch size for parallel inference (default: 8)")
    return p.parse_args()

def main():
    args = parse_args()
    
    model_path = args.model_path
    if not os.path.isfile(model_path):
        possible_file = os.path.join(model_path, "trained_model")
        if os.path.isfile(possible_file):
            model_path = possible_file
            print(f"Warning: Directory provided. Using model file: {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from: {model_path}")

    # Expand data file patterns
    filename_test = []
    if isinstance(args.data, list):
        for data_arg in args.data:
            filename_test.extend(glob.glob(data_arg))
    else:
        filename_test.extend(glob.glob(args.data))

    # Remove duplicates and sort
    filename_test = sorted(list(set(filename_test)))

    if not filename_test:
        print(f"Error: No data files found matching patterns: {args.data}")
        return

    print(f"Found {len(filename_test)} test files.")

    num_q = get_num_queries(model_path)

    spt = SPTnet_toolbox(
        path_saved_model=model_path,
        momentum=0.9,
        learning_rate=0.0002,
        batch_size=96,
        use_gpu=torch.cuda.is_available(),
        image_size=64,
        number_of_frame=30,
        num_queries=num_q
    )

    infer_batch_size = args.batch_size

    transformer3d = Transformer3d(
        d_model=256,
        dropout=0,
        nhead=8,
        dim_feedforward=1024,
        num_encoder_layers=6,
        num_decoder_layers=6,
        normalize_before=False
    )
    transformer = Transformer(
        d_model=256,
        dropout=0,
        nhead=8,
        dim_feedforward=1024,
        num_encoder_layers=6,
        num_decoder_layers=6,
        normalize_before=False
    )
    model = spt.SPTnet(
        num_classes=1,
        num_queries=spt.num_queries,
        num_frames=30,
        spatial_t=transformer,
        temporal_t=transformer3d,
        input_channel=512
    ).to(device)

    model.load_state_dict(torch.load(spt.path_saved_model), strict=False)
    model.eval()

    # Optional: if you really have multiple GPUs, uncomment the next 2 lines.
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    print("Initializing FileSampleDataset...")
    all_samples = FileSampleDataset(filename_test)
    all_results = []

    print(f"Starting batched inference (batch_size={infer_batch_size})...")
    # Batch only samples with the same (T, H, W), otherwise torch.stack will fail.
    for shape_key, indices in all_samples.shape_groups.items():
        subset = SubsetByIndices(all_samples, indices)
        test_dataloader = DataLoader(
            subset,
            batch_size=infer_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_inference
        )
        all_results.extend(run_batched_inference(model, test_dataloader))

    results_by_file = defaultdict(lambda: {
        'obj_estimation': [],
        'estimation_xy': [],
        'estimation_H': [],
        'estimation_C': [],
    })

    for rec in sorted(all_results, key=lambda x: (x['file_path'], x['sample_idx'])):
        results_by_file[rec['file_path']]['obj_estimation'].append(rec['obj_estimation'])
        results_by_file[rec['file_path']]['estimation_xy'].append(rec['estimation_xy'])
        results_by_file[rec['file_path']]['estimation_H'].append(rec['estimation_H'])
        results_by_file[rec['file_path']]['estimation_C'].append(rec['estimation_C'])

    save_dir = os.path.join(dirname(spt.path_saved_model), 'inference_results')
    os.makedirs(save_dir, exist_ok=True)

    for file_path, rec in results_by_file.items():
        base = os.path.splitext(basename(file_path))[0] + '.mat'
        estimation_obj = np.vstack(rec['obj_estimation'])
        estimation_obj = np.expand_dims(estimation_obj, axis=1) # Shape: [N, 1, Q, T] to match MATLAB GUI expectation
        estimation_xy = np.vstack(rec['estimation_xy'])
        estimation_H = np.vstack(rec['estimation_H'])
        estimation_C = np.vstack(rec['estimation_C'])

        output_path = os.path.join(save_dir, 'result_' + basename(base))
        sio.savemat(
            output_path,
            mdict={
                'obj_estimation': estimation_obj,
                'estimation_xy': estimation_xy,
                'estimation_H': estimation_H,
                'estimation_C': estimation_C,
            }
        )

    print(f'Done. Saved inference results to: {save_dir}')

if __name__ == "__main__":
    main()
