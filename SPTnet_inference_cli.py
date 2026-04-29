import argparse
import sys
import os
import glob
from SPTnet_toolbox import *
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import tifffile
import torch, torch.nn as nn
import numpy as np
from os.path import dirname, basename
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

class TiffDataset(Dataset):
    def __init__(self, tif_path):
        # load the full multi-page TIFF as a (T, H, W) array
        video = tifffile.imread(tif_path)
        if video.ndim == 2:
            raise ValueError(f"{tif_path} contains only one frame; need a time series.")
        self.video = video.astype(np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {'video': self.video}

def parse_args():
    p = argparse.ArgumentParser(description="SPTnet Inference on Colab")
    p.add_argument('-m', '--model-path', type=str, required=True, help="Path to the trained model file (e.g. .../trained_model)")
    p.add_argument('-d', '--data', type=str, nargs='+', required=True, help="Path(s) to test data files (.mat or .tif)")
    return p.parse_args()

def main():
    args = parse_args()
    
    model_path = args.model_path
    if not os.path.isfile(model_path):
        # Check if user provided directory instead of file path
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

    # Initialize Toolbox
    spt = SPTnet_toolbox(
        path_saved_model=model_path,
        momentum=0.9,
        learning_rate=0.0002,
        batch_size=96,
        use_gpu=torch.cuda.is_available(),
        image_size=64, # Default from original script
        number_of_frame=30, # Default from original script
        num_queries= 20 # Default from original script
    )

    # Model Setup
    transformer3d = Transformer3d(d_model=256,dropout=0,nhead=8,dim_feedforward=1024,num_encoder_layers=6,num_decoder_layers=6,normalize_before=False)
    transformer = Transformer(d_model=256,dropout=0,nhead=8,dim_feedforward=1024,num_encoder_layers=6,num_decoder_layers=6,normalize_before=False)
    
    print("Initializing model...")
    model = spt.SPTnet(num_classes=1, num_queries=spt.num_queries, num_frames=30, spatial_t=transformer,  temporal_t=transformer3d, input_channel = 512).to(device)

    # Run Inference
    for i, file in enumerate(filename_test):
        print(f"Processing ({i+1}/{len(filename_test)}): {file}")
        ext = os.path.splitext(file)[1].lower()
        base_name = os.path.splitext(basename(file))[0]
        
        try:
            if ext in ['.tif', '.tiff']:
                data_test = TiffDataset(file)
            else:
                # Assuming .mat file compatible with toolbox
                try:
                    data_test = spt.Transformer_mat2python(SPTnet_toolbox=spt, dataset_path=file)
                except Exception as e:
                    print(f"  Error loading MAT file with Toolbox: {e}")
                    continue
            
            test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, num_workers=0)
            
            spt.inference_with_SPTnet(model, test_dataloader)
            
            if not spt.total_obj_est:
                print("  Warning: No estimation results returned.")
                continue

            estimation_obj = np.vstack(spt.total_obj_est)
            estimation_xy = np.vstack(spt.total_xy_est)
            estimation_H = np.vstack(spt.total_H_est)
            estimation_C = np.vstack(spt.total_C_est)
            
            # Create output directory relative to model location
            output_dir = os.path.join(dirname(model_path), 'inference_results')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"  Created output directory: {output_dir}")
                
            output_filename = 'result_' + base_name + '.mat'
            output_path = os.path.join(output_dir, output_filename)
            
            sio.savemat(output_path,
                        mdict={'obj_estimation': estimation_obj, 
                               'estimation_xy': estimation_xy, 
                               'estimation_H': estimation_H,
                               'estimation_C': estimation_C})
            print(f"  Saved results to: {output_path}")
            
        except Exception as e:
            print(f"  Failed to process {file}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
