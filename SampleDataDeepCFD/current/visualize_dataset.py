import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import sys

def plot_dataset_head(dataX, dataY, num_samples=3):
    N = min(len(dataX), num_samples)
    if N == 0:
        print("Dataset is empty.")
        return
        
    fig, axs = plt.subplots(N, 4, figsize=(16, 4*N))
    
    # Ensure axs is a 2D array even for a single sample
    if N == 1: 
        axs = [axs]
    
    for i in range(N):
        x = dataX[i]
        y = dataY[i]
        
        sdf1 = x[0].T
        flow_region = x[1].T
        ux = y[0].T
        uy = y[1].T
        p = y[2].T
        
        # Geometry (SDF1)
        im0 = axs[i][0].imshow(sdf1, origin='lower', cmap='seismic')
        axs[i][0].set_title(f'Sample {i} SDF1')
        fig.colorbar(im0, ax=axs[i][0])
        
        im1 = axs[i][1].imshow(ux, origin='lower', cmap='jet')
        axs[i][1].set_title(f'Sample {i} Ux')
        fig.colorbar(im1, ax=axs[i][1])
        
        im2 = axs[i][2].imshow(uy, origin='lower', cmap='jet')
        axs[i][2].set_title(f'Sample {i} Uy')
        fig.colorbar(im2, ax=axs[i][2])
        
        im3 = axs[i][3].imshow(p, origin='lower', cmap='jet')
        axs[i][3].set_title(f'Sample {i} Cp (Pressure)')
        fig.colorbar(im3, ax=axs[i][3])
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the first few samples from a DeepCFD dataset")
    parser.add_argument("--dataX", default="../dataX.pkl", help="Path to full dataset dataX.pkl")
    parser.add_argument("--dataY", default="../dataY.pkl", help="Path to full dataset dataY.pkl")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to visualize from the start")
    args = parser.parse_args()
    
    print(f"Loading {args.dataX} and {args.dataY}...")
    try:
        with open(args.dataX, 'rb') as f:
            dataX = pickle.load(f)
        with open(args.dataY, 'rb') as f:
            dataY = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        sys.exit(1)
        
    print(f"Loaded dataX shape: {dataX.shape}")
    print(f"Loaded dataY shape: {dataY.shape}")
    plot_dataset_head(dataX, dataY, args.num_samples)
