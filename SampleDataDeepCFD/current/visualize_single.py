import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import sys

def plot_single(dataX, dataY, index=0):
    # dataX, dataY shape might be (N, 3, 172, 79)
    if index >= len(dataX):
        print(f"Error: index {index} is out of bounds for dataset of size {len(dataX)}")
        sys.exit(1)
        
    x = dataX[index]
    y = dataY[index]
    
    # We transpose for visualization depending on how the data was saved.
    # Usually DeepCFD stores (C, nx, ny), if so, shape is (3, 172, 79).
    sdf = x[0]
    u_in = x[1]
    v_in = x[2]
    
    ux = y[0]
    uy = y[1]
    p = y[2]
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    
    im0 = axs[0,0].imshow(sdf, origin='lower', cmap='binary')
    axs[0,0].set_title('SDF / Geometry')
    fig.colorbar(im0, ax=axs[0,0])
    
    im1 = axs[0,1].imshow(u_in, origin='lower', cmap='jet')
    axs[0,1].set_title('Inlet Ux')
    fig.colorbar(im1, ax=axs[0,1])
    
    im2 = axs[0,2].imshow(v_in, origin='lower', cmap='jet')
    axs[0,2].set_title('Inlet Uy')
    fig.colorbar(im2, ax=axs[0,2])
    
    im3 = axs[1,0].imshow(ux, origin='lower', cmap='jet')
    axs[1,0].set_title('Velocity Ux')
    fig.colorbar(im3, ax=axs[1,0])
    
    im4 = axs[1,1].imshow(uy, origin='lower', cmap='jet')
    axs[1,1].set_title('Velocity Uy')
    fig.colorbar(im4, ax=axs[1,1])
    
    im5 = axs[1,2].imshow(p, origin='lower', cmap='jet')
    axs[1,2].set_title('Pressure (Cp)')
    fig.colorbar(im5, ax=axs[1,2])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a single DeepCFD sample")
    parser.add_argument("--dataX", default="dataX.pkl", help="Path to dataX.pkl")
    parser.add_argument("--dataY", default="dataY.pkl", help="Path to dataY.pkl")
    parser.add_argument("--index", type=int, default=0, help="Index of sample to plot")
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
    plot_single(dataX, dataY, args.index)
