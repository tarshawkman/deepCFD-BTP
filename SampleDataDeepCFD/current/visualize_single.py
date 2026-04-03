import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import sys

def plot_single(dataX, dataY, index=0):
    if index >= len(dataX):
        print(f"Error: index {index} out of bounds")
        sys.exit(1)
        
    x = dataX[index]
    y = dataY[index]
    
    # Toy Dataset Ordering: 
    # dataX: Chan 0: SDF1, Chan 1: Mask, Chan 2: Inlet Velocity (SDF2)
    # dataY: Chan 0: Ux (Vertical), Chan 1: Uy (Horizontal), Chan 2: P
    
    sdf1 = x[0]
    mask = x[1]
    inlet = x[2]
    
    ux = y[0] # Vertical
    uy = y[1] # Horizontal
    p = y[2]
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    
    im0 = axs[0,0].imshow(sdf1, origin='lower', cmap='jet')
    axs[0,0].set_title('SDF1 (Obstacle)')
    fig.colorbar(im0, ax=axs[0,0])
    
    im1 = axs[0,1].imshow(mask, origin='lower', cmap='binary')
    axs[0,1].set_title('Flow Region (Mask)')
    fig.colorbar(im1, ax=axs[0,1])
    
    im2 = axs[0,2].imshow(inlet, origin='lower', cmap='jet')
    axs[0,2].set_title('Inlet Velocity (SDF2)')
    fig.colorbar(im2, ax=axs[0,2])
    
    im3 = axs[1,0].imshow(ux, origin='lower', cmap='jet')
    axs[1,0].set_title('Inlet Axis Velocity (Ux)')
    fig.colorbar(im3, ax=axs[1,0])
    
    im4 = axs[1,1].imshow(uy, origin='lower', cmap='jet')
    axs[1,1].set_title('Cross Axis Velocity (Uy)')
    fig.colorbar(im4, ax=axs[1,1])
    
    im5 = axs[1,2].imshow(p, origin='lower', cmap='jet')
    axs[1,2].set_title('Pressure (P)')
    fig.colorbar(im5, ax=axs[1,2])
    
    plt.suptitle(f"Sample Index: {index}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataX", default="dataX.pkl")
    parser.add_argument("--dataY", default="dataY.pkl")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    
    with open(args.dataX, 'rb') as f: dataX = pickle.load(f)
    with open(args.dataY, 'rb') as f: dataY = pickle.load(f)
    print(f"Loaded Shape: {dataX.shape}")
    plot_single(dataX, dataY, args.index)
