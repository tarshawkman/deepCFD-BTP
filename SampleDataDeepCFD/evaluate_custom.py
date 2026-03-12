import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from Models.UNetEx import UNetEx

def test_custom_airfoil(model_path, dataX_path, dataY_path, output_image_path="comparison.png"):
    # ------------- 1. Setup Device & Model -------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Needs to match the model architecture trained in DeepCFD.py
    kernel_size = 5
    filters = [8, 16, 32, 32]
    bn = False
    wn = False
    
    model = UNetEx(3, 3, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # ------------- 2. Load the Custom Data -------------
    print("Loading custom airfoil data...")
    with open(dataX_path, "rb") as f:
        dataX = pickle.load(f)
    with open(dataY_path, "rb") as f:
        dataY = pickle.load(f)
        
    x_tensor = torch.FloatTensor(dataX).to(device)
    
    # Load scaling factors
    import os
    norm_x_path = os.path.join(os.path.dirname(model_path), "norm_x.pkl")
    norm_y_path = os.path.join(os.path.dirname(model_path), "norm_y.pkl")
    if os.path.exists(norm_x_path) and os.path.exists(norm_y_path):
        norm_x = pickle.load(open(norm_x_path, "rb")).to(device)
        norm_y = pickle.load(open(norm_y_path, "rb")).to(device)
    else:
        norm_x = torch.tensor(1.0).to(device)
        norm_y = torch.tensor(1.0).to(device)
        print("Warning: normalization files not found. Proceeding without scaling.")
    
    x_tensor = x_tensor / norm_x
    y_tensor = torch.FloatTensor(dataY).to(device)
    
    # ------------- 3. Make Prediction -------------
    with torch.no_grad():
        pred_tensor = model(x_tensor)
        
    # Un-normalize prediction to match raw dataY for fair comparison
    pred_tensor = pred_tensor * norm_y
        
    # Convert back to numpy
    pred = pred_tensor.cpu().numpy()[0]   # shape: (3, 172, 79) -> (C, Y, X)
    true = y_tensor.cpu().numpy()[0]      # shape: (3, 172, 79)
    error = np.abs(true - pred)           # shape: (3, 172, 79)
    
    # ------------- 4. Plotting (DeepCFD Paper Style) -------------
    print("Generating comparison plot...")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    
    channels = ["Ux [m/s]", "Uy [m/s]", "p [Pa]"]
    titles = ["simpleFOAM\n(SPVP_Airfoil)", "DeepCFD", "Error"]
    
    for i in range(3): # For each channel (Ux, Uy, P)
        # Calculate min/max for consistent colorbars across True and Pred
        vmin = min(np.min(true[i]), np.min(pred[i]))
        vmax = max(np.max(true[i]), np.max(pred[i]))
        
        # 1. Ground Truth (simpleFOAM/spvp_airfoil)
        ax = axes[i, 0]
        # Transpose [172, 79] -> [79, 172] to match paper's landscape orientation if desired
        # Or leave as is, depending on your array layout. Usually DeepCFD axes are (X, Y)
        im1 = ax.imshow(true[i], cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
        if i == 0: ax.set_title(titles[0], fontsize=16)
        ax.set_ylabel(channels[i], fontsize=14)
        ax.invert_yaxis()
        fig.colorbar(im1, ax=ax, orientation='horizontal', pad=0.15)
        
        # 2. Prediction (DeepCFD)
        ax = axes[i, 1]
        im2 = ax.imshow(pred[i], cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
        if i == 0: ax.set_title(titles[1], fontsize=16)
        ax.invert_yaxis()
        fig.colorbar(im2, ax=ax, orientation='horizontal', pad=0.15)
        
        # 3. Absolute Error
        ax = axes[i, 2]
        im3 = ax.imshow(error[i], cmap='jet', aspect='auto')
        if i == 0: ax.set_title(titles[2], fontsize=16)
        ax.invert_yaxis()
        fig.colorbar(im3, ax=ax, orientation='horizontal', pad=0.15)

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"Success! Saved visualization to: {output_image_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Custom Airfoil DeepCFD Data")
    parser.add_argument("--model", type=str, default="c:/Users/Aaditya Saraf/Desktop/btp/SampleDataDeepCFD/Run/modelWeights.pth", help="Path to trained .pth model")
    parser.add_argument("--dataX", type=str, default="c:/Users/Aaditya Saraf/Desktop/btp/PanelMethodCode/test_out/custom_dataX.pkl", help="Path to custom dataX.pkl")
    parser.add_argument("--dataY", type=str, default="c:/Users/Aaditya Saraf/Desktop/btp/PanelMethodCode/test_out/custom_dataY.pkl", help="Path to custom dataY.pkl")
    parser.add_argument("--out", type=str, default="paper_comparison.png", help="Path to save output image")
    args = parser.parse_args()
    
    test_custom_airfoil(args.model, args.dataX, args.dataY, args.out)
