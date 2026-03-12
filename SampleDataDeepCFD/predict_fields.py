import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from Models.UNetEx import UNetEx

def predict_and_save(model_path, dataX_path):
    # ------------- 1. Setup Device & Model -------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # UNet Architecture (Must match how it was trained)
    kernel_size = 5
    filters = [8, 16, 32, 32]
    bn = False
    wn = False
    
    model = UNetEx(3, 3, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # ------------- 2. Load the Custom Data -------------
    print(f"Loading input data from: {dataX_path}")
    with open(dataX_path, "rb") as f:
        dataX = pickle.load(f)
        
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
    
    # ------------- 3. Make Prediction -------------
    print("Generating prediction...")
    with torch.no_grad():
        pred_tensor = model(x_tensor)
        
    # Un-normalize prediction
    pred_tensor = pred_tensor * norm_y
        
    # Convert back to numpy
    pred = pred_tensor.cpu().numpy()[0]   # shape: (3, 79, 172) -> (C, Y, X)
    
    ux = pred[0]
    uy = pred[1]
    p  = pred[2]
    
    # ------------- 4. Save Individual Plots -------------
    def save_plot(data, title, colormap, filename):
        plt.figure(figsize=(10, 5))
        # DeepCFD shapes are usually (79, 172), we plot them directly
        im = plt.imshow(data, cmap=colormap, aspect='auto')
        plt.title(title, fontsize=16)
        plt.colorbar(im, orientation='horizontal', pad=0.15)
        plt.gca().invert_yaxis() # Match physical coordinates
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    print("Saving predicted fields...")
    # Using 'jet' as it is standard in CFD visualization matching the paper
    save_plot(ux, "Predicted Ux Velocity [m/s]", "jet", "pred_Ux.png")
    save_plot(uy, "Predicted Uy Velocity [m/s]", "jet", "pred_Uy.png")
    save_plot(p,  "Predicted Pressure [Pa]", "jet", "pred_P.png")
    
    print("All predictions successfully saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict and Save DeepCFD Fields")
    parser.add_argument("--model", type=str, default="Run/modelWeights.pth", help="Path to trained .pth model")
    parser.add_argument("--dataX", type=str, default="custom_dataX.pkl", help="Path to input dataX.pkl")
    args = parser.parse_args()
    
    predict_and_save(args.model, args.dataX)
