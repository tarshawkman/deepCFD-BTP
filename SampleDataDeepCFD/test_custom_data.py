import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from Models.UNetEx import UNetEx

def test_model(model_path, custom_x_path, custom_y_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Needs the exact same filters/kernel_size as trained in DeepCFD.py
    kernel_size = 5
    filters = [8, 16, 32, 32]
    bn = False
    wn = False
    
    model = UNetEx(3, 3, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("Loading custom data...")
    with open(custom_x_path, "rb") as f:
        dataX = pickle.load(f)
    with open(custom_y_path, "rb") as f:
        dataY = pickle.load(f)
        
    x_tensor = torch.FloatTensor(dataX).to(device)
    y_tensor = torch.FloatTensor(dataY).to(device)
    
    with torch.no_grad():
        out = model(x_tensor)
        
    out_np = out.cpu().numpy()[0]
    y_np = y_tensor.cpu().numpy()[0]
    x_np = x_tensor.cpu().numpy()[0]
    
    # Plotting channel 0 (Ux)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(x_np[0], cmap='gray')
    plt.title("Airfoil Geometry")
    
    plt.subplot(1, 3, 2)
    plt.imshow(y_np[0], cmap='jet')
    plt.title("Ground Truth (spvp_airfoil) Ux")
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(out_np[0], cmap='jet')
    plt.title("DeepCFD Predicted Ux")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('custom_test_Ux.png')
    print("Saved comparison image to custom_test_Ux.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="c:/Users/Aaditya Saraf/Desktop/btp/SampleDataDeepCFD/Run/modelWeights.pth")
    parser.add_argument("--dataX", type=str, default="c:/Users/Aaditya Saraf/Desktop/btp/PanelMethodCode/test_out/custom_dataX.pkl")
    parser.add_argument("--dataY", type=str, default="c:/Users/Aaditya Saraf/Desktop/btp/PanelMethodCode/test_out/custom_dataY.pkl")
    args = parser.parse_args()
    
    test_model(args.model, args.dataX, args.dataY)
