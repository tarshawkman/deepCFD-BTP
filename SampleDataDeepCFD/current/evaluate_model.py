import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from scipy.signal import savgol_filter
import argparse
import torch # Assuming he's using PyTorch for DeepCFD

def extract_cp(P, mask, Vinf, p_ref):
    """ Helper to extract Cp curve from an NxY pressure field and mask """
    obstacle = mask > 0.5
    fluid_boundary = binary_dilation(obstacle) & (~obstacle)
    y_idx, x_idx = np.where(fluid_boundary)
    
    chord_min, chord_max = np.min(y_idx), np.max(y_idx)
    x_c = (y_idx - chord_min) / (max(1, chord_max - chord_min))
    Cp_vals = (P[y_idx, x_idx] - p_ref) / (0.5 * Vinf**2)
    
    x_center = np.mean(x_idx) if len(x_idx) > 0 else 0
    left_mask, right_mask = x_idx < x_center, x_idx > x_center
    
    def smooth_and_sort(x, cp):
        if len(x) == 0: return x, cp
        s = np.argsort(x)
        x_s, cp_s = x[s], cp[s]
        w = min(11, len(cp_s) if len(cp_s) % 2 != 0 else len(cp_s)-1)
        return x_s, savgol_filter(cp_s, w, 3) if w > 3 else cp_s

    xl, cpl = smooth_and_sort(x_c[left_mask], Cp_vals[left_mask])
    xr, cpr = smooth_and_sort(x_c[right_mask], Cp_vals[right_mask])
    return xl, cpl, xr, cpr

def evaluate_predictions(dataX_path, dataY_path, model_weights_path, num_samples=5):
    # 1. Load Data
    with open(dataX_path, 'rb') as f: dataX = pickle.load(f)
    with open(dataY_path, 'rb') as f: dataY = pickle.load(f)
    
    # Select random samples
    indices = np.random.choice(len(dataX), min(num_samples, len(dataX)), replace=False)
    
    # 2. Load Model & Predict
    # NOTE: Adjust the Model class import below based on your actual Colab code!
    # from model import UNetEx  # Example DeepCFD model import
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = UNetEx(3, 3).to(device)
    # model.load_state_dict(torch.load(model_weights_path, map_location=device))
    # model.eval()
    
    # X_tensor = torch.FloatTensor(dataX[indices]).to(device)
    # with torch.no_grad():
    #     Y_pred = model(X_tensor).cpu().numpy()
    
    # --- For demonstration purposes if you just want to test this script without logic ---
    # We will simulate Y_pred by adding tiny noise to Ground Truth
    print("WARNING: Model inference logic is commented out because network architecture is unknown.")
    print("Using noisy Ground Truth as simulated predictions to demonstrate plots.")
    Y_pred = dataY[indices] + np.random.normal(0, 0.005, dataY[indices].shape)
    
    X_test = dataX[indices]
    Y_test = dataY[indices]
    
    Vinf = 0.1 # Constant from our generation script
    
    for i, idx in enumerate(indices):
        mask = X_test[i, 1]
        
        # Extract Vertical Velocity (Ux)
        ux_gt = Y_test[i, 0] 
        ux_pred = Y_pred[i, 0]
        ux_err = np.abs(ux_gt - ux_pred)
        
        # Extract Pressure (P)
        p_gt = Y_test[i, 2]
        p_pred = Y_pred[i, 2]
        p_err = np.abs((p_gt - np.mean(p_gt[0, :])) - (p_pred - np.mean(p_pred[0, :])))
        pref_gt = np.mean(p_gt[0, :])
        pref_pred = np.mean(p_pred[0, :])
        
        # --- Plot 1: Visual Heatmap Comparisons (Ux & Pressure) ---
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Velocity
        im0 = axs[0, 0].imshow(ux_gt, origin='lower', cmap='jet')
        axs[0, 0].set_title(f'Ground Truth (Ux)')
        fig.colorbar(im0, ax=axs[0, 0])
        
        im1 = axs[0, 1].imshow(ux_pred, origin='lower', cmap='jet')
        axs[0, 1].set_title(f'Model Prediction (Ux)')
        fig.colorbar(im1, ax=axs[0, 1])
        
        im2 = axs[0, 2].imshow(ux_err, origin='lower', cmap='hot')
        axs[0, 2].set_title(f'Velocity Absolute Error')
        fig.colorbar(im2, ax=axs[0, 2])
        
        # Row 2: Pressure
        im3 = axs[1, 0].imshow(p_gt, origin='lower', cmap='jet')
        axs[1, 0].set_title(f'Ground Truth (Pressure)')
        fig.colorbar(im3, ax=axs[1, 0])
        
        im4 = axs[1, 1].imshow(p_pred, origin='lower', cmap='jet')
        axs[1, 1].set_title(f'Model Prediction (Pressure)')
        fig.colorbar(im4, ax=axs[1, 1])
        
        im5 = axs[1, 2].imshow(p_err, origin='lower', cmap='hot')
        axs[1, 2].set_title(f'Pressure Absolute Error')
        fig.colorbar(im5, ax=axs[1, 2])
        
        plt.suptitle(f'DeepCFD Performance - Sample {idx}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'heatmap_compare_{idx}.png', dpi=150)
        plt.close()
        
        # --- Plot 2: Cp Curve Point Comparisons ---
        xl_gt, cpl_gt, xr_gt, cpr_gt = extract_cp(p_gt, mask, Vinf, pref_gt)
        xl_pr, cpl_pr, xr_pr, cpr_pr = extract_cp(p_pred, mask, Vinf, pref_pred)
        
        plt.figure(figsize=(10, 6))
        
        # Ground Truth
        plt.plot(xl_gt, cpl_gt, 'b-', linewidth=3, label='GT Left (Upper) Surf')
        plt.plot(xr_gt, cpr_gt, 'b--', linewidth=3, label='GT Right (Lower) Surf')
        
        # Prediction
        plt.plot(xl_pr, cpl_pr, 'r-', linewidth=1.5, label='Pred Left (Upper) Surf')
        plt.plot(xr_pr, cpr_pr, 'r--', linewidth=1.5, label='Pred Right (Lower) Surf')
        
        plt.xlabel('x / c (Normalized Chord)')
        plt.ylabel('Cp (Pressure Coefficient)')
        plt.title(f'Cp Coefficient Curve Match - Sample {idx}')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'cp_compare_{idx}.png', dpi=150)
        plt.close()

    print(f"Evaluation complete. Saved qualitative & quantitative plot sets for {num_samples} unseen testing airfoils.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataX", default="dataX_test.pkl")
    parser.add_argument("--dataY", default="dataY_test.pkl")
    parser.add_argument("--weights", default="model.pt")
    args = parser.parse_args()
    evaluate_predictions(args.dataX, args.dataY, args.weights)
