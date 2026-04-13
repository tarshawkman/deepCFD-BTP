import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from scipy.signal import savgol_filter
import argparse

def plot_cp(dataX_file, dataY_file):
    print(f"Loading {dataX_file} and {dataY_file}...")
    with open(dataX_file, 'rb') as f:
        dataX = pickle.load(f)[0] # shape (3, 172, 79)
    with open(dataY_file, 'rb') as f:
        dataY = pickle.load(f)[0] # shape (3, 172, 79)
        
    mask = dataX[1]
    inlet = dataX[2]
    P = dataY[2]
    
    # Extract Vinf and far-field reference pressure
    Vinf = np.max(inlet)
    p_ref = np.mean(P[0, :]) # Mean pressure at inlet boundary, far away from obstacle
    
    # Find fluid cells exactly adjacent to the airfoil
    obstacle = mask > 0.5
    fluid_boundary = binary_dilation(obstacle) & (~obstacle)
    
    y_idx, x_idx = np.where(fluid_boundary)
    
    # The airfoil is oriented vertically, so Y-axis represents the chord.
    # Flow comes from bottom (Y=0), so min(Y) is the Leading Edge (x/c = 0)
    # and max(Y) is the Trailing Edge (x/c = 1)
    chord_min, chord_max = np.min(y_idx), np.max(y_idx)
    x_c = (y_idx - chord_min) / (chord_max - chord_min) 
    
    # Calculate Cp from the CFD/LBM pressure field data
    # Standard formula: Cp = (p - p_ref) / (0.5 * rho * Vinf^2)
    # LBM pressure is kinematic (p/rho), so we assume rho=1.0
    Cp_vals = (P[y_idx, x_idx] - p_ref) / (0.5 * Vinf**2)
    
    # Separate into left (upper) and right (lower) surfaces
    x_center = np.mean(x_idx)
    left_mask = x_idx < x_center
    right_mask = x_idx > x_center
    
    x_c_left = x_c[left_mask]
    cp_left = Cp_vals[left_mask]
    
    x_c_right = x_c[right_mask]
    cp_right = Cp_vals[right_mask]
    
    # Sort values from Leading Edge (0) to Trailing Edge (1)
    sort_l = np.argsort(x_c_left)
    x_c_left, cp_left = x_c_left[sort_l], cp_left[sort_l]
    
    sort_r = np.argsort(x_c_right)
    x_c_right, cp_right = x_c_right[sort_r], cp_right[sort_r]
    
    # Apply a gentle Savitzky-Golay filter to smooth out the staircase noise
    # inherently present when sampling a jagged Cartesian grid mask
    def smooth(y):
        window = min(11, len(y) if len(y) % 2 != 0 else len(y)-1)
        if window > 3:
            return savgol_filter(y, window, 3)
        return y

    cp_left_smooth = smooth(cp_left)
    cp_right_smooth = smooth(cp_right)
    
    # -----------------------------
    # PLOTTING
    # -----------------------------
    plt.figure(figsize=(10, 6))
    
    # Plot smooth lines
    plt.plot(x_c_left, cp_left_smooth, 'b-', linewidth=2.5, label='Left Surface')
    plt.plot(x_c_right, cp_right_smooth, 'r--', linewidth=2.5, label='Right Surface')
    
    # Plot raw CFD sampled dots faintly in the background to show real grid data
    plt.plot(x_c_left, cp_left, 'bo', alpha=0.3, markersize=3)
    plt.plot(x_c_right, cp_right, 'ro', alpha=0.3, markersize=3)
    
    plt.xlabel('x / c (Normalized Chord)')
    plt.ylabel('Cp (Pressure Coefficient)')
    plt.title('Extracted Cp Distribution from DeepCFD Generation Pipeline')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    # In aerodynamics, Cp is conventionally plotted inverted (negative up)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plot_path = 'cp_plot.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Extraction Complete! Plot saved successfully to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataX", default="dataX.pkl")
    parser.add_argument("--dataY", default="dataY.pkl")
    args = parser.parse_args()
    plot_cp(args.dataX, args.dataY)
