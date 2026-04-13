import numpy as np
import pickle
import os
import glob
from matplotlib.path import Path
from scipy.ndimage import distance_transform_edt
from gen_navier import solve_lbm, run_xfoil
import traceback

def batch_generate(dat_folder, out_folder, xfoil_exe, limit=None):
    # Find all dat files (remembering the nested folder structure)
    dat_files = glob.glob(os.path.join(dat_folder, "**", "*.dat"), recursive=True)
    if limit: dat_files = dat_files[:limit]
    
    all_dataX, all_dataY = [], []
    success_count = 0
    Ny, Nx = 172, 79
    Vinf = 0.1

    print(f"Found {len(dat_files)} airfoils. Starting generation...")
    
    for idx, dat_file in enumerate(dat_files):
        try:
            print(f"[{idx+1}/{len(dat_files)}] Processing {os.path.basename(dat_file)}...")
            XB, YB = run_xfoil(dat_file, xfoil_exe)
            
            # Rotate, Scale, Center
            XB_r, YB_r = -YB, XB
            chord = np.max(YB_r) - np.min(YB_r)
            if chord <= 0: continue
            
            sc = 40.0 / chord
            XB_r *= sc; YB_r *= sc
            XB_r += 39.0 - np.mean(XB_r)
            YB_r += 45.0 - np.min(YB_r)
            
            Xgrid, Ygrid = np.meshgrid(np.arange(Nx), np.arange(Ny))
            path = Path(np.column_stack((XB_r, YB_r)))
            mask = path.contains_points(np.column_stack((Xgrid.flatten(), Ygrid.flatten()))).reshape((Ny, Nx)).astype(float)
            
            # Solve Navier-Stokes
            u, p = solve_lbm(mask, Vinf, max_iters=5000)
            
            Uy, Ux = u[0], u[1]
            Uy[mask > 0.5] = 0; Ux[mask > 0.5] = 0; p[mask > 0.5] = 0
            
            all_dataY.append(np.stack([Uy, Ux, p]))
            
            sdf1 = distance_transform_edt(1 - mask)
            inlet = np.full((Ny,Nx), Vinf)
            all_dataX.append(np.stack([sdf1, mask, inlet]))
            
            success_count += 1
            
        except Exception as e:
            print(f"  -> Failed to process {os.path.basename(dat_file)}")
            
    print(f"\nSuccessfully generated {success_count} samples.")
    
    final_dataX = np.stack(all_dataX, axis=0) # Shape: (N, 3, 172, 79)
    final_dataY = np.stack(all_dataY, axis=0) # Shape: (N, 3, 172, 79)
    
    with open(os.path.join(out_folder, "dataX_full.pkl"), "wb") as f:
        pickle.dump(final_dataX, f)
    with open(os.path.join(out_folder, "dataY_full.pkl"), "wb") as f:
        pickle.dump(final_dataY, f)
    print("Saved dataX_full.pkl and dataY_full.pkl!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dat_dir", required=True)
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--xfoil", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of airfoils to process for testing")
    args = parser.parse_args()
    batch_generate(args.dat_dir, args.out_dir, args.xfoil, args.limit)
