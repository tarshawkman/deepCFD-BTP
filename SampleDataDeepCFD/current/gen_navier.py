import numpy as np
import pickle
import argparse
from matplotlib.path import Path
from scipy.ndimage import distance_transform_edt
import subprocess
import os
import shutil

def run_xfoil(dat_file_path, xfoil_exe):
    if os.path.exists("t.dat"): os.remove("t.dat")
    shutil.copy(dat_file_path, "t.dat")
    if os.path.exists("af.txt"): os.remove("af.txt")
    
    # PPAR N 170 re-panels to exactly 170 evenly-distributed panels — same as generate_data.py
    with open("xfoil_input.txt", "w") as f:
        f.write("LOAD t.dat\nPPAR\nN 170\n\n\nPSAV af.txt\nQUIT\n")
        
    subprocess.run(f'"{xfoil_exe}" < xfoil_input.txt', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    coords = np.loadtxt("af.txt", skiprows=1)
    return coords[:,0], coords[:,1]

def solve_lbm(mask, V_inlet=0.1, max_iters=5000):
    Ny, Nx = mask.shape
    tau = 0.6
    omega = 1.0 / tau
    obstacle = mask > 0.5

    cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    u = np.zeros((2, Ny, Nx))
    u[0, :, :] = V_inlet
    u[:, obstacle] = 0.0
    rho = np.ones((Ny, Nx))

    def get_equilibrium(rho_f, u_f):
        feq = np.zeros((9, rho_f.shape[0], rho_f.shape[1]))
        usqr = 1.5 * (u_f[0]**2 + u_f[1]**2)
        for i in range(9):
            cu = 3.0 * (cy[i] * u_f[0] + cx[i] * u_f[1])
            feq[i] = weights[i] * rho_f * (1.0 + cu + 0.5 * cu**2 - usqr)
        return feq

    fin = get_equilibrium(rho, u)
    print("Executing LBM Navier-Stokes solver...")
    
    for iter in range(max_iters):
        # Outlet zero-gradient
        for i in [4, 7, 8]:
            fin[i, -1, :] = fin[i, -2, :]
            
        # Inlet Fixed velocity
        u_inlet = np.zeros((2, Nx))
        u_inlet[0, :] = V_inlet
        feq_inlet = get_equilibrium(rho[1, :].reshape(1, Nx), u_inlet.reshape((2, 1, Nx)))
        for i in [2, 5, 6]:
            fin[i, 0, :] = feq_inlet[i, 0, :]

        # Stream
        fout = fin.copy()
        for i in range(9):
            fin[i] = np.roll(fin[i], cx[i], axis=1)
            fin[i] = np.roll(fin[i], cy[i], axis=0)

        # Left / Right boundaries
        for i in [1, 5, 8]:
            fin[i, :, 0] = fout[opposite[i], :, 0]
        for i in [3, 6, 7]:
            fin[i, :, -1] = fout[opposite[i], :, -1]

        # Obstacle bounce back
        for i in range(9):
            fin[i, obstacle] = fout[opposite[i], obstacle]

        # Macroscopic
        rho = np.sum(fin, axis=0)
        u[0] = np.sum(fin * cy[:, np.newaxis, np.newaxis], axis=0) / rho
        u[1] = np.sum(fin * cx[:, np.newaxis, np.newaxis], axis=0) / rho

        # Collision
        feq = get_equilibrium(rho, u)
        fin = fin - omega * (fin - feq)

        if iter % 1000 == 0:
            print(f"  timestep {iter}/{max_iters}")

    p = (rho - 1.0) / 3.0
    return u, p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dat_file", required=True)
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--xfoil_exe", required=True)
    args = parser.parse_args()

    Ny, Nx = 172, 79
    XB, YB = run_xfoil(args.dat_file, args.xfoil_exe)
    
    # Work in the SAME physical coordinate space as generate_data.py
    # Rotate 90° CCW: chord lies along Y axis (vertical flow)
    XB_r = -YB   # physical coords, chord ≈ 1.0 unit
    YB_r = XB

    # Build a tight channel around the airfoil — same margins as generate_data.py
    # X: airfoil width ± 0.3 chord,  Y: airfoil chord ± 0.5 chord
    x_lo, x_hi = np.min(XB_r) - 0.3, np.max(XB_r) + 0.3
    y_lo, y_hi = np.min(YB_r) - 0.5, np.max(YB_r) + 0.5

    # Create the 79×172 grid in physical units (linspace = same as generate_data.py)
    xx = np.linspace(x_lo, x_hi, Nx)
    yy = np.linspace(y_lo, y_hi, Ny)
    XX, YY = np.meshgrid(xx, yy)   # YY varies along rows (flow direction)

    # Rasterize in physical coordinates — high effective resolution
    path = Path(np.column_stack((XB_r, YB_r)))
    mask = path.contains_points(np.column_stack((XX.flatten(), YY.flatten()))).reshape((Ny, Nx)).astype(float)
    
    Vinf = 0.1
    u, p = solve_lbm(mask, Vinf, max_iters=6000)
    
    Uy, Ux = u[0], u[1]
    Uy[mask > 0.5] = 0
    Ux[mask > 0.5] = 0
    p[mask > 0.5] = 0
    
    dataY = np.expand_dims(np.stack([Uy, Ux, p]), 0)
    sdf1 = distance_transform_edt(1 - mask)
    inlet = np.full((Ny,Nx), Vinf)
    dataX = np.expand_dims(np.stack([sdf1, mask, inlet]), 0)
    
    with open(os.path.join(args.out_dir, "dataX.pkl"), "wb") as f: pickle.dump(dataX, f)
    with open(os.path.join(args.out_dir, "dataY.pkl"), "wb") as f: pickle.dump(dataY, f)
    print("Navier-Stokes (LBM) Data Generation Successful!")

if __name__ == "__main__":
    main()
