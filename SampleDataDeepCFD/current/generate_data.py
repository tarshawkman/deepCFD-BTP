import numpy as np
import os
import pickle
import subprocess
import shutil
import argparse
from matplotlib.path import Path
from scipy.spatial import cKDTree

def compute_influence_matrices(XC, YC, XB, YB, phi, S):
    N = len(XC)
    I, J, K, L = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
    for j in range(N):
        dx, dy = XC - XB[j], YC - YB[j]
        A = -dx * np.cos(phi[j]) - dy * np.sin(phi[j])
        B = dx**2 + dy**2
        Cn, Dn = np.sin(phi - phi[j]), -dx * np.sin(phi) + dy * np.cos(phi)
        Ct, Dt = -np.cos(phi - phi[j]), dx * np.cos(phi) + dy * np.sin(phi)
        Cnv, Dnv = -np.cos(phi - phi[j]), dx * np.cos(phi) + dy * np.sin(phi)
        Ctv, Dtv = np.sin(phi[j] - phi), dx * np.sin(phi) - dy * np.cos(phi)
        val = B - A**2
        E = np.sqrt(np.maximum(val, 0))
        mask = (B > 0)
        term1 = np.zeros_like(B)
        term1[mask] = 0.5 * np.log((S[j]**2 + 2*A[mask]*S[j] + B[mask]) / B[mask])
        mask_e = (E > 0)
        term2 = np.zeros_like(E)
        term2[mask_e] = (np.arctan2(S[j]+A[mask_e], E[mask_e]) - np.arctan2(A[mask_e], E[mask_e])) / E[mask_e]
        I[:, j] = Cn * term1 + (Dn - A*Cn) * term2
        J[:, j] = Ct * term1 + (Dt - A*Ct) * term2
        K[:, j] = Cnv * term1 + (Dnv - A*Cnv) * term2
        L[:, j] = Ctv * term1 + (Dtv - A*Ctv) * term2
        I[j, j], L[j, j] = np.pi, np.pi
    return I, J, K, L

def compute_velocities_vectorized(XP, YP, XB, YB, phi, S, lam, gamma, Vinf):
    Ny, Nx = XP.shape
    XPf, YPf = XP.flatten(), YP.flatten()
    mVx, mVy = np.zeros_like(XPf), np.zeros_like(XPf)
    for j in range(len(S)):
        dx, dy = XPf - XB[j], YPf - YB[j]
        A, B = -dx * np.cos(phi[j]) - dy * np.sin(phi[j]), dx**2 + dy**2
        Cx, Cy = -np.cos(phi[j]), -np.sin(phi[j])
        Nxv, Nyv = np.sin(phi[j]), -np.cos(phi[j])
        val = B - A**2
        E = np.sqrt(np.maximum(val, 0))
        mask = (B > 0)
        term1 = np.zeros_like(B)
        term1[mask] = 0.5 * np.log((S[j]**2 + 2*A[mask]*S[j] + B[mask]) / B[mask])
        mask_e = (E > 0)
        term2 = np.zeros_like(E)
        term2[mask_e] = (np.arctan2(S[j]+A[mask_e], E[mask_e]) - np.arctan2(A[mask_e], E[mask_e])) / E[mask_e]
        Mx = Cx * term1 + (XPf - XB[j] - A*Cx) * term2
        My = Cy * term1 + (YPf - YB[j] - A*Cy) * term2
        Nx_ = Nxv * term1 + (-(YPf - YB[j]) - A*Nxv) * term2
        Ny_ = Nyv * term1 + (XPf - XB[j] - A*Nyv) * term2
        mVx += (lam[j] * Mx - gamma * Nx_) / (2 * np.pi)
        mVy += (lam[j] * My - gamma * Ny_) / (2 * np.pi)
    return (Vinf + mVx).reshape(Ny, Nx), mVy.reshape(Ny, Nx)

def create_dataset_for_airfoil(dat_file_path, output_dir, xfoil_exe='xfoil.exe', nGridY=172, nGridX=79, Vinf=0.1):
    os.makedirs(output_dir, exist_ok=True)
    fbase = os.path.basename(dat_file_path.replace("\\", "/"))
    shutil.copy(dat_file_path, "tmp.dat")
    with open('xf.inp', 'w') as f:
        f.write("LOAD tmp.dat\nPPAR\nN 170\n\n\nPSAV af.txt\nQUIT\n")
    subprocess.run([xfoil_exe], stdin=open('xf.inp','r'), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        c = np.loadtxt("af.txt")
        XB, YB = c[:,0], c[:,1]
    except:
        print("XFOIL Failed"); return
    finally:
        for f in ['xf.inp', 'tmp.dat', 'af.txt']:
            if os.path.exists(f): os.remove(f)
            
    numPan = len(XB) - 1
    if np.sum([(XB[i+1]-XB[i])*(YB[i+1]+YB[i]) for i in range(numPan)]) < 0:
        XB, YB = np.flip(XB), np.flip(YB)
        
    XC, YC, S, phi = np.zeros(numPan), np.zeros(numPan), np.zeros(numPan), np.zeros(numPan)
    for i in range(numPan):
        XC[i], YC[i] = 0.5*(XB[i]+XB[i+1]), 0.5*(YB[i]+YB[i+1])
        dx, dy = XB[i+1]-XB[i], YB[i+1]-YB[i]
        S[i], phi[i] = np.sqrt(dx**2 + dy**2), np.arctan2(dy, dx)
        
    I, J, K, L = compute_influence_matrices(XC, YC, XB, YB, phi, S)
    A = np.zeros((numPan+1, numPan+1))
    A[:numPan, :numPan], A[:numPan, numPan] = I, -np.sum(K, axis=1)
    A[numPan, :numPan] = J[0,:] + J[numPan-1,:]
    A[numPan, numPan] = -np.sum(L[0,:] + L[numPan-1,:]) + 2*np.pi
    b = np.zeros(numPan+1)
    b[:numPan] = -Vinf * 2 * np.pi * np.cos(phi + np.pi/2)
    b[numPan] = -Vinf * 2 * np.pi * (np.sin(phi[0]+np.pi/2) + np.sin(phi[-1]+np.pi/2))
    res = np.linalg.solve(A, b)
    lam, gamma = res[:-1], res[-1]

    # Rotate 90 CCW: (x,y) -> (-y, x). Foil vertical on Y. Flow on Y.
    XB_r, YB_r = -YB, XB 
    yV = [np.min(YB_r)-0.5, np.max(YB_r)+0.5]
    xV = [np.min(XB_r)-0.3, np.max(XB_r)+0.3]
    XX, YY = np.meshgrid(np.linspace(xV[0], xV[1], nGridX), np.linspace(yV[0], yV[1], nGridY))
    
    print("Computing flow field...")
    VxH, VyH = compute_velocities_vectorized(YY, -XX, XB, YB, phi, S, lam, gamma, Vinf)
    Uy, Ux = VxH, -VyH
    
    path = Path(np.column_stack((XB_r, YB_r)))
    mask = path.contains_points(np.column_stack((XX.flatten(), YY.flatten()))).reshape(nGridY, nGridX)
    
    # Pressure P = Vinf^2 - V^2 (scaled to match toy data range ~0.05)
    # Scaling factor might be needed if Vinf=0.1. (0.01 range). Toy data 0.05.
    P = (Vinf**2 - (Ux**2 + Uy**2)) * 3.0 # Approximate scaling factor
    
    Uy[mask] = Ux[mask] = P[mask] = 0
    sdf1 = cKDTree(np.column_stack((XB_r, YB_r))).query(np.column_stack((XX.flatten(), YY.flatten())))[0].reshape(nGridY, nGridX)
    sdf1[mask] *= -1
    sdf2 = np.minimum(XX - xV[0], xV[1] - XX) * 20.0 
    
    dataX = np.expand_dims(np.stack([sdf1, (~mask).astype(np.float32), sdf2], axis=0), 0)
    dataY = np.expand_dims(np.stack([Uy, Ux, P], axis=0), 0)
    
    with open(os.path.join(output_dir, "dataX.pkl"), 'wb') as f: pickle.dump(dataX.astype(np.float32), f)
    with open(os.path.join(output_dir, "dataY.pkl"), 'wb') as f: pickle.dump(dataY.astype(np.float32), f)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dat_file", required=True); parser.add_argument("--out_dir", default="."); parser.add_argument("--xfoil_exe", default="xfoil.exe")
    args = parser.parse_args()
    create_dataset_for_airfoil(args.dat_file, args.out_dir, args.xfoil_exe)
