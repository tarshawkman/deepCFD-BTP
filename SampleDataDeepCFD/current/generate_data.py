import numpy as np
import os
import pickle
import subprocess
import shutil
import argparse
from matplotlib.path import Path

def COMPUTE_IJ_SPM(XC, YC, XB, YB, phi, S):
    numPan = len(XC)
    I = np.zeros((numPan, numPan))
    J = np.zeros((numPan, numPan))
    
    for i in range(numPan):
        for j in range(numPan):
            if j != i:
                A = -(XC[i]-XB[j])*np.cos(phi[j]) - (YC[i]-YB[j])*np.sin(phi[j])
                B = (XC[i]-XB[j])**2 + (YC[i]-YB[j])**2
                Cn = np.sin(phi[i]-phi[j])
                Dn = -(XC[i]-XB[j])*np.sin(phi[i]) + (YC[i]-YB[j])*np.cos(phi[i])
                Ct = -np.cos(phi[i]-phi[j])
                Dt = (XC[i]-XB[j])*np.cos(phi[i]) + (YC[i]-YB[j])*np.sin(phi[i])
                
                val = B - A**2
                E = np.sqrt(val) if val >= 0 else 0.0
                
                if B > 0:
                    term1_I = 0.5*Cn*np.log((S[j]**2 + 2*A*S[j] + B)/B)
                    term1_J = 0.5*Ct*np.log((S[j]**2 + 2*A*S[j] + B)/B)
                else:
                    term1_I = 0.0
                    term1_J = 0.0
                    
                if E > 0:
                    term2_I = ((Dn-A*Cn)/E) * (np.arctan2((S[j]+A), E) - np.arctan2(A, E))
                    term2_J = ((Dt-A*Ct)/E) * (np.arctan2((S[j]+A), E) - np.arctan2(A, E))
                else:
                    term2_I = 0.0
                    term2_J = 0.0
                    
                I[i,j] = term1_I + term2_I
                J[i,j] = term1_J + term2_J
                
                if np.isnan(I[i,j]) or np.isinf(I[i,j]):
                    I[i,j] = 0.0
                if np.isnan(J[i,j]) or np.isinf(J[i,j]):
                    J[i,j] = 0.0
    return I, J

def COMPUTE_KL_VPM(XC, YC, XB, YB, phi, S):
    numPan = len(XC)
    K = np.zeros((numPan, numPan))
    L = np.zeros((numPan, numPan))
    
    for i in range(numPan):
        for j in range(numPan):
            if j != i:
                A = -(XC[i]-XB[j])*np.cos(phi[j]) - (YC[i]-YB[j])*np.sin(phi[j])
                B = (XC[i]-XB[j])**2 + (YC[i]-YB[j])**2
                Cn = -np.cos(phi[i]-phi[j])
                Dn = (XC[i]-XB[j])*np.cos(phi[i]) + (YC[i]-YB[j])*np.sin(phi[i])
                Ct = np.sin(phi[j]-phi[i])
                Dt = (XC[i]-XB[j])*np.sin(phi[i]) - (YC[i]-YB[j])*np.cos(phi[i])
                
                val = B - A**2
                E = np.sqrt(val) if val >= 0 else 0.0
                
                if B > 0:
                    term1_K = 0.5*Cn*np.log((S[j]**2 + 2*A*S[j] + B)/B)
                    term1_L = 0.5*Ct*np.log((S[j]**2 + 2*A*S[j] + B)/B)
                else:
                    term1_K = 0.0
                    term1_L = 0.0
                    
                if E > 0:
                    term2_K = ((Dn-A*Cn)/E) * (np.arctan2((S[j]+A), E) - np.arctan2(A, E))
                    term2_L = ((Dt-A*Ct)/E) * (np.arctan2((S[j]+A), E) - np.arctan2(A, E))
                else:
                    term2_K = 0.0
                    term2_L = 0.0
                    
                K[i,j] = term1_K + term2_K
                L[i,j] = term1_L + term2_L
                
                if np.isnan(K[i,j]) or np.isinf(K[i,j]):
                    K[i,j] = 0.0
                if np.isnan(L[i,j]) or np.isinf(L[i,j]):
                    L[i,j] = 0.0
    return K, L

def STREAMLINE_SPM(XP, YP, XB, YB, phi, S):
    numPan = len(XB) - 1
    Mx = np.zeros(numPan)
    My = np.zeros(numPan)
    
    for j in range(numPan):
        A = -(XP-XB[j])*np.cos(phi[j]) - (YP-YB[j])*np.sin(phi[j])
        B = (XP-XB[j])**2 + (YP-YB[j])**2
        Cx = -np.cos(phi[j])
        Dx = XP - XB[j]
        Cy = -np.sin(phi[j])
        Dy = YP - YB[j]
        
        val = B - A**2
        E = np.sqrt(val) if val >= 0 else 0.0
        
        if B > 0:
            term1_Mx = 0.5*Cx*np.log((S[j]**2 + 2*A*S[j] + B)/B)
            term1_My = 0.5*Cy*np.log((S[j]**2 + 2*A*S[j] + B)/B)
        else:
            term1_Mx = 0.0
            term1_My = 0.0
            
        if E > 0:
            term2_Mx = ((Dx - A*Cx)/E) * (np.arctan2((S[j]+A), E) - np.arctan2(A, E))
            term2_My = ((Dy - A*Cy)/E) * (np.arctan2((S[j]+A), E) - np.arctan2(A, E))
        else:
            term2_Mx = 0.0
            term2_My = 0.0
            
        Mx[j] = term1_Mx + term2_Mx
        My[j] = term1_My + term2_My
        
        if np.isnan(Mx[j]) or np.isinf(Mx[j]): Mx[j] = 0.0
        if np.isnan(My[j]) or np.isinf(My[j]): My[j] = 0.0
    return Mx, My

def STREAMLINE_VPM(XP, YP, XB, YB, phi, S):
    numPan = len(XB) - 1
    Nx = np.zeros(numPan)
    Ny = np.zeros(numPan)
    
    for j in range(numPan):
        A = -(XP-XB[j])*np.cos(phi[j]) - (YP-YB[j])*np.sin(phi[j])
        B = (XP-XB[j])**2 + (YP-YB[j])**2
        Cx = np.sin(phi[j])
        Dx = -(YP-YB[j])
        Cy = -np.cos(phi[j])
        Dy = XP - XB[j]
        
        val = B - A**2
        E = np.sqrt(val) if val >= 0 else 0.0
        
        if B > 0:
            term1_Nx = 0.5*Cx*np.log((S[j]**2 + 2*A*S[j] + B)/B)
            term1_Ny = 0.5*Cy*np.log((S[j]**2 + 2*A*S[j] + B)/B)
        else:
            term1_Nx = 0.0
            term1_Ny = 0.0
            
        if E > 0:
            term2_Nx = ((Dx - A*Cx)/E) * (np.arctan2((S[j]+A), E) - np.arctan2(A, E))
            term2_Ny = ((Dy - A*Cy)/E) * (np.arctan2((S[j]+A), E) - np.arctan2(A, E))
        else:
            term2_Nx = 0.0
            term2_Ny = 0.0
            
        Nx[j] = term1_Nx + term2_Nx
        Ny[j] = term1_Ny + term2_Ny
        
        if np.isnan(Nx[j]) or np.isinf(Nx[j]): Nx[j] = 0.0
        if np.isnan(Ny[j]) or np.isinf(Ny[j]): Ny[j] = 0.0
    return Nx, Ny

def create_dataset_for_airfoil(dat_file_path, output_dir, xfoil_exe='xfoil.exe', nGridY=172, nGridX=79, Vinf=1.0, AoA=0.0):
    PPAR = {
        'N': '170',
        'P': '4',
        'T': '1',
        'R': '1',
        'XT': '1 1',
        'XB': '1 1'
    }
    
    os.makedirs(output_dir, exist_ok=True)
    flnm = os.path.basename(dat_file_path.replace("\\", "/"))
    airfoilName = flnm[:-4]
    
    # Needs to be a local dat file so xfoil can handle path lengths well
    local_dat = f"local_{flnm}"
    shutil.copy(dat_file_path, local_dat)
    
    saveFlnm = f"Save_{airfoilName}.txt"
    saveFlnmCp = f"Save_{airfoilName}_Cp.txt"
    saveFlnmPol = f"Save_{airfoilName}_Pol.txt"
    
    for f in [saveFlnm, saveFlnmCp, saveFlnmPol]:
        if os.path.exists(f):
            os.remove(f)
            
    with open('xfoil_input.inp', 'w') as fid:
        fid.write(f"LOAD {local_dat}\n")
        fid.write("PPAR\n")
        fid.write(f"N {PPAR['N']}\n")
        fid.write(f"P {PPAR['P']}\n")
        fid.write(f"T {PPAR['T']}\n")
        fid.write(f"R {PPAR['R']}\n")
        fid.write(f"XT {PPAR['XT']}\n")
        fid.write(f"XB {PPAR['XB']}\n")
        fid.write("\n\n") # Apply and back out to XFOIL menu
        fid.write(f"PSAV {saveFlnm}\n")
        fid.write("OPER\n")
        fid.write("Pacc 1 \n\n\n")
        fid.write(f"Alfa {AoA}\n")
        fid.write(f"CPWR {saveFlnmCp}\n")
        fid.write("PWRT\n")
        fid.write(f"{saveFlnmPol}\n")
        fid.write("\n")
        fid.write("QUIT\n")
        
    success = True
    with open('xfoil_input.inp', 'r') as fid_in:
        try:
            subprocess.run([xfoil_exe], stdin=fid_in, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Error executing XFOIL ({xfoil_exe}):", e)
            print("Make sure XFOIL is available in your PATH or provided via --xfoil_exe.")
            success = False

    # cleanup input file and local dat
    for f in ['xfoil_input.inp', local_dat]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception as e:
                pass
                
    if not success:
        return
            
    try:
        af_coords = np.loadtxt(saveFlnm)
        os.remove(saveFlnm)
        XB = af_coords[:, 0]
        YB = af_coords[:, 1]
    except Exception as e:
        print("Failed to read XFOIL output coordinates.", e)
        return

    # clean other xfoil outputs
    for f in [saveFlnmCp, saveFlnmPol]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except: pass

    if len(XB) == 0:
        print("Empty coordinates.")
        return

    numPts = len(XB)
    numPan = numPts - 1
    
    edge = np.zeros(numPan)
    for i in range(numPan):
        edge[i] = (XB[i+1]-XB[i])*(YB[i+1]+YB[i])
    sumEdge = np.sum(edge)
    
    if sumEdge < 0:
        XB = np.flip(XB)
        YB = np.flip(YB)
        
    XC = np.zeros(numPan)
    YC = np.zeros(numPan)
    S = np.zeros(numPan)
    phiD = np.zeros(numPan)
    
    for i in range(numPan):
        XC[i] = 0.5*(XB[i]+XB[i+1])
        YC[i] = 0.5*(YB[i]+YB[i+1])
        dx = XB[i+1]-XB[i]
        dy = YB[i+1]-YB[i]
        S[i] = np.sqrt(dx**2 + dy**2)
        phiD[i] = np.degrees(np.arctan2(dy, dx))
        if phiD[i] < 0:
            phiD[i] += 360
            
    deltaD = phiD + 90
    betaD = deltaD - AoA
    betaD[betaD > 360] -= 360
    
    phi = np.radians(phiD)
    beta = np.radians(betaD)
    
    # -------------------------------------------------------------
    # source and vortex panel methods
    # -------------------------------------------------------------
    I, J = COMPUTE_IJ_SPM(XC, YC, XB, YB, phi, S)
    K, L = COMPUTE_KL_VPM(XC, YC, XB, YB, phi, S)
    
    A = np.zeros((numPan+1, numPan+1))
    for i in range(numPan):
        for j in range(numPan):
            if i == j:
                A[i,j] = np.pi
            else:
                A[i,j] = I[i,j]
                
    for i in range(numPan):
        A[i, numPan] = -np.sum(K[i,:])
        
    for j in range(numPan):
        A[numPan, j] = J[0, j] + J[numPan-1, j]
        
    A[numPan, numPan] = -np.sum(L[0,:] + L[numPan-1,:]) + 2*np.pi
    
    b = np.zeros(numPan+1)
    for i in range(numPan):
        b[i] = -Vinf*2*np.pi*np.cos(beta[i])
    b[numPan] = -Vinf*2*np.pi*(np.sin(beta[0]) + np.sin(beta[numPan-1]))
    
    resArr = np.linalg.solve(A, b)
    
    lam = resArr[:-1]
    gamma = resArr[-1]
    
    print(f"Starting {nGridX}x{nGridY} Grid Evaluation (Vectorized)...")
    
    xVals = [np.min(XB)-0.5, np.max(XB)+0.5]
    yVals = [np.min(YB)-0.3, np.max(YB)+0.3]
    
    Xgrid = np.linspace(xVals[0], xVals[1], nGridX)
    Ygrid = np.linspace(yVals[0], yVals[1], nGridY)
    
    # Use xy indexing to physically match gen_data.py
    XX, YY = np.meshgrid(Xgrid, Ygrid) # XX is (79, 172)
    
    Vx = np.zeros((nGridY, nGridX))
    Vy = np.zeros((nGridY, nGridX))
    
    from scipy.spatial.distance import cdist
    airfoil_path = Path(np.column_stack((XB, YB)))
    
    pts = np.column_stack((XX.flatten(), YY.flatten()))
    bndy = np.column_stack((XB, YB))
    
    print("Computing SDF1, Flow Region, and SDF2...")
    dists = np.min(cdist(pts, bndy), axis=1)
    SDF1 = dists.reshape((nGridY, nGridX))
    
    inside_mask_flat = airfoil_path.contains_points(pts)
    inside_mask = inside_mask_flat.reshape((nGridY, nGridX))
    SDF1[inside_mask] = -SDF1[inside_mask]
    
    # Flow Region Channel on native frame (79, 172)
    # y ranges from 0 to 78, x ranges from 0 to 171
    FlowRegion = np.ones((nGridY, nGridX), dtype=np.float32)
    FlowRegion[0, :]  = 2  # Bottom Y wall
    FlowRegion[-1, :] = 2  # Top Y wall
    FlowRegion[:, 0]  = 3  # Inlet X (left side)
    FlowRegion[:, -1] = 4  # Outlet X (right side)
    FlowRegion[inside_mask] = 0
    
    # SDF 2: Distance to non-slip walls (Y boundaries)
    SDF2 = np.zeros((nGridY, nGridX))
    for m in range(nGridY):
        d_wall = min(abs(Ygrid[m] - Ygrid[0]), abs(Ygrid[-1] - Ygrid[m]))
        SDF2[m, :] = d_wall
        
    print("Evaluating velocity field... (Vectorized for maximum performance)")
    YY_offset = YY + 1e-8
    Vx_ind = np.zeros((nGridY, nGridX))
    Vy_ind = np.zeros((nGridY, nGridX))
    
    for j in range(numPan):
        A = -(XX-XB[j])*np.cos(phi[j]) - (YY_offset-YB[j])*np.sin(phi[j])
        B = (XX-XB[j])**2 + (YY_offset-YB[j])**2
        
        val = B - A**2
        E_mask = val > 0
        E = np.zeros_like(val)
        E[E_mask] = np.sqrt(val[E_mask])
        
        # SPM component
        Cx_spm = -np.cos(phi[j])
        Cy_spm = -np.sin(phi[j])
        Dx_spm = XX - XB[j]
        Dy_spm = YY_offset - YB[j]
        
        log_term = np.zeros_like(B)
        B_mask = B > 0
        log_term[B_mask] = np.log((S[j]**2 + 2*A[B_mask]*S[j] + B[B_mask]) / B[B_mask])
        
        atan_term = np.zeros_like(E)
        atan_term[E_mask] = np.arctan2(S[j]+A[E_mask], E[E_mask]) - np.arctan2(A[E_mask], E[E_mask])
        
        E_safe = np.where(E_mask, E, 1.0)
        
        Mx = 0.5 * Cx_spm * log_term
        My = 0.5 * Cy_spm * log_term
        Mx[E_mask] += ((Dx_spm[E_mask] - A[E_mask]*Cx_spm) / E_safe[E_mask]) * atan_term[E_mask]
        My[E_mask] += ((Dy_spm[E_mask] - A[E_mask]*Cy_spm) / E_safe[E_mask]) * atan_term[E_mask]
        
        Mx = np.nan_to_num(Mx, nan=0.0)
        My = np.nan_to_num(My, nan=0.0)
        
        # VPM component
        Cx_vpm = np.sin(phi[j])
        Cy_vpm = -np.cos(phi[j])
        Dx_vpm = -(YY_offset - YB[j])
        Dy_vpm = XX - XB[j]
        
        Nx = 0.5 * Cx_vpm * log_term
        Ny = 0.5 * Cy_vpm * log_term
        Nx[E_mask] += ((Dx_vpm[E_mask] - A[E_mask]*Cx_vpm) / E_safe[E_mask]) * atan_term[E_mask]
        Ny[E_mask] += ((Dy_vpm[E_mask] - A[E_mask]*Cy_vpm) / E_safe[E_mask]) * atan_term[E_mask]
        
        Nx = np.nan_to_num(Nx, nan=0.0)
        Ny = np.nan_to_num(Ny, nan=0.0)
        
        Vx_ind += lam[j] * Mx / (2*np.pi) - gamma * Nx / (2*np.pi)
        Vy_ind += lam[j] * My / (2*np.pi) - gamma * Ny / (2*np.pi)
    
    Vx = Vinf*np.cos(np.radians(AoA)) + Vx_ind
    Vy = Vinf*np.sin(np.radians(AoA)) + Vy_ind
    
    Vx[inside_mask] = 0.0
    Vy[inside_mask] = 0.0
    
    Vxy = np.sqrt(Vx**2 + Vy**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        CpXY = 1 - (Vxy/Vinf)**2
        
    p = np.nan_to_num(0.5 * CpXY, nan=0.0)
    Vx = np.nan_to_num(Vx, nan=0.0)
    Vy = np.nan_to_num(Vy, nan=0.0)
    
    # -----------------------------
    # APPLY TRANSPOSE ORIENTATION
    # -----------------------------
    # Extrapolates (79, 172) native array into exactly (172, 79)
    SDF1 = SDF1.T
    FlowRegion = FlowRegion.T
    SDF2 = SDF2.T
    Vx = Vx.T
    Vy = Vy.T
    p = p.T
    
    # Format properly
    dataX = np.stack([SDF1, FlowRegion, SDF2], axis=0).astype(np.float32)
    dataX = np.expand_dims(dataX, axis=0) # shape (1, 3, 172, 79)
    
    dataY = np.stack([Vx, Vy, p], axis=0).astype(np.float32)
    dataY = np.expand_dims(dataY, axis=0)
    
    out_X = os.path.join(output_dir, "dataX.pkl")
    out_Y = os.path.join(output_dir, "dataY.pkl")
    
    with open(out_X, 'wb') as f:
        pickle.dump(dataX, f)
        
    with open(out_Y, 'wb') as f:
        pickle.dump(dataY, f)
        
    print(f"Success. Wrote DataX to {out_X} and DataY to {out_Y}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create DeepCFD Data from Airfoil .dat")
    parser.add_argument("--dat_file", type=str, required=True, help="Path to airfoil .dat file")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory for .pkl files")
    parser.add_argument("--xfoil_exe", type=str, default="xfoil.exe", help="Path to the xfoil executable")
    parser.add_argument("--grid_x", type=int, default=172, help="Grid dimension X (Length of domain)")
    parser.add_argument("--grid_y", type=int, default=79, help="Grid dimension Y (Height of domain)")
    parser.add_argument("--vinf", type=float, default=1.0, help="Freestream velocity")
    parser.add_argument("--aoa", type=float, default=0.0, help="Angle of attack in degrees")
    args = parser.parse_args()
    
    create_dataset_for_airfoil(
        dat_file_path=args.dat_file,
        output_dir=args.out_dir,
        xfoil_exe=args.xfoil_exe,
        nGridX=args.grid_x,
        nGridY=args.grid_y,
        Vinf=args.vinf,
        AoA=args.aoa
    )
