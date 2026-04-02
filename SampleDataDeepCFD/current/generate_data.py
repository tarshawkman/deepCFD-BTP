import numpy as np
import os
import pickle
import subprocess
import shutil
import argparse
from scipy.spatial.distance import cdist
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
    
    # -------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------
    print(f"Evaluating flow field on {nGridY}x{nGridX} grid...")
    xVals = [np.min(XB)-0.5, np.max(XB)+0.5]
    yVals = [np.min(YB)-0.3, np.max(YB)+0.3]
    
    Xgrid = np.linspace(xVals[0], xVals[1], nGridX)
    Ygrid = np.linspace(yVals[0], yVals[1], nGridY)
    XX, YY = np.meshgrid(Xgrid, Ygrid)
    
    Vx = np.zeros((nGridY, nGridX))
    Vy = np.zeros((nGridY, nGridX))
    SDF = np.zeros((nGridY, nGridX))
    
    from scipy.spatial.distance import cdist
    airfoil_pts = np.column_stack((XB, YB))
    airfoil_path = Path(airfoil_pts)
    
    pts = np.column_stack((XX.ravel(), YY.ravel()))
    is_inside = airfoil_path.contains_points(pts).reshape(nGridY, nGridX)
    dist_to_obstacle = cdist(pts, airfoil_pts).min(axis=1).reshape(nGridY, nGridX)
    
    sdf1 = np.where(is_inside, -dist_to_obstacle, dist_to_obstacle)
    
    # SDF 2: distance to left and right walls (vertical flow setup)
    sdf2 = np.minimum(XX - xVals[0], xVals[1] - XX)
    
    # Flow region channel
    # 0=obstacle, 1=fluid, 2=wall, 3=inlet, 4=outlet
    flow_region = np.ones((nGridY, nGridX), dtype=np.float32)
    flow_region[is_inside] = 0.0
    
    # Assuming transposed boundaries for vertical flow representation
    flow_region[:, 0] = 2.0  # Left wall
    flow_region[:, -1] = 2.0 # Right wall
    flow_region[0, :] = 3.0  # Bottom inlet
    flow_region[-1, :] = 4.0 # Top outlet
    
    for m in range(nGridY):
        for n in range(nGridX):
            if is_inside[m, n]:
                Vx[m,n] = 0
                Vy[m,n] = 0
            else:
                XP = XX[m,n]
                YP = YY[m,n]
                Mx, My = STREAMLINE_SPM(XP, YP, XB, YB, phi, S)
                Nx, Ny = STREAMLINE_VPM(XP, YP, XB, YB, phi, S)
                Vx[m,n] = Vinf*np.cos(np.radians(AoA)) + np.sum(lam*Mx/(2*np.pi)) + np.sum(-gamma*Nx/(2*np.pi))
                Vy[m,n] = Vinf*np.sin(np.radians(AoA)) + np.sum(lam*My/(2*np.pi)) + np.sum(-gamma*Ny/(2*np.pi))
                
    Vxy = np.sqrt(Vx**2 + Vy**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        CpXY = 1 - (Vxy/Vinf)**2
        
    CpXY = np.nan_to_num(CpXY, nan=0.0)
    Vx = np.nan_to_num(Vx, nan=0.0)
    Vy = np.nan_to_num(Vy, nan=0.0)
    
    # Cast safely to float32 for deep learning pipelines but float64 is fine too
    dataX = np.stack([sdf1, flow_region, sdf2], axis=0).astype(np.float32)
    dataX = np.expand_dims(dataX, axis=0)
    
    dataY = np.stack([Vx, Vy, CpXY], axis=0).astype(np.float32)
    dataY = np.expand_dims(dataY, axis=0)
    
    out_X = os.path.join(output_dir, "dataX.pkl")
    out_Y = os.path.join(output_dir, "dataY.pkl")
    
    with open(out_X, 'wb') as f:
        pickle.dump(dataX, f)
        
    with open(out_Y, 'wb') as f:
        pickle.dump(dataY, f)
        
    print(f"Success. Wrote DataX of shape {dataX.shape} to {out_X} and DataY of shape {dataY.shape} to {out_Y}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create DeepCFD Data from Airfoil .dat")
    parser.add_argument("--dat_file", type=str, required=True, help="Path to airfoil .dat file")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory for .pkl files")
    parser.add_argument("--xfoil_exe", type=str, default="xfoil.exe", help="Path to the xfoil executable")
    parser.add_argument("--grid_y", type=int, default=172, help="Grid dimension Y")
    parser.add_argument("--grid_x", type=int, default=79, help="Grid dimension X")
    parser.add_argument("--vinf", type=float, default=1.0, help="Freestream velocity")
    parser.add_argument("--aoa", type=float, default=0.0, help="Angle of attack in degrees")
    args = parser.parse_args()
    
    create_dataset_for_airfoil(
        dat_file_path=args.dat_file,
        output_dir=args.out_dir,
        xfoil_exe=args.xfoil_exe,
        nGridY=args.grid_y,
        nGridX=args.grid_x,
        Vinf=args.vinf,
        AoA=args.aoa
    )
