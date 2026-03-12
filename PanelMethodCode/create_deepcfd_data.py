import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.path import Path
from spvp_airfoil import XFOIL, COMPUTE_IJ_SPM, COMPUTE_KL_VPM, STREAMLINE_SPM, STREAMLINE_VPM

def create_dataset_for_airfoil(dat_file_path, output_dir, nGridY=172, nGridX=79):
    # -------------------------------------------------------------
    # KNOWNS
    # -------------------------------------------------------------
    flagAirfoil = {
        'XFoilCreate': 0,
        'XFoilLoad': 1
    }
    Vinf = 1.0
    AoA  = 0.0
    
    PPAR = {
        'N': '170',
        'P': '4',
        'T': '1',
        'R': '1',
        'XT': '1 1',
        'XB': '1 1'
    }
    
    import shutil
    flnm = os.path.basename(dat_file_path.replace("\\", "/"))
    airfoilName = flnm[:-4]
    
    # Copy to local dir for xfoil to read easily
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
        
    import subprocess
    with open('xfoil_input.inp', 'r') as fid_in:
        subprocess.run(['xfoil.exe'], stdin=fid_in, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists('xfoil_input.inp'):
        os.remove('xfoil_input.inp')
    if os.path.exists(local_dat):
        os.remove(local_dat)
        
    # Read geometry
    try:
        # Some xfoil dumps might have strange formatting, we skip lines if needed but usually standard float
        af_coords = np.loadtxt(saveFlnm)
        os.remove(saveFlnm)
        XB = af_coords[:, 0]
        YB = af_coords[:, 1]
    except:
        XB = np.array([])
        YB = np.array([])

    if len(XB) == 0:
        print("Failed to run XFOIL or read coordinates.")
        return

    numPts = len(XB)
    numPan = numPts - 1
    
    # Check for direction of points
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
    # DeepCFD Grid Evaluation
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
    
    airfoil_path = Path(np.column_stack((XB, YB)))
    
    for m in range(nGridY):
        for n in range(nGridX):
            XP = XX[m,n]
            YP = YY[m,n]
            
            Mx, My = STREAMLINE_SPM(XP, YP, XB, YB, phi, S)
            Nx, Ny = STREAMLINE_VPM(XP, YP, XB, YB, phi, S)
            
            if airfoil_path.contains_point((XP,YP)):
                Vx[m,n] = 0
                Vy[m,n] = 0
                SDF[m,n] = 1.0  # inside obstacle
            else:
                Vx[m,n] = Vinf*np.cos(np.radians(AoA)) + np.sum(lam*Mx/(2*np.pi)) + np.sum(-gamma*Nx/(2*np.pi))
                Vy[m,n] = Vinf*np.sin(np.radians(AoA)) + np.sum(lam*My/(2*np.pi)) + np.sum(-gamma*Ny/(2*np.pi))
                SDF[m,n] = 0.0  # fluid
                
    Vxy = np.sqrt(Vx**2 + Vy**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        CpXY = 1 - (Vxy/Vinf)**2
        
    # Replace nans
    CpXY = np.nan_to_num(CpXY, nan=0.0)
    Vx = np.nan_to_num(Vx, nan=0.0)
    Vy = np.nan_to_num(Vy, nan=0.0)
        
    # Formatting for DeepCFD: Shape (1, 3, 172, 79) -> N=1
    # DataX: [SDF, V_in_x, V_in_y]
    # Here V_in_x = Vinf * cos(AoA) everywhere, V_in_y = Vinf * sin(AoA) everywhere
    V_in_x = np.full((nGridY, nGridX), Vinf * np.cos(np.radians(AoA)))
    V_in_y = np.full((nGridY, nGridX), Vinf * np.sin(np.radians(AoA)))
    
    dataX = np.stack([SDF, V_in_x, V_in_y], axis=0)
    dataX = np.expand_dims(dataX, axis=0)
    
    # DataY: [Vx, Vy, Cp] or [Vx, Vy, Pressure]
    dataY = np.stack([Vx, Vy, CpXY], axis=0)
    dataY = np.expand_dims(dataY, axis=0)
    
    out_X = os.path.join(output_dir, "custom_dataX.pkl")
    out_Y = os.path.join(output_dir, "custom_dataY.pkl")
    
    with open(out_X, 'wb') as f:
        pickle.dump(dataX, f)
        
    with open(out_Y, 'wb') as f:
        pickle.dump(dataY, f)
        
    print(f"Successfully generated custom_dataX.pkl and custom_dataY.pkl in {output_dir}")
    print(f"DataX shape: {dataX.shape}")
    print(f"DataY shape: {dataY.shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create DeepCFD Data from Airfoil .dat")
    parser.add_argument("dat_file", type=str, help="Path to airfoil .dat file")
    parser.add_argument("out_dir", type=str, help="Output directory for .pkl files")
    args = parser.parse_args()
    
    create_dataset_for_airfoil(args.dat_file, args.out_dir)
