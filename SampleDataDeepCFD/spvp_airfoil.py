import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import tkinter as tk
from tkinter import filedialog
from scipy.interpolate import RegularGridInterpolator
from matplotlib.path import Path

def XFOIL(NACA, PPAR, AoA, flagAirfoil):
    xFoilResults = {}
    success = 0
    airfoilName = ""
    flnm = ""
    
    if flagAirfoil['XFoilCreate'] == 1:
        airfoilName = NACA
        xFoilResults['afName'] = airfoilName
        success = 1
    elif flagAirfoil['XFoilLoad'] == 1:
        root = tk.Tk()
        root.withdraw()
        # Ensure focus
        root.attributes('-topmost', 1)
        file_path = filedialog.askopenfilename(initialdir="./Airfoil_DAT_Selig", title="Select Airfoil File", filetypes=[("DAT files", "*.dat")])
        root.destroy()
        if not file_path:
            return xFoilResults, 0
        flnm = os.path.basename(file_path)
        airfoilName = flnm[:-4]
        xFoilResults['afName'] = airfoilName
        success = 1
    
    saveFlnm = f"Save_{airfoilName}.txt"
    saveFlnmCp = f"Save_{airfoilName}_Cp.txt"
    saveFlnmPol = f"Save_{airfoilName}_Pol.txt"
    
    for f in [saveFlnm, saveFlnmCp, saveFlnmPol]:
        if os.path.exists(f):
            os.remove(f)
            
    with open('xfoil_input.inp', 'w') as fid:
        if flagAirfoil['XFoilLoad'] == 1:
            fid.write(f"LOAD ./Airfoil_DAT_Selig/{flnm}\n")
        elif flagAirfoil['XFoilCreate'] == 1:
            fid.write(f"NACA {NACA}\n")
            
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
        
    with open('xfoil_input.inp', 'r') as fid_in:
        subprocess.run(['xfoil.exe'], stdin=fid_in, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists('xfoil_input.inp'):
        os.remove('xfoil_input.inp')
        
    # READ CP DATA
    try:
        cp_data = np.loadtxt(saveFlnmCp, skiprows=3)
        os.remove(saveFlnmCp)
        xFoilResults['X'] = cp_data[:, 0]
        xFoilResults['Y'] = cp_data[:, 1]
        xFoilResults['CP'] = cp_data[:, 2]
    except:
        xFoilResults['X'] = np.array([])
        xFoilResults['Y'] = np.array([])
        xFoilResults['CP'] = np.array([])
        
    # READ AIRFOIL COORDINATES
    try:
        # Some xfoil dumps might have strange formatting, we skip lines if needed but usually standard float
        af_coords = np.loadtxt(saveFlnm)
        os.remove(saveFlnm)
        xFoilResults['XB'] = af_coords[:, 0]
        xFoilResults['YB'] = af_coords[:, 1]
    except:
        xFoilResults['XB'] = np.array([])
        xFoilResults['YB'] = np.array([])

    # READ POLAR DATA
    try:
        with open(saveFlnmPol, 'r') as f:
            lines = f.readlines()
        # header might be line 12 in 0-indexed terms (so index 12 corresponds to 13th line)
        # We find the first line that parses to numbers
        pol_vals = None
        for line in lines:
            parts = line.split()
            if len(parts) >= 7 and parts[0].replace('.','',1).replace('-','',1).isdigit():
                pol_vals = [float(x) for x in parts]
                break
        os.remove(saveFlnmPol)
        if pol_vals:
            xFoilResults['CL'] = pol_vals[1]
            xFoilResults['CD'] = pol_vals[2]
            xFoilResults['CM'] = pol_vals[4]
        else:
            xFoilResults['CL'] = 0.0
            xFoilResults['CD'] = 0.0
            xFoilResults['CM'] = 0.0
    except:
        xFoilResults['CL'] = 0.0
        xFoilResults['CD'] = 0.0
        xFoilResults['CM'] = 0.0

    return xFoilResults, success

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

def COMPUTE_CIRCULATION(a, b, x0, y0, numT, Vx, Vy, Xgrid, Ygrid):
    tEnd = (2*np.pi) - ((2*np.pi)/numT)
    t = np.linspace(0, tEnd, numT)
    xC = a*np.cos(t) + x0
    yC = b*np.sin(t) + y0
    
    interp_Vx = RegularGridInterpolator((Ygrid, Xgrid), Vx, bounds_error=False, fill_value=0.0)
    interp_Vy = RegularGridInterpolator((Ygrid, Xgrid), Vy, bounds_error=False, fill_value=0.0)
    
    pts = np.vstack((yC, xC)).T
    VxC = interp_Vx(pts)
    VyC = interp_Vy(pts)
    
    Gamma = -(np.trapz(VxC, x=xC) + np.trapz(VyC, x=yC))
    return Gamma, xC, yC, VxC, VyC

if __name__ == '__main__':
    # -------------------------------------------------------------
    # KNOWNS
    # -------------------------------------------------------------
    flagAirfoil = {
        'XFoilCreate': 0,
        'XFoilLoad': 1
    }
    Vinf = 1.0
    AoA  = 0.0
    NACA = '0015'
    
    # 1=on, 0=off
    flagPlot = [1, 1, 1, 1, 1, 1]
    
    PPAR = {
        'N': '170',
        'P': '4',
        'T': '1',
        'R': '1',
        'XT': '1 1',
        'XB': '1 1'
    }
    
    xFoilResults, success = XFOIL(NACA, PPAR, AoA, flagAirfoil)
    if success == 0:
        print("XFOIL generation/loading failed or was cancelled.")
        exit()
        
    afName = xFoilResults['afName']
    xFoilX = xFoilResults['X']
    xFoilY = xFoilResults['Y']
    xFoilCP = xFoilResults['CP']
    XB = xFoilResults['XB']
    YB = xFoilResults['YB']
    xFoilCL = xFoilResults['CL']
    xFoilCD = xFoilResults['CD']
    xFoilCM = xFoilResults['CM']
    
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
    
    # Panel Method Geometry
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
    
    # Compute Source and Vortex Panel Strengths
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
    
    # Compute panel velocities and Cp
    Vt = np.zeros(numPan)
    Cp = np.zeros(numPan)
    for i in range(numPan):
        term1 = Vinf*np.sin(beta[i])
        term2 = (1/(2*np.pi)) * np.sum(lam * J[i,:])
        term3 = gamma/2
        term4 = -(gamma/(2*np.pi)) * np.sum(L[i,:])
        
        Vt[i] = term1 + term2 + term3 + term4
        Cp[i] = 1 - (Vt[i]/Vinf)**2
        
    # Lift and Moment
    CN = -Cp * S * np.sin(beta)
    CA = -Cp * S * np.cos(beta)
    
    CL = np.sum(CN * np.cos(np.radians(AoA))) - np.sum(CA * np.sin(np.radians(AoA)))
    CM = np.sum(Cp * (XC-0.25) * S * np.cos(phi))
    
    print('======= RESULTS =======')
    print('Lift Coefficient (CL)')
    print(f'\tSPVP : {CL:.4f}')
    print(f'\tK-J  : {2*np.sum(gamma*S):.4f}')
    print(f'\tXFOIL: {xFoilCL:.4f}')
    print('Moment Coefficient (CM)')
    print(f'\tSPVP : {CM:.4f}')
    print(f'\tXFOIL: {xFoilCM:.4f}')
    
    # Compute streamlines
    Vx, Vy = None, None
    if flagPlot[4] or flagPlot[5]:
        nGridX = 100
        nGridY = 100
        xVals = [np.min(XB)-0.5, np.max(XB)+0.5]
        yVals = [np.min(YB)-0.3, np.max(YB)+0.3]
        
        stepsize = 0.01
        slPct = 25
        Ysl = np.linspace(yVals[0], yVals[1], int((slPct/100)*nGridY))
        
        Xgrid = np.linspace(xVals[0], xVals[1], nGridX)
        Ygrid = np.linspace(yVals[0], yVals[1], nGridY)
        XX, YY = np.meshgrid(Xgrid, Ygrid)
        
        Vx = np.zeros((nGridY, nGridX))
        Vy = np.zeros((nGridY, nGridX))
        
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
                else:
                    Vx[m,n] = Vinf*np.cos(np.radians(AoA)) + np.sum(lam*Mx/(2*np.pi)) + np.sum(-gamma*Nx/(2*np.pi))
                    Vy[m,n] = Vinf*np.sin(np.radians(AoA)) + np.sum(lam*My/(2*np.pi)) + np.sum(-gamma*Ny/(2*np.pi))
                    
        Vxy = np.sqrt(Vx**2 + Vy**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            CpXY = 1 - (Vxy/Vinf)**2
            
        aa = 0.75
        bb = 0.25
        x0 = 0.5
        y0 = 0
        numT = 5000
        Circulation, xC, yC, VxC, VyC = COMPUTE_CIRCULATION(aa, bb, x0, y0, numT, Vx, Vy, Xgrid, Ygrid)
        
        print('======= CIRCULATION RESULTS =======')
        print(f'Sum of L     : {np.sum(lam*S):.7f}')
        print(f'Sum of G     : {np.sum(gamma*S):.7f}')
        print(f'Circulation  : {Circulation:.7f}')
        print(f'K-J from G   : {2*np.sum(gamma*S):.7f}')
        print(f'K-J from Circ: {2*Circulation:.7f}')
        print('===================================')
        
    # Plotting
    if flagPlot[0]:
        plt.figure(1)
        plt.fill(XB, YB, 'k')
        for i in range(numPan):
            x_pts = [XC[i], XC[i] + S[i]*np.cos(np.radians(betaD[i]+AoA))]
            y_pts = [YC[i], YC[i] + S[i]*np.sin(np.radians(betaD[i]+AoA))]
            plt.plot(x_pts, y_pts, 'r-', linewidth=2)
        plt.xlabel('X Units')
        plt.ylabel('Y Units')
        plt.axis('equal')
        plt.gcf().canvas.manager.set_window_title('Geometric Bounds and Normals')
        
    if flagPlot[1]:
        plt.figure(2)
        plt.plot(XB, YB, 'k-', linewidth=3)
        plt.plot(XB[:2], YB[:2], 'g-', linewidth=2, label='First Panel')
        plt.plot(XB[1:3], YB[1:3], 'm-', linewidth=2, label='Second Panel')
        plt.plot(XB, YB, 'ko', markerfacecolor='k', label='Boundary')
        plt.plot(XC, YC, 'ko', markerfacecolor='r', label='Control')
        plt.legend()
        plt.xlabel('X Units')
        plt.ylabel('Y Units')
        plt.axis('equal')
        plt.gcf().canvas.manager.set_window_title('Panels')
        
    if flagPlot[2]:
        plt.figure(3)
        Cps = np.abs(Cp*0.25)
        for i in range(len(Cps)):
            x_pts = [XC[i], XC[i] + Cps[i]*np.cos(np.radians(betaD[i]+AoA))]
            y_pts = [YC[i], YC[i] + Cps[i]*np.sin(np.radians(betaD[i]+AoA))]
            if Cp[i] < 0:
                plt.plot(x_pts, y_pts, 'r-', linewidth=2)
            else:
                plt.plot(x_pts, y_pts, 'b-', linewidth=2)
        plt.fill(XB, YB, 'k')
        plt.xlabel('X Units')
        plt.ylabel('Y Units')
        plt.axis('equal')
        plt.gcf().canvas.manager.set_window_title('Cp Vectors')
        
    if flagPlot[3] and len(xFoilCP) > 0:
        plt.figure(4)
        midIndX = len(xFoilCP)//2
        midIndS = len(Cp)//2
        plt.plot(xFoilX[:midIndX], xFoilCP[:midIndX], 'b-', linewidth=2, label='XFOIL Upper')
        plt.plot(xFoilX[midIndX:], xFoilCP[midIndX:], 'r-', linewidth=2, label='XFOIL Lower')
        plt.plot(XC[:midIndS], Cp[:midIndS], 'ks', markerfacecolor='r', label='VPM Upper')
        plt.plot(XC[midIndS:], Cp[midIndS:], 'ks', markerfacecolor='b', label='VPM Lower')
        plt.legend()
        plt.xlabel('X Coordinate')
        plt.ylabel('Cp')
        plt.xlim([0, 1])
        plt.gca().invert_yaxis()
        plt.title(f"Airfoil: {afName}, CL_VPM/CL_XFOIL = {2*np.sum(gamma*S):.4g}/{xFoilCL:.4g}")
        plt.gcf().canvas.manager.set_window_title('Cp Comparison')
        
    if flagPlot[4] and Vx is not None:
        plt.figure(5)
        start_pts = np.column_stack((np.full_like(Ysl, xVals[0]), Ysl))
        try:
            plt.streamplot(XX, YY, Vx, Vy, start_points=start_pts, density=5)
        except:
            # Fallback if start_points isn't supported in current matplotlib version
            plt.streamplot(XX, YY, Vx, Vy, density=1.5)
        plt.fill(XB, YB, 'k')
        plt.xlabel('X Units')
        plt.ylabel('Y Units')
        plt.axis('equal')
        plt.xlim(xVals)
        plt.ylim(yVals)
        plt.gcf().canvas.manager.set_window_title('Streamlines')
        
    if flagPlot[5] and Vx is not None:
        plt.figure(6)
        plt.contourf(XX, YY, CpXY, 100)
        plt.fill(XB, YB, 'k')
        plt.xlabel('X Units')
        plt.ylabel('Y Units')
        plt.axis('equal')
        plt.xlim(xVals)
        plt.ylim(yVals)
        plt.gcf().canvas.manager.set_window_title('Pressure Contour')
        
    if any(flagPlot):
        plt.show()
