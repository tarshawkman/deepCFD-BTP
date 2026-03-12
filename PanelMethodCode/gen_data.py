import numpy as np
import pickle
from matplotlib.path import Path

# Corrected Import
from spvp_airfoil import (
    COMPUTE_IJ_SPM,
    COMPUTE_KL_VPM,
    STREAMLINE_SPM,
    STREAMLINE_VPM
)

# -----------------------------
# SETTINGS
# -----------------------------
airfoil_file = "naca0015.dat"
Vinf = 1.0
AoA = 0.0

# DeepCFD Expected Dimensions!
Nx = 172
Ny = 79

# -----------------------------
# LOAD AIRFOIL
# -----------------------------
coords = np.loadtxt(airfoil_file, skiprows=1)
XB = coords[:,0]
YB = coords[:,1]

numPts = len(XB)
numPan = numPts - 1

edge = np.zeros(numPan)
for i in range(numPan):
    edge[i] = (XB[i+1]-XB[i])*(YB[i+1]+YB[i])

if np.sum(edge) < 0:
    XB = np.flip(XB)
    YB = np.flip(YB)

# -----------------------------
# PANEL GEOMETRY
# -----------------------------
XC = np.zeros(numPan)
YC = np.zeros(numPan)
S  = np.zeros(numPan)
phi = np.zeros(numPan)

for i in range(numPan):
    XC[i] = 0.5*(XB[i] + XB[i+1])
    YC[i] = 0.5*(YB[i] + YB[i+1])
    dx = XB[i+1] - XB[i]
    dy = YB[i+1] - YB[i]
    S[i] = np.sqrt(dx**2 + dy**2)
    phi[i] = np.arctan2(dy,dx)

delta = phi + np.pi/2
beta  = delta - np.radians(AoA)

# -----------------------------
# PANEL INFLUENCE MATRICES
# -----------------------------
I, J = COMPUTE_IJ_SPM(XC, YC, XB, YB, phi, S)
K, L = COMPUTE_KL_VPM(XC, YC, XB, YB, phi, S)

A = np.zeros((numPan+1, numPan+1))
for i in range(numPan):
    for j in range(numPan):
        if i == j: A[i,j] = np.pi
        else:      A[i,j] = I[i,j]

for i in range(numPan): A[i,numPan] = -np.sum(K[i,:])
for j in range(numPan): A[numPan,j] = J[0,j] + J[numPan-1,j]

A[numPan,numPan] = -np.sum(L[0,:] + L[numPan-1,:]) + 2*np.pi
b = np.zeros(numPan+1)
for i in range(numPan): b[i] = -Vinf*2*np.pi*np.cos(beta[i])
b[numPan] = -Vinf*2*np.pi*(np.sin(beta[0]) + np.sin(beta[numPan-1]))

sol = np.linalg.solve(A,b)
lam = sol[:-1]
gamma = sol[-1]

# -----------------------------
# GRID FOR FLOW FIELD
# -----------------------------
xVals = [np.min(XB)-0.5, np.max(XB)+0.5]
yVals = [np.min(YB)-0.3, np.max(YB)+0.3]

Xgrid = np.linspace(xVals[0], xVals[1], Nx)
Ygrid = np.linspace(yVals[0], yVals[1], Ny)
XX,YY = np.meshgrid(Xgrid,Ygrid)

Vx = np.zeros((Ny,Nx))
Vy = np.zeros((Ny,Nx))
SDF = np.zeros((Ny,Nx)) # For DataX

airfoil_path = Path(np.column_stack((XB,YB)))

# -----------------------------
# COMPUTE VELOCITY FIELD
# -----------------------------
print(f"Starting {Nx}x{Ny} Grid Evaluation... This will take ~2 minutes!")
for m in range(Ny):
    for n in range(Nx):
        XP = XX[m,n]
        YP = YY[m,n]

        if airfoil_path.contains_point((XP,YP)):
            SDF[m,n] = 1.0 # Inside obstacle
            continue

        Mx,My = STREAMLINE_SPM(XP,YP,XB,YB,phi,S)
        Nx_,Ny_ = STREAMLINE_VPM(XP,YP,XB,YB,phi,S)

        Vx[m,n] = Vinf*np.cos(np.radians(AoA)) + np.sum(lam*Mx/(2*np.pi)) - np.sum(gamma*Nx_/(2*np.pi))
        Vy[m,n] = Vinf*np.sin(np.radians(AoA)) + np.sum(lam*My/(2*np.pi)) - np.sum(gamma*Ny_/(2*np.pi))

# -----------------------------
# FORMAT FOR DEEPCFD
# -----------------------------
Vmag = np.sqrt(Vx**2 + Vy**2)
Cp = 1 - (Vmag/Vinf)**2

# dataY = [Vx, Vy, Pressure]
dataY_sample = np.stack([Vx, Vy, 0.5 * Cp], axis=0) # shape: (3, 79, 172)
dataY_tensor = np.expand_dims(dataY_sample, axis=0) # shape: (1, 3, 79, 172)

# dataX = [SDF, V_in_x, V_in_y]
V_in_x = np.full((Ny,Nx), Vinf * np.cos(np.radians(AoA)))
V_in_y = np.full((Ny,Nx), Vinf * np.sin(np.radians(AoA)))
dataX_sample = np.stack([SDF, V_in_x, V_in_y], axis=0)
dataX_tensor = np.expand_dims(dataX_sample, axis=0)

# Save
with open("custom_dataX.pkl","wb") as f: pickle.dump(dataX_tensor, f)
with open("custom_dataY.pkl","wb") as f: pickle.dump(dataY_tensor, f)

print(f"Success! dataY Shape: {dataY_tensor.shape}, dataX Shape: {dataX_tensor.shape}")
