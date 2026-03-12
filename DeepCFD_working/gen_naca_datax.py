import numpy as np
import pickle
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# ==============================================================
# 1️⃣ Generate NACA0015 airfoil geometry
# ==============================================================
def naca4(code="0015", n_points=200):
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    x = np.linspace(0, 1, n_points)
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    if p == 0:
        yc = np.zeros_like(x)
    else:
        yc = np.where(x < p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1-2*p)+2*p*x - x**2))
    xu, yu = x, yc + yt
    xl, yl = x, yc - yt
    return np.concatenate([xu[::-1], xl[1:]]), np.concatenate([yu[::-1], yl[1:]])

# ==============================================================
# 2️⃣ Create computational grid (match 172×79)
# ==============================================================
Nx, Ny = 172, 79
X, Y = np.meshgrid(np.linspace(-0.5, 1.5, Nx), np.linspace(-0.5, 0.5, Ny))

# ==============================================================
# 3️⃣ Signed distance field for airfoil
# ==============================================================
x_airfoil, y_airfoil = naca4("0015")
pts = np.vstack([x_airfoil, y_airfoil]).T
tree = cKDTree(pts)
d, _ = tree.query(np.c_[X.ravel(), Y.ravel()])
SDF_obstacle = d.reshape(Ny, Nx)

# ==============================================================
# 4️⃣ Flow region mask
# ==============================================================
FlowRegion = np.ones_like(SDF_obstacle)
FlowRegion[SDF_obstacle < 0.005] = 0  # near obstacle = solid region
FlowRegion[Y < -0.45] = 3
FlowRegion[Y > 0.45] = 3

# ==============================================================
# 5️⃣ SDF from top/bottom walls
# ==============================================================
y_top, y_bottom = 0.5, -0.5
SDF_topbottom = np.minimum(y_top - Y, Y - y_bottom)

# ==============================================================
# 6️⃣ Combine and save
# ==============================================================
input_tensor = np.stack([SDF_obstacle, FlowRegion, SDF_topbottom], axis=0)
input_tensor = input_tensor[np.newaxis, ...]  # (1, 3, 172, 79)

with open("dataX_naca0015.pkl", "wb") as f:
    pickle.dump(input_tensor.astype(np.float32), f)

print("Saved dataX_naca0015.pkl with shape", input_tensor.shape)

# optional visualization
plt.imshow(SDF_obstacle, cmap='jet')
plt.title("SDF - NACA0015")
plt.colorbar()
plt.show()
