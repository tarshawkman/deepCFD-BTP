import numpy as np

def naca4(code="0015", n_points=200):
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    x = np.linspace(0,1,n_points)
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    yc = np.zeros_like(x)
    xu, yu = x, yc+yt
    xl, yl = x, yc-yt
    return np.concatenate([xu[::-1], xl[1:]]), np.concatenate([yu[::-1], yl[1:]])

x,y = naca4("0015", 300)
np.savetxt("naca0015.dat", np.c_[x,y], fmt="%.6f")
print("Saved naca0015.dat")
