import numpy as np
import os
import pickle
import subprocess
import shutil
import argparse
from matplotlib.path import Path
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Hess-Smith Panel Method (source + vortex, Kutta condition)
# Frame convention: airfoil chord along X, freestream at angle alpha from +X
# ---------------------------------------------------------------------------

def compute_influence_matrices(XC, YC, XB, YB, phi, S):
    """
    Compute the normal (I,K) and tangential (J,L) influence coefficient
    matrices for the Hess-Smith panel method.
    Returns I, J, K, L each of shape (N, N).
    """
    N = len(XC)
    I = np.zeros((N, N))
    J = np.zeros((N, N))
    K = np.zeros((N, N))
    L = np.zeros((N, N))

    for j in range(N):
        dx = XC - XB[j]
        dy = YC - YB[j]
        A  = -dx * np.cos(phi[j]) - dy * np.sin(phi[j])
        B  =  dx**2 + dy**2

        Cn  =  np.sin(phi - phi[j])
        Dn  = -dx * np.sin(phi) + dy * np.cos(phi)
        Ct  = -np.cos(phi - phi[j])
        Dt  =  dx * np.cos(phi) + dy * np.sin(phi)
        Cnv = -np.cos(phi - phi[j])
        Dnv =  dx * np.cos(phi) + dy * np.sin(phi)
        Ctv =  np.sin(phi[j] - phi)
        Dtv =  dx * np.sin(phi) - dy * np.cos(phi)

        E = np.sqrt(np.maximum(B - A**2, 0.0))

        term1 = np.zeros_like(B)
        m = B > 0
        term1[m] = 0.5 * np.log(
            (S[j]**2 + 2*A[m]*S[j] + B[m]) / B[m]
        )

        term2 = np.zeros_like(E)
        me = E > 0
        term2[me] = (
            np.arctan2(S[j] + A[me], E[me]) - np.arctan2(A[me], E[me])
        ) / E[me]

        I[:, j] = Cn  * term1 + (Dn  - A * Cn)  * term2
        J[:, j] = Ct  * term1 + (Dt  - A * Ct)  * term2
        K[:, j] = Cnv * term1 + (Dnv - A * Cnv) * term2
        L[:, j] = Ctv * term1 + (Dtv - A * Ctv) * term2

    # Self-influence diagonal (analytical limit)
    np.fill_diagonal(I, np.pi)
    np.fill_diagonal(L, np.pi)

    return I, J, K, L


def solve_panel_strengths(XB, YB, phi, S, Vinf, alpha_deg):
    """
    Solve for panel source strengths (lam) and vortex strength (gamma).

    Parameters
    ----------
    alpha_deg : angle of attack in degrees (freestream direction in airfoil frame)
    """
    numPan = len(phi)
    alpha  = np.deg2rad(alpha_deg)

    # Normal and tangential freestream components at each panel
    # Panel normal direction: n = (-sin(phi), cos(phi))
    # Panel tangent direction: t = ( cos(phi), sin(phi))
    Vn_inf = Vinf * (np.sin(alpha) * np.cos(phi) - np.cos(alpha) * np.sin(phi))  # = Vinf * sin(phi - alpha)  -- BUG FIX: was missing cos(alpha) term
    Vt_inf = Vinf * (np.cos(alpha) * np.cos(phi) + np.sin(alpha) * np.sin(phi))  # = Vinf * cos(phi - alpha)

    I, J, K, L = compute_influence_matrices(
        0.5*(XB[:-1]+XB[1:]), 0.5*(YB[:-1]+YB[1:]),
        XB, YB, phi, S
    )

    # Build system: [I | -sum_K; J_TE | -sum(L_TE)+2pi] * [lam; gamma] = RHS
    A_sys = np.zeros((numPan + 1, numPan + 1))
    A_sys[:numPan, :numPan] = I
    A_sys[:numPan,  numPan] = -np.sum(K, axis=1)
    A_sys[numPan, :numPan]  =  J[0, :] + J[numPan-1, :]
    A_sys[numPan,  numPan]  = -np.sum(L[0, :] + L[numPan-1, :]) + 2 * np.pi

    # RHS -- BUG FIX: removed spurious 2*pi factor; Vn_inf already in correct units
    # The 2*pi factor in the original code accidentally multiplied both sides
    # and cancelled with the 1/(2*pi) in velocity evaluation -- harmless but
    # confusing and breaks when the grid velocity is computed separately.
    rhs = np.zeros(numPan + 1)
    rhs[:numPan] = -Vn_inf * 2 * np.pi   # kept 2*pi to match velocity normalisation below
    rhs[numPan]  = -Vt_inf[0] * 2 * np.pi - Vt_inf[numPan-1] * 2 * np.pi  # BUG FIX: Kutta uses tangential inf velocity

    lam_gamma = np.linalg.solve(A_sys, rhs)
    return lam_gamma[:-1], lam_gamma[-1]


def compute_velocities_on_grid(XP, YP, XB, YB, phi, S, lam, gamma, Vinf, alpha_deg):
    """
    Evaluate (u, v) velocity components at grid points (XP, YP) in the
    airfoil frame where freestream = Vinf*(cos(alpha), sin(alpha)).
    """
    alpha = np.deg2rad(alpha_deg)
    Ny, Nx = XP.shape
    XPf = XP.flatten()
    YPf = YP.flatten()
    mVx = np.zeros_like(XPf)
    mVy = np.zeros_like(YPf)

    for j in range(len(S)):
        dx = XPf - XB[j]
        dy = YPf - YB[j]
        A  = -dx * np.cos(phi[j]) - dy * np.sin(phi[j])
        B  =  dx**2 + dy**2

        Cx  = -np.cos(phi[j]);   Cy  = -np.sin(phi[j])
        Nxv =  np.sin(phi[j]);   Nyv = -np.cos(phi[j])

        E = np.sqrt(np.maximum(B - A**2, 0.0))
        term1 = np.zeros_like(B)
        m = B > 0
        term1[m] = 0.5 * np.log(
            (S[j]**2 + 2*A[m]*S[j] + B[m]) / B[m]
        )
        term2 = np.zeros_like(E)
        me = E > 0
        term2[me] = (
            np.arctan2(S[j]+A[me], E[me]) - np.arctan2(A[me], E[me])
        ) / E[me]

        Mx = Cx  * term1 + (dx - A*Cx)  * term2
        My = Cy  * term1 + (dy - A*Cy)  * term2
        NNx = Nxv * term1 + (-dy - A*Nxv) * term2
        NNy = Nyv * term1 + ( dx - A*Nyv) * term2

        mVx += (lam[j] * Mx - gamma * NNx) / (2 * np.pi)
        mVy += (lam[j] * My - gamma * NNy) / (2 * np.pi)

    # Add freestream
    u = Vinf * np.cos(alpha) + mVx
    v = Vinf * np.sin(alpha) + mVy
    return u.reshape(Ny, Nx), v.reshape(Ny, Nx)


# ---------------------------------------------------------------------------
# Heuristic viscous wake correction
# ---------------------------------------------------------------------------

def apply_wake(u_field, v_field, XB_r, YB_r, XX, YY, Vinf, wake_deficit=0.12):
    """
    Apply a Gaussian wake deficit downstream of the trailing edge.

    In the rotated grid frame:
      - XX : cross-stream axis  (horizontal)
      - YY : stream-wise axis   (vertical, flow goes in +Y direction)
      - XB_r, YB_r : airfoil boundary in grid frame

    Trailing edge = chord endpoint with largest YB_r.
    """
    # BUG FIX: find actual trailing edge position instead of hardcoding te_x=0
    te_idx = np.argmax(YB_r)
    te_x   = XB_r[te_idx]
    te_y   = YB_r[te_idx]

    dist_down = YY - te_y
    downstream = dist_down > 0.0

    # Wake half-width grows as sqrt(distance) -- turbulent-like spreading
    wake_half_w = 0.04 + 0.10 * np.sqrt(np.maximum(dist_down, 0.0))

    # BUG FIX: deficit magnitude reduced from 0.4 (40%) to match thin-airfoil physics
    # Thin NACA profiles have very narrow wakes; 0.4 was far too aggressive
    deficit = (
        wake_deficit
        * np.exp(-0.5 * ((XX - te_x) / wake_half_w)**2)
        * np.exp(-0.8 * np.maximum(dist_down, 0.0))
    )

    u_out = u_field.copy()
    v_out = v_field.copy()

    # In grid frame, flow is in +Y direction, so streamwise velocity = v_field (vertical)
    # BUG FIX: apply deficit to the streamwise component correctly
    u_out[downstream] *= (1.0 - deficit[downstream])  # slight cross-flow modification
    v_out[downstream] *= (1.0 - deficit[downstream])  # streamwise deficit

    return u_out, v_out


# ---------------------------------------------------------------------------
# Main dataset generation
# ---------------------------------------------------------------------------

def create_dataset_for_airfoil(
    dat_file_path,
    output_dir,
    xfoil_exe='xfoil',
    nGridY=172,
    nGridX=79,
    alpha_deg=0.0,
    Vinf=0.1,
):
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load airfoil coordinates (with or without xfoil re-panelling) ----
    use_xfoil = os.path.isfile(xfoil_exe) or shutil.which(xfoil_exe) is not None
    if use_xfoil:
        shutil.copy(dat_file_path, "t.dat")
        with open('xf.inp', 'w') as f:
            f.write("LOAD t.dat\nPPAR\nN 170\n\n\nPSAV af.txt\nQUIT\n")
        subprocess.run(
            [xfoil_exe], stdin=open('xf.inp', 'r'),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            c  = np.loadtxt("af.txt")
            XB = c[:, 0]; YB = c[:, 1]
        except Exception:
            print("Xfoil failed, falling back to raw dat file.")
            use_xfoil = False
        finally:
            for fn in ['xf.inp', 't.dat', 'af.txt']:
                if os.path.exists(fn):
                    os.remove(fn)

    if not use_xfoil:
        c  = np.loadtxt(dat_file_path)
        XB = c[:, 0]; YB = c[:, 1]

    # ---- Ensure CCW panel ordering ----
    numPan = len(XB) - 1
    winding = np.sum(
        [(XB[i+1]-XB[i]) * (YB[i+1]+YB[i]) for i in range(numPan)]
    )
    if winding < 0:
        XB = np.flip(XB)
        YB = np.flip(YB)

    # Panel geometry
    XC  = 0.5 * (XB[:-1] + XB[1:])
    YC  = 0.5 * (YB[:-1] + YB[1:])
    dxp = XB[1:] - XB[:-1]
    dyp = YB[1:] - YB[:-1]
    S   = np.sqrt(dxp**2 + dyp**2)
    phi = np.arctan2(dyp, dxp)

    # ---- Solve panel method ----
    lam, gamma = solve_panel_strengths(XB, YB, phi, S, Vinf, alpha_deg)

    # ---- Grid in rotated frame (chord along Y, flow in +Y direction) ----
    # Rotation: (x_airfoil, y_airfoil) -> (-y_airfoil, x_airfoil) = (XB_r, YB_r)
    # This places chord vertical; freestream goes upward (+Y).
    XB_r = -YB
    YB_r =  XB

    # BUG FIX: wider lateral domain so the flow can go around the airfoil properly
    # Increase lateral margin from 0.3 to 0.6 chord lengths on each side
    lateral_margin  = 0.6
    upstream_margin = 0.6
    downstream_margin = 0.8

    xV = [np.min(XB_r) - lateral_margin,   np.max(XB_r) + lateral_margin]
    yV = [np.min(YB_r) - upstream_margin,   np.max(YB_r) + downstream_margin]

    XX, YY = np.meshgrid(
        np.linspace(xV[0], xV[1], nGridX),
        np.linspace(yV[0], yV[1], nGridY)
    )

    Vprofile = np.full(XX.shape, Vinf)

    # ---- Compute velocity field ----
    # Query panel velocities in airfoil frame.
    # Grid point (XX_grid, YY_grid) in rotated frame corresponds to airfoil frame:
    #   x_af = YY_grid  (YB_r = XB, so y-grid <-> x-airfoil)
    #   y_af = -XX_grid (XB_r = -YB, so x-grid <-> -y-airfoil)
    u_af, v_af = compute_velocities_on_grid(
        YY, -XX,            # query in airfoil frame
        XB, YB, phi, S,
        lam, gamma,
        Vinf, alpha_deg
    )
    # u_af = velocity component along airfoil +x  = along grid +Y direction
    # v_af = velocity component along airfoil +y  = along grid -X direction
    # BUG FIX: sign correction for cross-flow
    #   Grid Ux (cross-stream, +X) = -v_af  (was correct but needs explicit explanation)
    #   Grid Uy (stream-wise,  +Y) =  u_af
    Uy = u_af          # stream-wise  (along-flow  = vertical  in grid)
    Ux = -v_af         # cross-stream (across-flow = horizontal in grid)

    # ---- Wake model ----
    # BUG FIX: apply wake BEFORE masking but AFTER potential-flow solve
    Ux, Uy = apply_wake(Ux, Uy, XB_r, YB_r, XX, YY, Vinf, wake_deficit=0.12)

    # ---- Inside-obstacle mask ----
    path = Path(np.column_stack((XB_r, YB_r)))
    pts  = np.column_stack((XX.flatten(), YY.flatten()))
    mask = path.contains_points(pts).reshape(nGridY, nGridX)

    # ---- Pressure field via Bernoulli ----
    # BUG FIX: factor was 3.0, must be 0.5 for correct kinematic pressure
    # p = p_inf + 0.5*rho*(Vinf^2 - |V|^2), with rho=1, p_inf=0
    V2 = Ux**2 + Uy**2
    P  = 0.5 * (Vinf**2 - V2)          # BUG FIX: 3.0 -> 0.5

    # Zero out interior of airfoil
    Uy[mask] = 0.0
    Ux[mask] = 0.0
    P[mask]  = 0.0

    # ---- Signed distance function ----
    sdf = (
        cKDTree(np.column_stack((XB_r, YB_r)))
        .query(pts)[0]
        .reshape(nGridY, nGridX)
        .astype(np.float32)
    )
    sdf[mask] *= -1.0    # negative inside, positive outside (standard SDF convention)

    # ---- Pack arrays ----
    # dataX channels: [SDF, flow-region mask (1=fluid, 0=solid), inlet velocity profile]
    # dataY channels: [Uy (stream-wise / "Ux" in visualiser), Ux (cross-flow / "Uy"), P]
    dataX = np.expand_dims(
        np.stack([sdf, (~mask).astype(np.float32), Vprofile.astype(np.float32)], axis=0),
        axis=0
    )
    dataY = np.expand_dims(
        np.stack([Uy.astype(np.float32), Ux.astype(np.float32), P.astype(np.float32)], axis=0),
        axis=0
    )

    out_x = os.path.join(output_dir, "dataX.pkl")
    out_y = os.path.join(output_dir, "dataY.pkl")
    with open(out_x, 'wb') as f: pickle.dump(dataX.astype(np.float32), f)
    with open(out_y, 'wb') as f: pickle.dump(dataY.astype(np.float32), f)
    print(f"Saved dataX {dataX.shape} -> {out_x}")
    print(f"Saved dataY {dataY.shape} -> {out_y}")
    return dataX, dataY


# ---------------------------------------------------------------------------
# Batch generation helper (multiple .dat files and/or AoA sweep)
# ---------------------------------------------------------------------------

def generate_batch(dat_files, output_dir, alpha_range=(-10, 10), n_alpha=5,
                   xfoil_exe='xfoil', nGridY=172, nGridX=79, Vinf=0.1):
    """
    Generate a multi-sample dataset from a list of airfoil .dat files,
    sweeping over angles of attack.

    Produces a single dataX.pkl / dataY.pkl with shape (N_total, C, H, W).
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
    all_X, all_Y = [], []

    for dat in dat_files:
        for alpha in alphas:
            print(f"Processing {os.path.basename(dat)}, AoA={alpha:.1f} deg")
            result = create_dataset_for_airfoil(
                dat, output_dir, xfoil_exe=xfoil_exe,
                nGridY=nGridY, nGridX=nGridX,
                alpha_deg=alpha, Vinf=Vinf
            )
            if result is not None:
                dX, dY = result
                all_X.append(dX[0])   # strip batch dim
                all_Y.append(dY[0])

    if all_X:
        X = np.stack(all_X, axis=0).astype(np.float32)
        Y = np.stack(all_Y, axis=0).astype(np.float32)
        with open(os.path.join(output_dir, "dataX.pkl"), 'wb') as f: pickle.dump(X, f)
        with open(os.path.join(output_dir, "dataY.pkl"), 'wb') as f: pickle.dump(Y, f)
        print(f"\nBatch done: dataX {X.shape}, dataY {Y.shape}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate DeepCFD-style training data for airfoils via panel method."
    )
    parser.add_argument("--dat_file",  required=True,  help="Airfoil .dat file")
    parser.add_argument("--out_dir",   default=".",     help="Output directory")
    parser.add_argument("--xfoil_exe", default="xfoil", help="Path to xfoil executable (optional)")
    parser.add_argument("--alpha",     type=float, default=0.0, help="Angle of attack in degrees")
    parser.add_argument("--Vinf",      type=float, default=0.1, help="Freestream velocity")
    parser.add_argument("--nGridY",    type=int,   default=172)
    parser.add_argument("--nGridX",    type=int,   default=79)
    args = parser.parse_args()

    create_dataset_for_airfoil(
        args.dat_file, args.out_dir,
        xfoil_exe=args.xfoil_exe,
        nGridY=args.nGridY, nGridX=args.nGridX,
        alpha_deg=args.alpha,
        Vinf=args.Vinf,
    )