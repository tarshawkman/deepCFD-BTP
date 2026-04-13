"""
generate_data.py  –  newCurrent version
========================================
Generates ONE sample of DeepCFD-compatible dataX.pkl and dataY.pkl from a
single airfoil .dat file, using the Hess-Smith Source+Vortex Panel Method
(SVPM) for potential flow.

The raw .dat file is first refined to ~200 points using cubic spline
interpolation so that even sparse coordinate files (e.g. 35-point NACA files)
produce smooth results suitable for the panel method and SDF computation.

Toy-dataset format (re-engineered from dataX/dataY inspection):
  dataX shape : (1, 3, 172, 79)
    Ch 0 – SDF1  : signed distance to airfoil surface (negative inside, clipped at -50)
    Ch 1 – BC    : boundary-condition marker  0=solid 1=fluid 2=wall 3=top 4=bottom
    Ch 2 – Vinf  : triangular inlet-velocity profile across the channel width
                   (mimics the Poiseuille-like parabolic profile in the toy data)

  dataY shape : (1, 3, 172, 79)
    Ch 0 – Ux   : stream-wise velocity  (flow goes in +Y of grid → "Ux" in visualiser)
    Ch 1 – Uy   : cross-stream velocity (horizontal perturbations)
    Ch 2 – P    : static pressure → 0.5*(Vinf²−|V|²)  (Bernoulli, density=1)

Grid conventions (matches toy dataset):
  • Grid: 172 rows × 79 cols
  • Airfoil chord along Y-axis (rotated 90° CCW from standard)
  • Freestream flows vertically upward (+Y direction)
  • Domain: chord-centered, with ±0.6 chord lateral margin
    and 0.6 upstream + 0.8 downstream margin

Method:
  Standard potential-flow panel method gives correct Ux/Uy away from surface.
  A heuristic viscous wake is added downstream of the trailing edge.
  The paper used OpenFOAM (simpleFoam, laminar, Re≈100-1000), but panel
  method matches the topology well enough for training/demonstration purposes.

Usage:
  python generate_data.py --dat_file NACA0015.dat [--out_dir .] [--alpha 0.0] [--Vinf_peak 0.1]
"""

import numpy as np
import os
import pickle
import argparse
import warnings
from matplotlib.path import Path
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# GRID PARAMETERS  (must match toy dataset: 172 × 79)
# ─────────────────────────────────────────────────────────────────────────────
NGRID_Y = 172          # stream-wise resolution (rows)
NGRID_X = 79           # lateral resolution (columns)
LAT_MARGIN   = 0.60    # chord-lengths left/right of airfoil
UP_MARGIN    = 0.60    # chord-lengths upstream of leading edge
DOWN_MARGIN  = 0.80    # chord-lengths downstream of trailing edge
SDF_CLIP     = -50.0   # clip interior SDF at this value (matching toy dataset)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: LOAD AIRFOIL COORDINATES
# ─────────────────────────────────────────────────────────────────────────────

def load_airfoil_dat(dat_file):
    """
    Robustly load a Selig-format .dat file.
    Skips header lines that cannot be parsed as floats.
    Returns (XB, YB) arrays with chord in range ~[0,1].
    """
    pts = []
    with open(dat_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                try:
                    pts.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    pass  # header line or comment
    if len(pts) < 5:
        raise ValueError(f"Could not parse enough coordinate pairs from {dat_file}")
    arr = np.array(pts)
    return arr[:, 0], arr[:, 1]


def ensure_ccw(XB, YB):
    """Ensure CCW panel ordering (required by Hess-Smith formulation)."""
    edge = np.sum([(XB[i+1]-XB[i]) * (YB[i+1]+YB[i]) for i in range(len(XB)-1)])
    if edge < 0:
        return XB[::-1], YB[::-1]
    return XB, YB


def close_airfoil(XB, YB):
    """Close the airfoil contour (add first point at end if not already closed)."""
    if not (np.isclose(XB[0], XB[-1]) and np.isclose(YB[0], YB[-1])):
        XB = np.append(XB, XB[0])
        YB = np.append(YB, YB[0])
    return XB, YB


def refine_airfoil(XB, YB, n_points=200):
    """
    Refine a coarse airfoil contour to `n_points` via cubic spline interpolation.
    This is critical for .dat files with fewer than ~100 points — without this
    the panel method gives poor results and the SDF / mask are jagged.

    The contour must already be closed before calling this function.
    """
    if len(XB) >= n_points:
        return XB, YB   # already fine enough

    # Parameterise by cumulative arc-length
    dx = np.diff(XB)
    dy = np.diff(YB)
    ds = np.hypot(dx, dy)
    s  = np.concatenate([[0.0], np.cumsum(ds)])
    s /= s[-1]   # normalise 0→1

    # Periodic cubic spline (airfoil is a closed loop)
    try:
        cs_x = CubicSpline(s, XB, bc_type='periodic')
        cs_y = CubicSpline(s, YB, bc_type='periodic')
    except ValueError:
        # Fallback if periodicity assumption fails (e.g. last ≠ first)
        cs_x = CubicSpline(s, XB)
        cs_y = CubicSpline(s, YB)

    s_fine = np.linspace(0.0, 1.0, n_points, endpoint=False)
    XB_ref = cs_x(s_fine)
    YB_ref = cs_y(s_fine)

    # Re-close the contour
    XB_ref = np.append(XB_ref, XB_ref[0])
    YB_ref = np.append(YB_ref, YB_ref[0])
    return XB_ref, YB_ref


# ─────────────────────────────────────────────────────────────────────────────
# PANEL METHOD CORE  (Hess-Smith Source + Vortex)
# ─────────────────────────────────────────────────────────────────────────────

def panel_geometry(XB, YB):
    """Compute panel geometry: centroids (XC,YC), lengths S, angles phi."""
    XC  = 0.5 * (XB[:-1] + XB[1:])
    YC  = 0.5 * (YB[:-1] + YB[1:])
    dX  = XB[1:] - XB[:-1]
    dY  = YB[1:] - YB[:-1]
    S   = np.hypot(dX, dY)
    phi = np.arctan2(dY, dX)
    return XC, YC, S, phi


def _log_atan_integrals(dx_vec, dy_vec, phi_j, S_j):
    """
    Compute the two fundamental integrals (log-term t1 and arctan-term t2)
    used in both source and vortex influence calculations.
    """
    A = -dx_vec * np.cos(phi_j) - dy_vec * np.sin(phi_j)
    B =  dx_vec**2 + dy_vec**2
    E =  np.sqrt(np.maximum(B - A**2, 0.0))

    t1 = np.zeros_like(B)
    m  = B > 1e-14
    t1[m] = 0.5 * np.log((S_j**2 + 2*A[m]*S_j + B[m]) / B[m])

    t2 = np.zeros_like(E)
    me = E > 1e-8
    t2[me] = (np.arctan2(S_j + A[me], E[me]) - np.arctan2(A[me], E[me])) / E[me]

    return A, B, E, t1, t2


def compute_influence_matrices(XC, YC, XB, YB, phi, S):
    """
    Build normal-velocity (I,K) and tangential-velocity (J,L) influence
    coefficient matrices for the Hess-Smith SVPM.

    Returns I, J, K, L each of shape (N, N).
    """
    N = len(XC)
    I = np.zeros((N, N));  J = np.zeros((N, N))
    K = np.zeros((N, N));  L = np.zeros((N, N))

    for j in range(N):
        dx = XC - XB[j]
        dy = YC - YB[j]
        A, B, E, t1, t2 = _log_atan_integrals(dx, dy, phi[j], S[j])

        # Source influence (normal = Cn*t1 + (Dn-A*Cn)*t2)
        Cn =  np.sin(phi - phi[j])
        Dn = -dx * np.sin(phi) + dy * np.cos(phi)
        Ct = -np.cos(phi - phi[j])
        Dt =  dx * np.cos(phi) + dy * np.sin(phi)

        # Vortex influence (rotated coefficients)
        Cnv = -np.cos(phi - phi[j])
        Dnv =  dx * np.cos(phi) + dy * np.sin(phi)
        Ctv =  np.sin(phi[j] - phi)
        Dtv =  dx * np.sin(phi) - dy * np.cos(phi)

        I[:, j] = Cn  * t1 + (Dn  - A*Cn)  * t2
        J[:, j] = Ct  * t1 + (Dt  - A*Ct)  * t2
        K[:, j] = Cnv * t1 + (Dnv - A*Cnv) * t2
        L[:, j] = Ctv * t1 + (Dtv - A*Ctv) * t2

    # Diagonal self-influence (analytical limit = π)
    np.fill_diagonal(I, np.pi)
    np.fill_diagonal(L, np.pi)

    return I, J, K, L


def solve_panel_strengths(XB, YB, phi, S, Vinf, alpha_deg):
    """
    Solve the (N+1)×(N+1) linear system for source strengths lam[N] and
    vortex strength gamma (scalar Kutta condition).

    Returns lam (N,) and gamma (scalar).
    """
    N     = len(phi)
    alpha = np.deg2rad(alpha_deg)

    # Normal and tangential freestream velocity components at each panel centroid
    Vn = Vinf * np.sin(phi - alpha)   # normal component (should be zeroed by BCs)
    Vt = Vinf * np.cos(phi - alpha)   # tangential component

    XC  = 0.5 * (XB[:-1] + XB[1:])
    YC  = 0.5 * (YB[:-1] + YB[1:])
    I, J, K, L = compute_influence_matrices(XC, YC, XB, YB, phi, S)

    A_sys = np.zeros((N+1, N+1))
    A_sys[:N, :N] = I
    A_sys[:N,  N] = -np.sum(K, axis=1)
    A_sys[ N, :N] =  J[0, :] + J[N-1, :]
    A_sys[ N,  N] = -np.sum(L[0, :] + L[N-1, :]) + 2*np.pi

    # RHS: normal-flow BC + Kutta condition
    rhs          = np.zeros(N+1)
    rhs[:N]      = -Vn * 2*np.pi
    rhs[ N]      = -(Vt[0] + Vt[N-1]) * 2*np.pi   # Kutta: tangential at both TEs

    sol   = np.linalg.solve(A_sys, rhs)
    lam   = sol[:-1]
    gamma = sol[-1]
    return lam, gamma


# ─────────────────────────────────────────────────────────────────────────────
# VELOCITY FIELD EVALUATION ON GRID
# ─────────────────────────────────────────────────────────────────────────────

def compute_velocity_field(XP, YP, XB, YB, phi, S, lam, gamma, Vinf, alpha_deg):
    """
    Evaluate (u, v) velocity at a set of grid points in the AIRFOIL frame
    (freestream = Vinf*(cos α, sin α)).

    XP, YP : 2-D arrays of query points (airfoil frame)
    Returns u, v arrays of same shape.
    """
    alpha  = np.deg2rad(alpha_deg)
    shape  = XP.shape
    XPf    = XP.flatten()
    YPf    = YP.flatten()
    N      = len(S)

    u_panel = np.zeros(len(XPf))
    v_panel = np.zeros(len(XPf))

    for j in range(N):
        dx = XPf - XB[j]
        dy = YPf - YB[j]
        A, B, E, t1, t2 = _log_atan_integrals(dx, dy, phi[j], S[j])

        # Source velocity kernel
        Sx = -np.cos(phi[j]) * t1 + (dx - A*(-np.cos(phi[j]))) * t2
        Sy = -np.sin(phi[j]) * t1 + (dy - A*(-np.sin(phi[j]))) * t2

        # Vortex velocity kernel (perpendicular to source)
        Vx_k =  np.sin(phi[j]) * t1 + (-dy - A*np.sin(phi[j])) * t2
        Vy_k = -np.cos(phi[j]) * t1 + ( dx - A*(-np.cos(phi[j]))) * t2

        u_panel += (lam[j] * Sx - gamma * Vx_k) / (2*np.pi)
        v_panel += (lam[j] * Sy - gamma * Vy_k) / (2*np.pi)

    u = Vinf * np.cos(alpha) + u_panel
    v = Vinf * np.sin(alpha) + v_panel
    return u.reshape(shape), v.reshape(shape)


# ─────────────────────────────────────────────────────────────────────────────
# HEURISTIC VISCOUS WAKE
# ─────────────────────────────────────────────────────────────────────────────

def apply_wake(Uy_grid, Ux_grid, XB_r, YB_r, XX, YY, Vinf, wake_deficit=0.10):
    """
    Add a Gaussian wake deficit downstream of the trailing edge in grid frame.

    Grid frame convention: +Y is stream-wise (flow direction), +X is lateral.
    Trailing edge = point on XB_r, YB_r with maximum YB_r.

    The deficit reduces the stream-wise velocity (Uy) by a Gaussian
    bell that widens and decays moving downstream.
    """
    te_idx = np.argmax(YB_r)
    te_x   = XB_r[te_idx]
    te_y   = YB_r[te_idx]

    dist_down   = YY - te_y
    downstream  = dist_down > 0.0

    # Wake half-width grows like sqrt(distance) – turbulent-spreading approximation
    half_w = 0.04 + 0.10 * np.sqrt(np.maximum(dist_down, 0.0))

    # Gaussian profile (centred on TE x-position) × exponential decay
    deficit_field = (
        wake_deficit
        * np.exp(-0.5 * ((XX - te_x) / half_w)**2)
        * np.exp(-1.0 * np.maximum(dist_down, 0.0))
    )

    Uy_out = Uy_grid.copy()
    Uy_out[downstream] *= (1.0 - deficit_field[downstream])
    return Uy_out, Ux_grid


# ─────────────────────────────────────────────────────────────────────────────
# CHANNEL BUILDERS  (matching toy dataset structure)
# ─────────────────────────────────────────────────────────────────────────────

def build_sdf_channel(XB_r, YB_r, pts, inside_mask, shape, clip=SDF_CLIP):
    """
    Signed Distance Function channel.
    Positive outside the airfoil, negative inside (clipped at `clip`).
    """
    tree    = cKDTree(np.column_stack((XB_r, YB_r)))
    dist, _ = tree.query(pts)
    sdf     = dist.reshape(shape).astype(np.float32)
    sdf[inside_mask] = np.maximum(clip, -sdf[inside_mask])
    return sdf


def build_bc_channel(shape, inside_mask):
    """
    Boundary-condition marker channel (matches toy dataset Ch1):
      0 = obstacle interior
      1 = fluid domain
      2 = left/right lateral walls (inlet sides)
      3 = top boundary (upstream inlet)
      4 = bottom boundary (downstream outlet)
    """
    nY, nX = shape
    bc = np.ones(shape, dtype=np.float32)          # default: fluid

    # Mark obstacle interior
    bc[inside_mask] = 0.0

    # Mark boundaries (1-pixel wide)
    bc[:, 0]   = 2.0    # left wall
    bc[:, -1]  = 2.0    # right wall
    bc[-1, :]  = 3.0    # top  (upstream, flow enters from top in grid frame)
    bc[0,  :]  = 4.0    # bottom (downstream outlet)

    # Corners: already covered by the last two lines
    return bc


def build_vinf_channel(shape, Vinf_peak):
    """
    Triangular inlet-velocity profile matching the toy dataset Ch2.

    The toy dataset has a symmetric triangular profile across the
    column axis (x-axis of the grid), rising from ~-0.005*Vinf at the
    edges to Vinf_peak at the centre column.

    This exactly reproduces the 40-value triangular shape observed in
    the reference dataset (Vinf_peak=0.385, edges=-0.005).
    """
    nY, nX = shape
    # Symmetrically varying across columns, constant along rows
    cols   = np.arange(nX)
    half   = (nX - 1) / 2.0
    # Triangle: goes from -eps at col=0, peaks at centre, back to -eps at col=nX-1
    eps    = 0.005 * Vinf_peak / 0.385   # scale edge offset with Vinf
    tri    = (1.0 - np.abs(cols - half) / half) * (Vinf_peak + eps) - eps
    # Tile to 2D (same profile for all rows)
    vinf_ch = np.tile(tri, (nY, 1)).astype(np.float32)
    return vinf_ch


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DATASET CREATOR
# ─────────────────────────────────────────────────────────────────────────────

def create_dataset(
    dat_file,
    out_dir=".",
    alpha_deg=0.0,
    Vinf_peak=0.1,
    nGridY=NGRID_Y,
    nGridX=NGRID_X,
):
    """
    Full pipeline: .dat file → dataX.pkl + dataY.pkl

    Parameters
    ----------
    dat_file  : path to Selig-format airfoil coordinate file
    out_dir   : directory to save dataX.pkl and dataY.pkl
    alpha_deg : angle of attack in degrees
    Vinf_peak : peak freestream velocity (used for Bernoulli pressure)
    nGridY    : number of grid rows    (stream-wise)
    nGridX    : number of grid columns (lateral)
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"[1/8] Loading airfoil: {dat_file}")

    # ── 1. Load & condition airfoil geometry ─────────────────────────────────
    XB, YB = load_airfoil_dat(dat_file)

    # Normalise chord to [0,1] if not already
    chord = XB.max() - XB.min()
    if chord > 1.1 or chord < 0.5:
        print(f"  Warning: chord length = {chord:.3f}, normalising to unit chord.")
        XB = (XB - XB.min()) / chord
        YB = YB / chord   # same scale factor

    XB, YB = close_airfoil(XB, YB)
    XB, YB = ensure_ccw(XB, YB)
    print(f"      Raw coordinates: {len(XB)-1} panels")
    XB, YB = refine_airfoil(XB, YB, n_points=200)
    XB, YB = ensure_ccw(XB, YB)
    print(f"      Refined to:      {len(XB)-1} panels")

    # ── 2. Compute panel strengths ───────────────────────────────────────────
    print("[2/8] Computing panel geometry & strengths …")
    XC, YC, S, phi = panel_geometry(XB, YB)
    lam, gamma     = solve_panel_strengths(XB, YB, phi, S, Vinf_peak, alpha_deg)
    print(f"      lam: [{lam.min():.4f}, {lam.max():.4f}]  gamma={gamma:.4f}")

    # ── 3. Build grid in ROTATED frame ───────────────────────────────────────
    # Rotation 90° CCW: x_rot = -y_orig,  y_rot = x_orig
    # Airfoil chord now lies along Y-axis; freestream flows in +Y direction
    print("[3/8] Building computational grid (rotated 90° CCW) …")
    XB_r, YB_r = -YB, XB

    x_min = XB_r.min() - LAT_MARGIN
    x_max = XB_r.max() + LAT_MARGIN
    y_min = YB_r.min() - UP_MARGIN
    y_max = YB_r.max() + DOWN_MARGIN

    xs = np.linspace(x_min, x_max, nGridX)
    ys = np.linspace(y_min, y_max, nGridY)
    XX, YY = np.meshgrid(xs, ys)    # shape (nGridY, nGridX)

    # ── 4. Evaluate velocity field ───────────────────────────────────────────
    # Grid query points in rotated frame → transform to airfoil frame:
    #   x_af =  y_rot,   y_af = -x_rot
    print("[4/8] Evaluating velocity field on grid (this may take a moment) …")
    u_af, v_af = compute_velocity_field(
        YY, -XX,                               # airfoil-frame query coords
        XB, YB, phi, S, lam, gamma, Vinf_peak, alpha_deg
    )
    # Convert back to grid/rotated frame:
    #   Uy (stream-wise, +Y)  =  u_af  (airfoil +x → grid +Y)
    #   Ux (cross-stream, +X) = -v_af  (airfoil +y → grid −X)
    Uy = u_af.astype(np.float32)
    Ux = (-v_af).astype(np.float32)

    # ── 5. Inside-obstacle mask ──────────────────────────────────────────────
    print("[5/8] Computing obstacle mask …")
    af_path = Path(np.column_stack((XB_r, YB_r)))
    pts     = np.column_stack((XX.flatten(), YY.flatten()))
    inside  = af_path.contains_points(pts).reshape(nGridY, nGridX)

    # ── 6. Viscous wake correction ───────────────────────────────────────────
    print("[6/8] Applying heuristic wake model …")
    Uy, Ux = apply_wake(Uy, Ux, XB_r, YB_r, XX, YY, Vinf_peak, wake_deficit=0.10)

    # Zero-out obstacle interior
    Uy[inside] = 0.0
    Ux[inside] = 0.0

    # ── 7. Pressure via Bernoulli ────────────────────────────────────────────
    # p = 0.5 * rho * (Vinf² − |V|²),  rho=1, p_inf=0
    print("[7/8] Computing Bernoulli pressure …")
    V2 = Ux**2 + Uy**2
    P  = (0.5 * (Vinf_peak**2 - V2)).astype(np.float32)
    P[inside] = 0.0

    # ── 8. Assemble dataX and dataY ──────────────────────────────────────────
    print("[8/8] Building dataX and dataY channels …")
    shape = (nGridY, nGridX)

    sdf_ch  = build_sdf_channel(XB_r, YB_r, pts, inside, shape)
    bc_ch   = build_bc_channel(shape, inside)
    vinf_ch = build_vinf_channel(shape, Vinf_peak)

    dataX = np.expand_dims(
        np.stack([sdf_ch, bc_ch, vinf_ch], axis=0), axis=0
    ).astype(np.float32)   # (1, 3, nGridY, nGridX)

    dataY = np.expand_dims(
        np.stack([Uy, Ux, P], axis=0), axis=0
    ).astype(np.float32)   # (1, 3, nGridY, nGridX)

    # ── Save ─────────────────────────────────────────────────────────────────
    x_path = os.path.join(out_dir, "dataX.pkl")
    y_path = os.path.join(out_dir, "dataY.pkl")
    with open(x_path, 'wb') as f: pickle.dump(dataX, f)
    with open(y_path, 'wb') as f: pickle.dump(dataY, f)

    print("\n── Dataset saved ──────────────────────────────────────────")
    print(f"  dataX: {dataX.shape}  →  {x_path}")
    print(f"  dataY: {dataY.shape}  →  {y_path}")
    print("\n── Channel statistics ─────────────────────────────────────")
    labels_x = ["SDF1 (obstacle)", "BC markers", "Vinf profile"]
    labels_y = ["Uy (stream-wise)", "Ux (cross-stream)", "P (pressure)"]
    for i, lbl in enumerate(labels_x):
        c = dataX[0, i]
        print(f"  dataX Ch{i} [{lbl}]: min={c.min():.4f}  max={c.max():.4f}  mean={c.mean():.4f}")
    for i, lbl in enumerate(labels_y):
        c = dataY[0, i]
        print(f"  dataY Ch{i} [{lbl}]: min={c.min():.4f}  max={c.max():.4f}  mean={c.mean():.4f}")
    print()

    return dataX, dataY


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate one DeepCFD-format sample from an airfoil .dat file."
    )
    ap.add_argument("--dat_file",  required=True,     help="Path to Selig-format .dat file")
    ap.add_argument("--out_dir",   default=".",        help="Output directory (default: current dir)")
    ap.add_argument("--alpha",     type=float, default=0.0,  help="Angle of attack in degrees")
    ap.add_argument("--Vinf",      type=float, default=0.1,  help="Peak inlet velocity (default: 0.1)")
    ap.add_argument("--nGridY",    type=int,   default=NGRID_Y)
    ap.add_argument("--nGridX",    type=int,   default=NGRID_X)
    args = ap.parse_args()

    create_dataset(
        dat_file  = args.dat_file,
        out_dir   = args.out_dir,
        alpha_deg = args.alpha,
        Vinf_peak = args.Vinf,
        nGridY    = args.nGridY,
        nGridX    = args.nGridX,
    )
