"""
visualize_physics.py  –  newCurrent version
=============================================
Physics-focused visualization: streamlines, pressure contours, and
velocity maps for the generated airfoil dataset.

Usage:
  python visualize_physics.py --dataX dataX.pkl --dataY dataY.pkl --save physics.png
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import warnings

warnings.filterwarnings("ignore")


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def visualize_physics(dataX, dataY, index=0, save_path=None):
    """
    Create a 3-panel physics visualization:
      Left  : Uy (stream-wise velocity) with streamlines
      Centre : Ux (cross-stream velocity) contour
      Right  : P (pressure) with high-pressure stagnation + low-pressure suction
    """
    x  = dataX[index]   # (3, H, W)
    y  = dataY[index]   # (3, H, W)

    sdf   = x[0]        # SDF1
    vinf  = x[2]        # inlet profile
    Uy    = y[0]        # stream-wise (primary flow direction)
    Ux    = y[1]        # cross-stream
    P     = y[2]        # pressure

    H, W  = Uy.shape
    xs    = np.linspace(-1, 1, W)
    ys    = np.linspace(-1, 1, H)
    XX, YY = np.meshgrid(xs, ys)

    # Obstacle mask
    inside = sdf < 0

    # ─── overall figure ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor('#0e1117')
    for ax in axes:
        ax.set_facecolor('#1a1d27')

    Vref  = float(np.abs(vinf).max())
    Vhalf = Vref * 0.5

    # ── Panel 1: Uy (stream-wise) + streamlines ──────────────────
    ax = axes[0]
    im1 = ax.imshow(Uy, origin='lower', cmap='jet',
                    vmin=Vref - Vhalf, vmax=Vref + Vhalf,
                    extent=[xs[0], xs[-1], ys[0], ys[-1]])
    # Streamlines (only in fluid region)
    Uy_sl = Uy.copy(); Uy_sl[inside] = np.nan
    Ux_sl = Ux.copy(); Ux_sl[inside] = np.nan
    try:
        ax.streamplot(XX, YY, Ux_sl, Uy_sl,
                      color='white', linewidth=0.6, density=1.2, arrowsize=0.8)
    except Exception:
        pass   # streamplot can fail with NaN grids in some matplotlib versions
    # Overlay obstacle
    ax.contourf(XX, YY, sdf.astype(float), levels=[-100, 0], colors=['#000000'], alpha=0.9)
    cb = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Uy (m/s)', color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
    ax.set_title('Stream-wise Velocity Uy\n+ Streamlines', color='white', fontsize=12, pad=10)
    ax.set_xlabel('x / chord', color='white')
    ax.set_ylabel('y / chord (flow→)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#444')

    # ── Panel 2: Ux (cross-stream) ───────────────────────────────
    ax = axes[1]
    ext_ux = max(abs(Ux.min()), abs(Ux.max()))
    im2 = ax.imshow(Ux, origin='lower', cmap='RdBu_r',
                    vmin=-ext_ux, vmax=ext_ux,
                    extent=[xs[0], xs[-1], ys[0], ys[-1]])
    ax.contourf(XX, YY, sdf.astype(float), levels=[-100, 0], colors=['#000000'], alpha=0.9)
    cb = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Ux (m/s)', color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
    ax.set_title('Cross-stream Velocity Ux\n(±symmetric around airfoil)', color='white', fontsize=12, pad=10)
    ax.set_xlabel('x / chord', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#444')

    # ── Panel 3: Pressure P ──────────────────────────────────────
    ax = axes[2]
    ext_p = max(abs(P.min()), abs(P.max()))
    im3 = ax.imshow(P, origin='lower', cmap='RdBu_r',
                    vmin=-ext_p, vmax=ext_p,
                    extent=[xs[0], xs[-1], ys[0], ys[-1]])
    ax.contourf(XX, YY, sdf.astype(float), levels=[-100, 0], colors=['#000000'], alpha=0.9)
    cb = plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('P (Pa, ρ=1)', color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
    ax.set_title('Static Pressure P = 0.5(Vinf²−|V|²)\n(red=high, blue=low)', color='white', fontsize=12, pad=10)
    ax.set_xlabel('x / chord', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#444')

    # ─── stats text ──────────────────────────────────────────────
    Vinf_peak = float(vinf.max())
    stats_str = (
        f"Vinf_peak={Vinf_peak:.3f} m/s\n"
        f"Uy: [{Uy.min():.4f}, {Uy.max():.4f}]\n"
        f"Ux: [{Ux.min():.4f}, {Ux.max():.4f}]\n"
        f"P:  [{P.min():.4f}, {P.max():.4f}]"
    )
    fig.text(0.5, -0.02, stats_str, ha='center', va='top',
             color='#aaaaaa', fontsize=9, fontfamily='monospace')

    fig.suptitle(
        "DeepCFD – Airfoil Flow Field (Panel Method)\n"
        "Flow direction: ↑  |  Grid: 172×79  |  Method: Hess-Smith SVPM + Heuristic Wake",
        fontsize=11, color='white', fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Physics figure saved: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Physics-focused visualization of generated DeepCFD data.")
    ap.add_argument("--dataX",  default="dataX.pkl", help="Path to dataX.pkl")
    ap.add_argument("--dataY",  default="dataY.pkl", help="Path to dataY.pkl")
    ap.add_argument("--index",  type=int, default=0)
    ap.add_argument("--save",   default=None,        help="Save figure to this path")
    args = ap.parse_args()

    x = load_pkl(args.dataX)
    y = load_pkl(args.dataY)
    print(f"Loaded dataX: {x.shape}  dataY: {y.shape}")
    visualize_physics(x, y, index=args.index, save_path=args.save)
