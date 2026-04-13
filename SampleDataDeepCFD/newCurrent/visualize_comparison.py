"""
visualize_comparison.py  –  newCurrent version
================================================
Visualize a generated (dataX, dataY) sample SIDE-BY-SIDE with the toy
dataset reference sample, to validate correctness of the generation pipeline.

Usage:
  # Compare generated files vs toy dataset
  python visualize_comparison.py \\
      --gen_x dataX.pkl --gen_y dataY.pkl \\
      --ref_x ../dataX.pkl --ref_y ../dataY.pkl \\
      --index 0 --save comparison.png

  # Just visualize a single generated sample (no reference)
  python visualize_comparison.py --gen_x dataX.pkl --gen_y dataY.pkl
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def channel_stats(arr):
    return f"min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-SAMPLE PANEL
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_panels(axs_row, x, y, title_prefix="", Vinf_override=None):
    """
    Fill a row of 6 axes:  SDF | BC/Inlet | Vinf | Uy | Ux | P
    axs_row : list/array of 6 Axes objects
    """
    sdf1  = x[0]
    bc    = x[1]
    vinf  = x[2]
    Uy    = y[0]
    Ux    = y[1]
    P     = y[2]

    # Estimate Vinf for clamp limits (peak of vinf channel)
    Vref  = Vinf_override if Vinf_override else np.abs(vinf).max()
    Vhalf = Vref * 0.5

    panels = [
        (sdf1, 'SDF1 (obstacle)',     'seismic',  None,      None),
        (bc,   'BC markers',           'tab10',    None,      None),
        (vinf, 'Inlet profile (Vinf)', 'viridis',  None,      None),
        (Uy,   'Uy – stream-wise',     'jet',      Vref-Vhalf, Vref+Vhalf),
        (Ux,   'Ux – cross-stream',   'jet',      -Vhalf,    Vhalf),
        (P,    'P – pressure',         'RdBu_r',   None,      None),
    ]

    for ax, (data, title, cmap, vmin, vmax) in zip(axs_row, panels):
        kwargs = dict(origin='lower', cmap=cmap)
        if vmin is not None: kwargs['vmin'] = vmin
        if vmax is not None: kwargs['vmax'] = vmax
        im = ax.imshow(data, **kwargs)
        ax.set_title(f"{title_prefix}{title}", fontsize=8)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def visualize_comparison(gen_x, gen_y, gen_idx=0,
                          ref_x=None, ref_y=None, ref_idx=0,
                          save_path=None):
    """
    Main function: create a rich side-by-side comparison figure.
    """
    x_gen = gen_x[gen_idx]    # (3, H, W)
    y_gen = gen_y[gen_idx]    # (3, H, W)

    has_ref = (ref_x is not None) and (ref_y is not None)
    n_rows  = 2 if has_ref else 1

    fig = plt.figure(figsize=(20, 5*n_rows + 1))
    fig.patch.set_facecolor('#0e1117')

    outer = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.45)

    def make_row(outer_cell, x, y, label, Vinf_override=None):
        inner = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer_cell, wspace=0.35)
        axs   = [fig.add_subplot(inner[0, c]) for c in range(6)]
        for ax in axs:
            ax.set_facecolor('#1a1d27')
        plot_sample_panels(axs, x, y, title_prefix="", Vinf_override=Vinf_override)
        # Row label
        fig.text(0.01, 0.5 if n_rows == 1 else (0.75 if label.startswith("Gen") else 0.25),
                 label, va='center', ha='left', rotation='vertical',
                 fontsize=11, color='white', fontweight='bold')

    make_row(outer[0], x_gen, y_gen, "Generated\n(Panel Method)")

    if has_ref:
        x_ref = ref_x[ref_idx]
        y_ref = ref_y[ref_idx]
        make_row(outer[1], x_ref, y_ref, "Reference\n(Toy Dataset)")

    # Print stats
    print("\n── Generated sample statistics ────────────────────────────────")
    chan_lbl_x = ["SDF1",       "BC markers", "Vinf profile"]
    chan_lbl_y = ["Uy (stream)","Ux (cross)", "P (pressure)"]
    for i, lbl in enumerate(chan_lbl_x):
        print(f"  dataX Ch{i} [{lbl}]:  {channel_stats(x_gen[i])}")
    for i, lbl in enumerate(chan_lbl_y):
        print(f"  dataY Ch{i} [{lbl}]:  {channel_stats(y_gen[i])}")

    if has_ref:
        print("\n── Reference toy-dataset statistics ───────────────────────────")
        for i, lbl in enumerate(chan_lbl_x):
            print(f"  dataX Ch{i} [{lbl}]:  {channel_stats(x_ref[i])}")
        for i, lbl in enumerate(chan_lbl_y):
            print(f"  dataY Ch{i} [{lbl}]:  {channel_stats(y_ref[i])}")

    fig.suptitle(
        "DeepCFD Data Visualisation – Generated vs Reference",
        fontsize=14, color='white', fontweight='bold', y=1.01
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"\n  Figure saved: {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-FILE QUICK VISUALISER  (6-panel layout)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_single(x, y, index=0, save_path=None, title=None):
    """
    Clean 6-panel visualisation for one sample (no reference).
    Matches the aesthetic of the toy-dataset visualiser.
    """
    sx  = x[index]     # (3,H,W)
    sy  = y[index]     # (3,H,W)
    Vref = np.abs(sx[2]).max()
    Vhalf = Vref * 0.5

    chan_x = [
        (sx[0], "SDF1 – obstacle distance", 'seismic',  None,        None),
        (sx[1], "BC markers",               'tab10',    None,        None),
        (sx[2], "Vinf – inlet profile",     'viridis',  None,        None),
    ]
    chan_y = [
        (sy[0], "Uy – stream-wise vel.",    'jet',   Vref-Vhalf,  Vref+Vhalf),
        (sy[1], "Ux – cross-stream vel.",   'jet',   -Vhalf,      Vhalf),
        (sy[2], "P – pressure (Bernoulli)", 'RdBu_r', None,       None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor('#0e1117')
    for ax in axes.flat:
        ax.set_facecolor('#1a1d27')

    for ax, (data, ttl, cmap, vmin, vmax) in zip(axes[0], chan_x):
        kw = dict(origin='lower', cmap=cmap)
        if vmin is not None: kw.update(vmin=vmin, vmax=vmax)
        im = ax.imshow(data, **kw)
        ax.set_title(ttl, color='white', fontsize=9)
        ax.axis('off')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

    for ax, (data, ttl, cmap, vmin, vmax) in zip(axes[1], chan_y):
        kw = dict(origin='lower', cmap=cmap)
        if vmin is not None: kw.update(vmin=vmin, vmax=vmax)
        im = ax.imshow(data, **kw)
        ax.set_title(ttl, color='white', fontsize=9)
        ax.axis('off')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

    suptitle = title or f"DeepCFD Sample – Index {index}"
    fig.suptitle(suptitle, fontsize=13, color='white', fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Figure saved: {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Visualise generated DeepCFD data with optional toy-dataset comparison."
    )
    ap.add_argument("--gen_x",  required=True, help="Generated dataX.pkl")
    ap.add_argument("--gen_y",  required=True, help="Generated dataY.pkl")
    ap.add_argument("--gen_idx", type=int, default=0)
    ap.add_argument("--ref_x",  default=None,  help="Reference (toy) dataX.pkl")
    ap.add_argument("--ref_y",  default=None,  help="Reference (toy) dataY.pkl")
    ap.add_argument("--ref_idx", type=int, default=0)
    ap.add_argument("--save",   default=None,  help="Save figure to this path instead of showing")
    ap.add_argument("--single", action="store_true",
                    help="Use the clean 6-panel single-sample layout (no side-by-side)")
    args = ap.parse_args()

    gen_x = load_pkl(args.gen_x)
    gen_y = load_pkl(args.gen_y)
    print(f"Generated  dataX: {gen_x.shape}  dataY: {gen_y.shape}")

    ref_x, ref_y = None, None
    if args.ref_x and args.ref_y:
        ref_x = load_pkl(args.ref_x)
        ref_y = load_pkl(args.ref_y)
        print(f"Reference  dataX: {ref_x.shape}  dataY: {ref_y.shape}")

    if args.single or (ref_x is None):
        visualize_single(gen_x, gen_y, index=args.gen_idx, save_path=args.save)
    else:
        visualize_comparison(
            gen_x, gen_y, gen_idx=args.gen_idx,
            ref_x=ref_x, ref_y=ref_y, ref_idx=args.ref_idx,
            save_path=args.save,
        )
