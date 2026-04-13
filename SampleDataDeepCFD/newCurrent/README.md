# newCurrent — DeepCFD Airfoil Data Generation Pipeline

## Overview

This folder contains scripts to generate **DeepCFD-compatible `dataX.pkl` and `dataY.pkl`** from a single airfoil `.dat` file, using the **Hess-Smith Source+Vortex Panel Method (SVPM)** for potential flow.

---

## Method Used in the DeepCFD Paper

The paper ([arXiv:2004.08826](https://arxiv.org/abs/2004.08826)) used **OpenFOAM** (`simpleFoam`, steady laminar Navier-Stokes solver) to generate ground-truth velocity (Ux, Uy) and pressure (P) fields for **2D arbitrary obstacle shapes** (bluff bodies) at various Reynolds numbers.

**Panel method vs OpenFOAM:**
| | Panel Method (this pipeline) | OpenFOAM (paper) |
|---|---|---|
| Flow type | Potential (inviscid, irrotational) | Viscous laminar (N-S) |
| Obstacle type | Airfoil only | Any 2D shape |
| Accuracy | Good away from surface | Full CFD accuracy |
| Speed | **Seconds** | Minutes to hours |
| Wake | Heuristic (Gaussian deficit) | Physical viscous wake |
| Pressure | Bernoulli: `0.5*(Vinf²-|V|²)` | Full N-S pressure |

**Is panel method sufficient?** For training/demonstration purposes with airfoils at low AoA (potential-flow regime), yes. The topology and structure of Ux, Uy, P fields match the physical flow qualitatively. For high AoA or separated flow, OpenFOAM would be needed.

---

## Dataset Format (matches toy dataset exactly)

```
dataX shape: (1, 3, 172, 79)   # (N_samples, channels, height, width)
dataY shape: (1, 3, 172, 79)
```

### dataX channels:
| Ch | Name | Description |
|----|------|-------------|
| 0 | SDF1 | Signed distance function to airfoil surface (negative inside) |
| 1 | BC   | Boundary-condition markers: 0=solid, 1=fluid, 2=walls, 3=top, 4=bottom |
| 2 | Vinf | Triangular inlet velocity profile across channel width |

### dataY channels:
| Ch | Name | Description |
|----|------|-------------|
| 0 | Uy   | Stream-wise velocity (flow goes in +Y / vertical direction) |
| 1 | Ux   | Cross-stream velocity (horizontal perturbations) |
| 2 | P    | Static pressure = 0.5*(Vinf²−|V|²)  (Bernoulli, ρ=1) |

### Grid conventions:
- Airfoil chord is rotated **90° CCW** → chord lies along the Y-axis
- Freestream flows **upward** (+Y direction)
- Domain: ±0.6 chord lateral, 0.6 chord upstream, 0.8 chord downstream

---

## Files

| File | Purpose |
|------|---------|
| `generate_data.py` | Main generator: `.dat` → `dataX.pkl` + `dataY.pkl` |
| `visualize_comparison.py` | Side-by-side comparison vs toy dataset reference |
| `visualize_physics.py` | Physics visualization: streamlines, pressure, velocity |

---

## Usage

### 1. Generate data from a `.dat` file

```bash
cd SampleDataDeepCFD/newCurrent

python generate_data.py \
    --dat_file "../../PanelMethodCode/naca0015.dat" \
    --out_dir "." \
    --Vinf 0.1 \
    --alpha 0.0
```

**Arguments:**
- `--dat_file` : Path to Selig-format airfoil coordinate file
- `--out_dir`  : Output directory (default: current dir)
- `--Vinf`     : Peak freestream velocity (default: 0.1)
- `--alpha`    : Angle of attack in degrees (default: 0.0)
- `--nGridY`   : Grid rows (default: 172, matches toy dataset)
- `--nGridX`   : Grid columns (default: 79, matches toy dataset)

### 2. Visualize the generated data (physics view)

```bash
python visualize_physics.py --dataX dataX.pkl --dataY dataY.pkl --save physics.png
```

### 3. Compare against toy dataset

```bash
python visualize_comparison.py \
    --gen_x dataX.pkl --gen_y dataY.pkl \
    --ref_x ../dataX.pkl --ref_y ../dataY.pkl \
    --save comparison.png
```

### 4. Quick single-sample overview

```bash
python visualize_comparison.py --gen_x dataX.pkl --gen_y dataY.pkl --single --save single_view.png
```

---

## Expected Output Statistics (NACA0015, Vinf=0.1, AoA=0°)

```
dataX Ch0 [SDF1]:     min≈-0.09  max≈1.05   (negative inside airfoil)
dataX Ch1 [BC]:       min=0.0    max=4.0    (boundary markers)
dataX Ch2 [Vinf]:     min≈-0.001 max=0.1    (triangular profile)

dataY Ch0 [Uy]:       min≈0.0    max≈0.18   (stream-wise, freestream≈0.1)
dataY Ch1 [Ux]:       min≈-0.05  max≈0.06   (cross-stream, antisymmetric)
dataY Ch2 [P]:        min≈-0.012 max≈0.002  (Bernoulli pressure)
```

---

## Physics Interpretation

The generated flow field is **physically consistent**:

- **Stagnation point** at the leading edge (bottom in rotated frame) → high pressure
- **Flow acceleration** at the widest airfoil section → lower pressure (suction)
- **Wake deficit** downstream of the trailing edge (top in rotated frame)
- **Antisymmetric cross-flow** (Ux) on both sides of the symmetric airfoil at 0° AoA
- **Streamlines** curve smoothly around the airfoil body
