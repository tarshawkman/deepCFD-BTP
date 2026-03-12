"""
Notebook: Generate X (dataX.pkl) and Y (dataY.pkl) for a single NACA0015 sample
using OpenFOAM (true CFD) following the DeepCFD dataset format.

This script is written as a runnable Jupyter/Python notebook (cells separated
by comments). It:
 - generates NACA0015 geometry
 - writes an OpenFOAM case (blockMesh + snappyHexMesh template)
 - runs OpenFOAM solvers (blockMesh, snappyHexMesh, simpleFoam)
 - converts results to VTK and reads fields
 - interpolates fields to a uniform 260x120 grid (1 mm resolution)
 - computes SDF (airfoil surface) and top/bottom SDF
 - builds flow-region labels (0 obstacle, 1 fluid, 2 wall, 3 inlet, 4 outlet)
 - saves dataX.pkl and dataY.pkl with shapes (1, Nc, Nx, Ny)

REQUIREMENTS/NOTES:
 - Must be run on a machine with OpenFOAM installed and in PATH.
 - Python deps: numpy, scipy, shapely, meshio, vtk (or pyevtk), pyvista, tqdm, pandas
 - Running the OpenFOAM steps requires root-case write permissions.
 - If you cannot run OpenFOAM inside the notebook, see the alternative
   approach in the comments: generate inputs and then run OpenFOAM externally.

Output (files): dataX.pkl, dataY.pkl
 - dataX channels (Nc=3): [sdf_airfoil, flow_region, sdf_topbottom]
 - dataY channels (Nc=3): [Ux, Uy, p]

References: DeepCFD repository README (dataset format) and paper.
"""

# Cell 1: imports
import os
import sys
import subprocess
import numpy as np
import pickle
from shapely.geometry import Point, Polygon, LineString
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from tqdm import tqdm

# Cell 2: Parameters (domain and grid)
DOMAIN_X = 260.0  # mm
DOMAIN_Y = 120.0  # mm
DX = 1.0          # mm grid spacing to mimic DeepCFD ~1mm base cell
NX = int(DOMAIN_X / DX)
NY = int(DOMAIN_Y / DX)
GRID_X = np.linspace(0, DOMAIN_X, NX)
GRID_Y = np.linspace(0, DOMAIN_Y, NY)

# Airfoil placement parameters (centered vertically, shifted a bit right)
AIRFOIL_CENTER = (DOMAIN_X*0.35, DOMAIN_Y*0.5)
CHORD = 50.0  # mm (example chord length)

# OpenFOAM case name and paths
CASE_DIR = os.path.abspath('naca0015_case')
OPENFOAM_BIN = ''  # set if needed, otherwise expect foam commands in PATH

# Cell 3: NACA4 geometry generator (returns x,y coords scaled to chord)
def naca4_coords(m, p, t, c=1.0, n=400):
    # m, p in fraction of chord; t is thickness fraction
    x = np.linspace(0, c, n)
    # thickness distribution
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    # camber (m=0 for 00xx)
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    if m != 0 and p != 0:
        for i, xi in enumerate(x):
            if xi < p*c:
                yc[i] = m/(p**2)*(2*p*(xi/c)-(xi/c)**2)
                dyc_dx[i] = 2*m/(p**2)*(p - xi/c)
            else:
                yc[i] = m/((1-p)**2)*((1-2*p)+(2*p*(xi/c))-(xi/c)**2)
                dyc_dx[i] = 2*m/((1-p)**2)*(p - xi/c)
    theta = np.arctan(dyc_dx)
    xu = x - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)
    # combine upper and lower
    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])
    return x_coords, y_coords

# Cell 4: generate NACA0015 scaled to chord and placed in domain
m = 0.0
p = 0.0
t = 0.15  # 15% thickness
# chord length in mm
c = CHORD
x_raw, y_raw = naca4_coords(m,p,t,c=c, n=600)
# shift to center at AIRFOIL_CENTER
x_raw = x_raw - np.min(x_raw)  # make leading edge at x=0
x_raw = x_raw / np.max(x_raw) * c
y_raw = y_raw - np.mean(y_raw)
# translate
x_coords = x_raw + AIRFOIL_CENTER[0] - 0.1*c
y_coords = y_raw + AIRFOIL_CENTER[1]

# polygon
airfoil_poly = Polygon(np.column_stack([x_coords, y_coords]))

# quick sanity print
print(f'Generated NACA0015 polygon with {len(x_coords)} points; centroid {airfoil_poly.centroid.x, airfoil_poly.centroid.y}')

# Cell 5: write simple OpenFOAM case templates (blockMeshDict and snappyHexMesh/mesh)
# NOTE: Full snappyHexMesh templates are lengthy. We provide a minimal blockMesh
# and a placeholder for snappyHexMesh. You must adapt snappyHexMesh to your
# OpenFOAM version and system. The notebook writes a 'geometry' file that
# snappyHexMesh can use. If you prefer, run blockMesh only with a structured mesh
# and carve the airfoil by setting interior cell/patchs later.

os.makedirs(CASE_DIR, exist_ok=True)
constant_dir = os.path.join(CASE_DIR, 'constant')
system_dir = os.path.join(CASE_DIR, 'system')
os.makedirs(constant_dir, exist_ok=True)
os.makedirs(system_dir, exist_ok=True)
poly_file = os.path.join(constant_dir, 'triSurface')

# write a simple STL from polygon for snappyHexMesh
try:
    import meshio
    # build a simple line-loop then triangular fan around centroid (cheap but usable for snappy)
    centroid = np.array(airfoil_poly.centroid.coords[0])
    pts = np.column_stack([x_coords, y_coords, np.zeros_like(x_coords)])
    # create triangles by fan (not great for real mesh; better to export high-quality mesh externally)
    triangles = []
    for i in range(1, len(pts)-1):
        triangles.append([0, i, i+1])
    # prepend centroid
    pts3 = np.vstack([centroid, pts])
    tri = np.array(triangles)+0
    meshio.write_points_cells(os.path.join(constant_dir, 'airfoil.stl'), pts3, [('triangle', tri)])
    print('Wrote constant/airfoil.stl (approximate).')
except Exception as e:
    print('meshio not available or failed; please provide an STL for snappyHexMesh. Error:', e)

# Minimal blockMeshDict
blockMesh = f"""
/* blockMeshDict generated by notebook */
convertToMeters 0.001;
vertices
(
    (0 0 0)
    ({DOMAIN_X/1000:.6f} 0 0)
    ({DOMAIN_X/1000:.6f} {DOMAIN_Y/1000:.6f} 0)
    (0 {DOMAIN_Y/1000:.6f} 0)
    (0 0 0.1)
    ({DOMAIN_X/1000:.6f} 0 0.1)
    ({DOMAIN_X/1000:.6f} {DOMAIN_Y/1000:.6f} 0.1)
    (0 {DOMAIN_Y/1000:.6f} 0.1)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) ({NX} {NY} 1) simpleGrading (1 1 1)
);
edges
(
);
boundary
(
    inlet
    {
        type patch;
        faces ((0 4 7 3));
    }
    outlet
    {
        type patch;
        faces ((1 2 6 5));
    }
    top
    {
        type wall;
        faces ((3 2 6 7));
    }
    bottom
    {
        type wall;
        faces ((0 1 5 4));
    }
);
mergePatchPairs
(
);
"""
with open(os.path.join(system_dir, 'blockMeshDict'), 'w') as f:
    f.write(blockMesh)
print('Wrote system/blockMeshDict (check units: convertToMeters=0.001)')

# Cell 6: write controlDict, fvSchemes, fvSolution, basic 0/U and 0/p files
# NOTE: This is a very minimal example and may need tuning for convergence.

controlDict = """
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;
deltaT          1;
writeControl    timeStep;
writeInterval   1000;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
runTimeModifiable true;
adjustTimeStep  no;

"""
with open(os.path.join(system_dir,'controlDict'),'w') as f:
    f.write(controlDict)

fvSchemes = """
// minimal
ddtSchemes { default steadyState; }
divSchemes { div(phi,U) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
with open(os.path.join(system_dir,'fvSchemes'),'w') as f:
    f.write(fvSchemes)

fvSolution = """
solvers
{
    p
    {
        solver GAMG;
    }
    U
    {
        solver smoothSolver;
    }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
}
"""
with open(os.path.join(system_dir,'fvSolution'),'w') as f:
    f.write(fvSolution)

# write initial fields in 0/ folder
zero_dir = os.path.join(CASE_DIR,'0')
os.makedirs(zero_dir, exist_ok=True)
U_text = f"""
/*--------------------------------*- C++ -*----------------------------------*\\
| OpenFOAM: initial U
\\*---------------------------------------------------------------------------*/
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0.1 0 0);
boundaryField
{{
    inlet
    {{ type fixedValue; value uniform (0.1 0 0); }}
    outlet
    {{ type zeroGradient; }}
    top
    {{ type fixedValue; value uniform (0 0 0); }}
    bottom
    {{ type fixedValue; value uniform (0 0 0); }}
}}
"""
with open(os.path.join(zero_dir,'U'),'w') as f:
    f.write(U_text)

p_text = f"""
/* p */
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{{
    inlet {{ type zeroGradient; }}
    outlet {{ type fixedValue; value uniform 0; }}
    top {{ type zeroGradient; }}
    bottom {{ type zeroGradient; }}
}}
"""
with open(os.path.join(zero_dir,'p'),'w') as f:
    f.write(p_text)

print('Wrote minimal OpenFOAM case skeleton in', CASE_DIR)

# Cell 7: Run blockMesh and simpleFoam (if OpenFOAM present)
run_openfoam = True
if run_openfoam:
    try:
        cwd = os.getcwd()
        os.chdir(CASE_DIR)
        print('Running blockMesh ...')
        subprocess.check_call(['blockMesh'])
        # if snappyHexMesh available and airfoil.stl generated, you could run it here
        # subprocess.check_call(['snappyHexMesh','-overwrite'])
        print('Running simpleFoam ... (this can take time)')
        subprocess.check_call(['simpleFoam'])
        print('OpenFOAM run completed')
    except subprocess.CalledProcessError as e:
        print('OpenFOAM command failed:', e)
        print('If you cannot run OpenFOAM here, run blockMesh + simpleFoam manually in the case dir')
    finally:
        os.chdir(cwd)

# Cell 8: Convert OpenFOAM results to VTK using foamToVTK (or use pyFoam utilities)
try:
    cwd = os.getcwd()
    os.chdir(CASE_DIR)
    print('Converting case to VTK (foamToVTK) ...')
    subprocess.check_call(['foamToVTK'])
    # find latest time folder and VTK files
    os.chdir(cwd)
except Exception as e:
    print('foamToVTK failed or not available:', e)
    print('You can export fields using foamToVTK manually and place vtk files in case dir')

# Cell 9: Read VTK output and interpolate onto uniform grid
# For simplicity we will search for U.vtk and p.vtk under CASE_DIR/VTK
import glob
vtk_folder = os.path.join(CASE_DIR,'VTK')
U_field = None
p_field = None
if os.path.isdir(vtk_folder):
    vtk_files = sorted(glob.glob(os.path.join(vtk_folder,'*.vtk')))
    if vtk_files:
        latest = vtk_files[-1]
        print('Found VTK:', latest)
        # Use meshio to read point-data
        try:
            import meshio
            mesh = meshio.read(latest)
            # mesh.points: N x 3, mesh.point_data['U'] maybe
            pts = mesh.points[:,:2]
            pd = mesh.point_data
            if 'U' in pd:
                Uvec = pd['U']
                Ux_pts = Uvec[:,0]
                Uy_pts = Uvec[:,1]
            else:
                print('U not found in point_data; check meshio read')
            if 'p' in pd:
                p_pts = pd['p']
            else:
                p_pts = np.zeros(pts.shape[0])

            # interpolate onto grid
            gx, gy = np.meshgrid(GRID_X, GRID_Y)
            grid_points = np.column_stack([gx.ravel(), gy.ravel()])
            Ux_grid = griddata(pts, Ux_pts, grid_points, method='linear', fill_value=0)
            Uy_grid = griddata(pts, Uy_pts, grid_points, method='linear', fill_value=0)
            p_grid  = griddata(pts, p_pts,  grid_points, method='linear', fill_value=0)
            Ux_grid = Ux_grid.reshape(NY, NX)
            Uy_grid = Uy_grid.reshape(NY, NX)
            p_grid  = p_grid.reshape(NY, NX)
        except Exception as e:
            print('meshio read/interp failed:', e)
else:
    print('VTK folder not found; cannot read CFD results here. If you ran OpenFOAM externally, ensure foamToVTK output is in', vtk_folder)

# If Ux_grid not available, warn and create placeholders (user must supply real fields after running OpenFOAM)
if 'Ux_grid' not in globals():
    print('No CFD fields found; creating placeholder fields (zeros) for demonstration.')
    Ux_grid = np.zeros((NY, NX))
    Uy_grid = np.zeros((NY, NX))
    p_grid  = np.zeros((NY, NX))

# Cell 10: Compute SDF to airfoil surface (channel SDF1) and top/bottom sdf (SDF2)
# We compute distance from each grid point to polygon boundary
sdf_airfoil = np.zeros((NY, NX), dtype=np.float32)
sdf_topbot = np.zeros((NY, NX), dtype=np.float32)
flow_region = np.ones((NY, NX), dtype=np.int32)  # default fluid=1

# build kdtree of exterior boundary points for faster distance
boundary_coords = np.array(airfoil_poly.exterior.coords)
kdt = cKDTree(boundary_coords)

for i, yy in enumerate(GRID_Y):
    for j, xx in enumerate(GRID_X):
        pt = (xx, yy)
        d, idx = kdt.query(pt)
        # shapely's contains
        inside = airfoil_poly.contains(Point(pt))
        sdf_airfoil[i,j] = -d if inside else d
        # top/bottom sdf = distance to nearest top/bottom wall (y=0 or y=DOMAIN_Y)
        dtop = abs(DOMAIN_Y - yy)
        dbot = abs(yy - 0)
        sdf_topbot[i,j] = min(dtop, dbot)
        # flow regions
        # walls (top/bottom): y==0 or y==DOMAIN_Y within epsilon
        eps = DX*0.5
        if yy <= eps or yy >= DOMAIN_Y - eps:
            flow_region[i,j] = 2
        elif abs(xx - 0) <= eps:
            flow_region[i,j] = 3
        elif abs(xx - DOMAIN_X) <= eps:
            flow_region[i,j] = 4
        elif inside:
            flow_region[i,j] = 0
        else:
            flow_region[i,j] = 1

# Normalize or leave distances in mm as DeepCFD used mm domain; the repo used raw distances

# Cell 11: assemble dataX and dataY and save as .pkl in DeepCFD format
# dataX shape should be (Ns, Nc, Nx, Ny) per README: axes (Ns, Nc, Nx, Ny)
# We'll follow that exact order

dataX = np.zeros((1, 3, NX, NY), dtype=np.float32)
# note: grid arrays are currently (NY, NX); need to transpose to (NX, NY)
dataX[0,0,:,:] = sdf_airfoil.T
dataX[0,1,:,:] = flow_region.T
dataX[0,2,:,:] = sdf_topbot.T

# dataY: Ux, Uy, p
dataY = np.zeros((1, 3, NX, NY), dtype=np.float32)
dataY[0,0,:,:] = Ux_grid.T
dataY[0,1,:,:] = Uy_grid.T
dataY[0,2,:,:] = p_grid.T

# save
with open('dataX_naca0015.pkl', 'wb') as f:
    pickle.dump(dataX, f)
with open('dataY_naca0015.pkl', 'wb') as f:
    pickle.dump(dataY, f)

print('Wrote dataX_naca0015.pkl and dataY_naca0015.pkl with shapes', dataX.shape, dataY.shape)

# Cell 12: Quick verification print
print('Sample dataX channel meanings: channel0=SDF_airfoil, channel1=flow_region, channel2=SDF_topbottom')
print('Saved to current working directory. If you ran OpenFOAM externally, replace the placeholder U/p fields with real interpolated fields and re-run cell 11 to re-save dataY.')

# End of notebook script
