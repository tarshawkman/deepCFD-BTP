import torch
import numpy as np
import pickle
import os
import glob
from matplotlib.path import Path
from scipy.ndimage import distance_transform_edt
import argparse
from tqdm import tqdm

def load_raw_dat(filepath):
    x, y = [], []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 2:
                try: x.append(float(parts[0])); y.append(float(parts[1]))
                except ValueError: pass
    if len(x) == 0: return None, None
    return np.array(x), np.array(y)

def solve_lbm_gpu_batched(mask_np, V_inlet=0.1, max_iters=4500, device='cuda'):
    """ Fully Batched GPU Engine - Processes multiple airfoils simultaneously! """
    B, Ny, Nx = mask_np.shape
    tau = 0.6
    omega = 1.0 / tau
    
    obstacle = torch.tensor(mask_np > 0.5, device=device, dtype=torch.bool)
    cx = torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], device=device, dtype=torch.float32)
    cy = torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1], device=device, dtype=torch.float32)
    weights = torch.tensor([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], device=device, dtype=torch.float32)
    opposite = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], device=device, dtype=torch.long)
    
    u = torch.zeros((B, 2, Ny, Nx), device=device, dtype=torch.float32)
    u[:, 0, :, :] = V_inlet
    
    # Broadcast zeros explicitly handling index structures
    u[:, 0, ...].masked_fill_(obstacle, 0.0)
    u[:, 1, ...].masked_fill_(obstacle, 0.0)
    
    rho = torch.ones((B, Ny, Nx), device=device, dtype=torch.float32)
    
    def get_equilibrium_batched(rho_f, u_f):
        if rho_f.dim() == 3: rho_f = rho_f.unsqueeze(1)
        usqr = 1.5 * (u_f[:, 0]**2 + u_f[:, 1]**2).unsqueeze(1)
        c_u0 = cy.view(1, 9, 1, 1) * u_f[:, 0].unsqueeze(1)
        c_u1 = cx.view(1, 9, 1, 1) * u_f[:, 1].unsqueeze(1)
        cu = 3.0 * (c_u0 + c_u1)
        return weights.view(1, 9, 1, 1) * rho_f * (1.0 + cu + 0.5 * cu**2 - usqr)

    fin = get_equilibrium_batched(rho, u)
    
    idx_top, idx_bot = [4, 7, 8], [2, 5, 6]
    idx_lft, idx_rgt = [3, 6, 7], [1, 5, 8]
    shifts = [(int(cy[i].item()), int(cx[i].item())) for i in range(9)]
    
    u_inlet_base = torch.zeros((B, 2, Nx), device=device, dtype=torch.float32)
    u_inlet_base[:, 0, :] = V_inlet
    u_inlet_exp = u_inlet_base.unsqueeze(2)
    fout = torch.empty_like(fin)
    
    for iter in range(max_iters):
        for i in idx_top: fin[:, i, -1, :] = fin[:, i, -2, :]
            
        rho_inlet = rho[:, 1, :].unsqueeze(1).unsqueeze(2)
        feq_inlet = get_equilibrium_batched(rho_inlet, u_inlet_exp)
        for i in idx_bot: fin[:, i, 0, :] = feq_inlet[:, i, 0, :]
            
        fout.copy_(fin)
        for i in range(9):
            sy, sx = shifts[i]
            if sx != 0 or sy != 0: fin[:, i] = torch.roll(fin[:, i], shifts=(sy, sx), dims=(1, 2))
                
        for i in idx_rgt: fin[:, i, :, 0] = fout[:, opposite[i], :, 0]
        for i in idx_lft: fin[:, i, :, -1] = fout[:, opposite[i], :, -1]
        
        for i in range(9):
            fin[:, i] = torch.where(obstacle, fout[:, opposite[i]], fin[:, i])
            
        rho = torch.sum(fin, dim=1)
        u[:, 0] = torch.sum(fin * cy.view(1, 9, 1, 1), dim=1) / rho
        u[:, 1] = torch.sum(fin * cx.view(1, 9, 1, 1), dim=1) / rho
        
        feq = get_equilibrium_batched(rho, u)
        fin = fin - omega * (fin - feq)
    
    p = (rho - 1.0) / 3.0
    return u.cpu().numpy(), p.cpu().numpy()

def batch_generate_gpu(dat_folder, out_folder):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dat_files = glob.glob(os.path.join(dat_folder, "**", "*.dat"), recursive=True)
    Ny, Nx = 172, 79
    Vinf = 0.1
    BATCH_SIZE = 64  # Evaluate 64 fluid simulations simultaneously

    print(f"=== HARDWARE: {device.upper()} | ENABLING BATCH TENSORS OF {BATCH_SIZE} ===")
    print(f"Found {len(dat_files)} airfoils. Matrix aggregation ready...")
    
    all_dataX, all_dataY = [], []
    curr_masks, curr_sdfs = [], []
    
    def execute_batch():
        if len(curr_masks) == 0: return
        B = len(curr_masks)
        mask_np = np.stack(curr_masks, axis=0) # (B, 172, 79)
        u, p = solve_lbm_gpu_batched(mask_np, Vinf, max_iters=4500, device=device)
        
        for i in range(B):
            Uy, Ux, P = u[i, 0], u[i, 1], p[i]
            Uy[curr_masks[i] > 0.5] = 0; Ux[curr_masks[i] > 0.5] = 0; P[curr_masks[i] > 0.5] = 0
            
            all_dataY.append(np.stack([Uy, Ux, P]))
            all_dataX.append(np.stack([curr_sdfs[i], curr_masks[i], np.full((Ny,Nx), Vinf)]))
            
        curr_masks.clear()
        curr_sdfs.clear()

    for idx, dat_file in enumerate(tqdm(dat_files)):
        try:
            XB, YB = load_raw_dat(dat_file)
            if XB is None or len(XB) < 10: continue
            
            chord = np.max(XB) - np.min(XB)
            if chord <= 0: continue
            
            XB_r, YB_r = -YB * (40.0/chord), XB * (40.0/chord)
            XB_r += 39.0 - np.mean(XB_r); YB_r += 45.0 - np.min(YB_r)
            
            Xgrid, Ygrid = np.meshgrid(np.arange(Nx), np.arange(Ny))
            mask = Path(np.column_stack((XB_r, YB_r))).contains_points(np.column_stack((Xgrid.flatten(), Ygrid.flatten()))).reshape((Ny, Nx)).astype(float)
            
            curr_masks.append(mask)
            curr_sdfs.append(distance_transform_edt(1 - mask))
            
            if len(curr_masks) == BATCH_SIZE: execute_batch()
        except: pass

    execute_batch() # Clear memory remainder
    
    final_dataX, final_dataY = np.stack(all_dataX, axis=0), np.stack(all_dataY, axis=0)
    print("\n--- MATRIX BATCH PROCESSING COMPLETE ---")
    print(f"Constructed Full Tensors: {final_dataX.shape}")
    
    with open(os.path.join(out_folder, "dataX_full.pkl"), "wb") as f: pickle.dump(final_dataX, f)
    with open(os.path.join(out_folder, "dataY_full.pkl"), "wb") as f: pickle.dump(final_dataY, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dat_dir", required=True)
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()
    batch_generate_gpu(args.dat_dir, args.out_dir)
