import torch
import pickle
import numpy as np
from Models.UNetEx import UNetEx

def check_mode_collapse():
    device = torch.device('cpu')
    model = UNetEx(3, 3, filters=[8, 16, 32, 32], kernel_size=5, batch_norm=True, weight_norm=True)
    model.load_state_dict(torch.load('Run/modelWeights.pth', map_location=device))
    model.eval()
    
    with torch.no_grad():
        x1 = torch.FloatTensor(pickle.load(open('sample1_dataX.pkl', 'rb')))
        x2 = torch.FloatTensor(pickle.load(open('sample2_dataX.pkl', 'rb')))
        x3 = torch.FloatTensor(pickle.load(open('sample3_dataX.pkl', 'rb')))
        
        p1 = model(x1).numpy()
        p2 = model(x2).numpy()
        p3 = model(x3).numpy()
        
        diff_1_2 = np.max(np.abs(p1 - p2))
        diff_2_3 = np.max(np.abs(p2 - p3))
        
        print(f"Max difference between pred 1 and pred 2: {diff_1_2}")
        print(f"Max difference between pred 2 and pred 3: {diff_2_3}")
        print(f"Pred 1 Array mean: {np.mean(p1)}, std: {np.std(p1)}")
        
if __name__ == '__main__':
    check_mode_collapse()
