import pickle
import numpy as np

with open('dataX.pkl', 'rb') as f:
    x_orig = pickle.load(f)[0]
with open('dataY.pkl', 'rb') as f:
    y_orig = pickle.load(f)[0]

print("--- dataX[0] (Input Channels) ---")
for i in range(3):
    c = x_orig[i]
    print(f"Chan {i}: min={np.min(c):.4f}, max={np.max(c):.4f}, mean={np.mean(c):.4f}")

print("--- dataY[0] (Output Channels) ---")
for i in range(3):
    c = y_orig[i]
    print(f"Chan {i}: min={np.min(c):.4f}, max={np.max(c):.4f}, mean={np.mean(c):.4f}")
