import pickle
import numpy as np
with open(r'c:\Users\Aaditya Saraf\Desktop\btp\SampleDataDeepCFD\dataX.pkl', 'rb') as f:
    x = pickle.load(f)

import matplotlib.pyplot as plt

with open(r'c:\Users\Aaditya Saraf\Desktop\btp\SampleDataDeepCFD\dataY.pkl', 'rb') as f:
    y = pickle.load(f)

for i in range(3):
    print(f"X ch{i} min={np.min(x[0, i])}, max={np.max(x[0, i])}")
for i in range(3):
    print(f"Y ch{i} min={np.min(y[0, i])}, max={np.max(y[0, i])}")

