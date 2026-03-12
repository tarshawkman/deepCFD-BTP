import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load files
with open("dataX.pkl", "rb") as f:
    X = pickle.load(f)
with open("dataY.pkl", "rb") as f:
    Y = pickle.load(f)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# Example: view one sample
i = 0
sample_X = X[i]
sample_Y = Y[i]

# Flatten to CSV-style for inspection
dfX = pd.DataFrame(sample_X.reshape(3, -1).T, columns=["SDF_obstacle", "FlowRegion", "SDF_topbottom"])
dfY = pd.DataFrame(sample_Y.reshape(3, -1).T, columns=["Ux", "Uy", "p"])

print(dfX.head())
print(dfY.head())

# Optional: visualize one channel
plt.imshow(sample_X[0], cmap='jet')
plt.title("SDF of Obstacle")
plt.show()
