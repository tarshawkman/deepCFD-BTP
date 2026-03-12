import pickle
import numpy as np
import os

def extract_sample(dataX_path, dataY_path, sample_idx=0, output_dir="."):
    print(f"Loading datasets from {dataX_path} and {dataY_path}")
    
    with open(dataX_path, "rb") as f:
        dataX = pickle.load(f)
        
    with open(dataY_path, "rb") as f:
        dataY = pickle.load(f)
        
    # dataX and dataY are lists / arrays of shape (981, 3, 172, 79) usually
    if isinstance(dataX, list):
        dataX = np.array(dataX)
    if isinstance(dataY, list):
        dataY = np.array(dataY)
        
    print(f"Original dataX shape: {dataX.shape}")
    print(f"Original dataY shape: {dataY.shape}")
    
    # Extract the requested sample and keep dimensions [1, 3, 172, 79]
    sampleX = np.expand_dims(dataX[sample_idx], axis=0)
    sampleY = np.expand_dims(dataY[sample_idx], axis=0)
    
    outX_path = os.path.join(output_dir, f"sample{sample_idx+1}_dataX.pkl")
    outY_path = os.path.join(output_dir, f"sample{sample_idx+1}_dataY.pkl")
    
    with open(outX_path, "wb") as f:
        pickle.dump(sampleX, f)
        
    with open(outY_path, "wb") as f:
        pickle.dump(sampleY, f)
        
    print(f"Extracted Sample #{sample_idx+1}.")
    print(f"Saved: {outX_path}  (Shape: {sampleX.shape})")
    print(f"Saved: {outY_path}  (Shape: {sampleY.shape})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataX", default="dataX.pkl", help="Original full dataset X")
    parser.add_argument("--dataY", default="dataY.pkl", help="Original full dataset Y")
    parser.add_argument("--index", type=int, default=0, help="0-indexed sample to extract")
    args = parser.parse_args()
    
    extract_sample(args.dataX, args.dataY, args.index)
