import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

# -------------------------------------------------------------
# 1. DEEPCFD MODEL ARCHITECTURE (UNet Convolution Template)
# -------------------------------------------------------------
class UNetEx(nn.Module):
    """ A simplified PyTorch CNN for 3-channel to 3-channel mapping """
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetEx, self).__init__()
        
        # Simple Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        b = self.bottleneck(self.pool(e1))
        u = self.up(b)
        
        import torch.nn.functional as F
        # Hardware padding to match odd geometric resolutions (79 / 2 == 39.5)
        if u.shape[-1] != x.shape[-1]: u = F.pad(u, (0, 1, 0, 0)) 
        if u.shape[-2] != x.shape[-2]: u = F.pad(u, (0, 0, 0, 1)) 
            
        out = self.dec1(torch.cat([e1, u], dim=1))
        return out

# -------------------------------------------------------------
# 2. DATA LOADING & SPLITTING
# -------------------------------------------------------------
print("Loading datasets...")

# NOTE FOR COLAB: You will change these strings to match where your Drive is mounted
# Example: '/content/drive/My Drive/dataX_full.pkl'
PATH_DATA_X = 'dataX_full.pkl' 
PATH_DATA_Y = 'dataY_full.pkl'

with open(PATH_DATA_X, 'rb') as f: X_all = pickle.load(f)
with open(PATH_DATA_Y, 'rb') as f: Y_all = pickle.load(f)

print(f"Loaded full data. Shape: {X_all.shape}")

# Split the data 80% Train, 20% Test
X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)

# Save the test data to disk immediately so evaluate_model.py can benchmark against it later!
with open('dataX_test.pkl', 'wb') as f: pickle.dump(X_test, f)
with open('dataY_test.pkl', 'wb') as f: pickle.dump(Y_test, f)
print(f"Saved independent Test Sets! X_train: {X_train.shape}, X_test: {X_test.shape}")

# Convert to PyTorch DataLoaders
dataset_train = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
loader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)

# -------------------------------------------------------------
# 3. PyTorch TRAINING LOOP
# -------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing Training Engine on hardware: {device}")

model = UNetEx(in_channels=3, out_channels=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 50

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for batch_X, batch_Y in loader_train:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_Y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_X.size(0)
        
    epoch_loss = running_loss / len(dataset_train)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.7f}")

# -------------------------------------------------------------
# 4. EXPORT WEIGHTS FOR PHASE 3
# -------------------------------------------------------------
torch.save(model.state_dict(), 'deepcfd_weights.pt')
print("\nTraining Phase Complete! Saved model state directly to 'deepcfd_weights.pt'")
