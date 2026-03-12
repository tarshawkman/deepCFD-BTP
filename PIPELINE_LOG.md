# DeepCFD Custom Dataset & Pipeline Log

This document records the scripts created and the terminal commands used to run the end-to-end DeepCFD custom airfoil evaluation pipeline.

## 1. Scripts Created / Modified

### `SampleDataDeepCFD/DeepCFD.py`
- **Purpose**: The main training script from the original paper.
- **Modifications**: 
  - Added data normalization. Scale factors (`norm_x` and `norm_y`) are calculated using max-absolute normalization across channels and saved as `norm_x.pkl` and `norm_y.pkl` within the `Run` directory. This is critical to prevent output averaging (mode collapse).
  - Explicitly disabled `bn` (Batch Norm) and `wn` (Weight Norm) (`bn=False`, `wn=False`) to match the paper's default hyperparameters, preventing gradient destruction when combined with the unscaled data.
  - Added `torch.save(DeepCFD.state_dict(), ...)` at the end to save the `modelWeights.pth` after training.
  - Replaced `float(torch.sum())` metrics with `.detach().item()` to fix a massive PyTorch memory leak that slowed down Google Colab GPU training.

### `SampleDataDeepCFD/train_functions.py`
- **Purpose**: Helper functions for the PyTorch training loop.
- **Modifications**:
  - Increased DataLoader `num_workers=2` and added `pin_memory=True` to prevent the Colab CPU from bottlenecking the T4 GPU during dataset loading.

### `PanelMethodCode/gen_data.py` (previously `gen_dataY.py` / `create_deepcfd_data.py`)
- **Purpose**: Generates custom `dataX.pkl` and `dataY.pkl` datasets directly from a `.dat` airfoil coordinate file (e.g. `naca0015.dat`).
- **How it works**: It completely bypasses XFOIL/OpenFOAM and uses the native `spvp_airfoil` Panel Method logic (Source/Vortex matrices) to evaluate the flow field. Crucially, it formats the output exactly into the `172x79` PyTorch tensor grid expected by the DeepCFD UNet.

### `SampleDataDeepCFD/evaluate_custom.py`
- **Purpose**: The final testing script. It loads the fully trained model (`modelWeights.pth`), inputs the newly generated `custom_dataX.pkl`, and compares the prediction against `custom_dataY.pkl`.
- **Modifications**: Updated to automatically load `norm_x.pkl` and `norm_y.pkl` from the model directory. It scales the custom input data down before inference, and scales the raw output prediction back up to physical units for an accurate comparison.
- **Output**: Generates a 3x3 Matplotlib plot (`paper_comparison.png`) showing Ground Truth vs DeepCFD vs Absolute Error, perfectly matching the aesthetic of the paper.

### `SampleDataDeepCFD/predict_fields.py`
- **Purpose**: Script to visualize individual channel predictions (Ux, Uy, P) given an input `dataX.pkl` and a trained model.
- **Modifications**: Also updated to apply the saved scaling factors (`norm_x.pkl`, `norm_y.pkl`) for stable inference and to restore `bn=False`, `wn=False` architecture constraints.

---

## 2. Environment Fix Commands Used
During execution, we ran into Python dependency crashes because `matplotlib` expected an older version of NumPy, but downgrading NumPy broke the newer `scipy.interpolate` module (which `spvp_airfoil` needs). 

We stabilized the environment with these terminal commands:
```bash
# Fix Matplotlib C-array crash
pip install "numpy<2"

# Downgrade SciPy to precisely match NumPy 1.x and prevent ufunc TypeErrors
pip install scipy==1.11.4
```

---

## 3. How to Run the Full Pipeline

**Step 1: Generate Custom Data**
Inside the `PanelMethodCode` folder, run the generation script to evaluate the `.dat` file and dump the `.pkl` tensors:
```bash
python gen_data.py
```
*(This creates `custom_dataX.pkl` and `custom_dataY.pkl` in that directory).*

**Step 2: Train Model on Google Colab**
Upload the `SampleDataDeepCFD` folder (with proper `DeepCFD.py` and `train_functions.py`) to Google Colab, select a T4 GPU runtime, and run:
```bash
python DeepCFD.py
```
Download the resulting `Run/modelWeights.pth`.

**Step 3: Evaluate & Visualize**
Inside the `SampleDataDeepCFD` folder, run the evaluation script, passing it the weights and the custom data you generated in Step 1:
```bash
python evaluate_custom.py --model Run/modelWeights.pth --dataX ../PanelMethodCode/custom_dataX.pkl --dataY ../PanelMethodCode/custom_dataY.pkl --out final_comparison.png
```
*(This produces `final_comparison.png`!).*
