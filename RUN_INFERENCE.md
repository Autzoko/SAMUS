# AutoSAMUS Inference on BUSI & BUS Datasets

Complete guide to run AutoSAMUS inference on the **BUSI** (Breast Ultrasound Images) and **BUS** (Dataset B) datasets.

## Overview

| Stage | Script | What it does |
|-------|--------|-------------|
| 1. Preprocess | `preprocess_datasets.py` | Converts raw BUSI/BUS into SAMUS format |
| 2. Checkpoint | (manual download) | Download the pre-trained AutoSAMUS weights |
| 3. Inference | `inference_autosamus.py` | Runs AutoSAMUS, evaluates, and visualizes |

## Prerequisites

### 1. Environment Setup

```bash
conda create -n SAMUS python=3.8
conda activate SAMUS
```

Install PyTorch (choose one based on your hardware):

```bash
# CUDA 11.1
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only (macOS / no GPU)
pip install torch torchvision

# Apple Silicon (MPS)
pip install torch torchvision
```

Install dependencies:

```bash
cd "/Volumes/Autzoko/MS Thesis/SAMUS"
pip install -r requirements.txt
```

> **Note:** If `thop` is not installed (used in `test.py` but not in our inference script), you can ignore it. Our inference script does not depend on `thop`.

### 2. Download Pre-trained Checkpoint

Download the SAMUS pre-trained model from:

**Google Drive:** [SAMUS checkpoint](https://drive.google.com/file/d/1nQjMAvbPeolNpCxQyU_HTiOiB5704pkH/view?usp=sharing)

Place it in the `checkpoints/` directory:

```bash
mkdir -p checkpoints
# After downloading, move it:
mv ~/Downloads/SAMUS.pth checkpoints/AutoSAMUS.pth
```

> The checkpoint file works for both SAMUS and AutoSAMUS. The `build_samus.py` loader handles weight mapping via `load_from_pretrained()` automatically.

> If you have a separately trained AutoSAMUS checkpoint, use that instead.

## Step 1: Preprocess Datasets

Convert the raw BUSI and BUS datasets into the SAMUS-compatible format:

```bash
cd "/Volumes/Autzoko/MS Thesis/SAMUS"

python preprocess_datasets.py \
    --busi_dir "/Users/langtian/Desktop/NYU/MS Thesis/3D SAM Foundation Model/Med3D/Data/BUSI" \
    --bus_dir  "/Users/langtian/Desktop/NYU/MS Thesis/3D SAM Foundation Model/Med3D/Data/BUS" \
    --output_dir ./data/processed
```

This creates:

```
data/processed/
├── MainPatient/
│   ├── class.json                    # {"Breast-BUSI-Ext": 2, "Breast-BUS-Ext": 2}
│   ├── test.txt                      # Combined test split
│   ├── test-Breast-BUSI-Ext.txt      # BUSI test split
│   ├── test-Breast-BUS-Ext.txt       # BUS test split
│   ├── train.txt / val.txt           # Empty (inference only)
│   └── train-*.txt / val-*.txt       # Empty (inference only)
├── Breast-BUSI-Ext/
│   ├── img/     # Grayscale PNGs
│   └── label/   # Binary mask PNGs (0/1)
└── Breast-BUS-Ext/
    ├── img/
    └── label/
```

**What the preprocessor does:**
- **BUSI:** Reads images from `benign/`, `malignant/`, `normal/` folders; merges multiple mask files (`_mask.png`, `_mask_1.png`, etc.) via union; converts to grayscale; binarizes masks to 0/1; skips `normal` images with empty masks.
- **BUS:** Reads from `original/` and `GT/`; converts 0/255 masks to 0/1.

## Step 2: Run Inference

### On BUSI

```bash
python inference_autosamus.py \
    --data_path ./data/processed \
    --dataset Breast-BUSI-Ext \
    --checkpoint ./checkpoints/AutoSAMUS.pth \
    --output_dir ./results \
    --batch_size 8 \
    --visualize
```

### On BUS

```bash
python inference_autosamus.py \
    --data_path ./data/processed \
    --dataset Breast-BUS-Ext \
    --checkpoint ./checkpoints/AutoSAMUS.pth \
    --output_dir ./results \
    --batch_size 8 \
    --visualize
```

### Run on CPU (if no GPU available)

```bash
python inference_autosamus.py \
    --data_path ./data/processed \
    --dataset Breast-BUSI-Ext \
    --checkpoint ./checkpoints/AutoSAMUS.pth \
    --device cpu \
    --batch_size 4 \
    --visualize
```

### Command-line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data_path` | `./data/processed` | Root of preprocessed data |
| `--dataset` | (required) | `Breast-BUSI-Ext` or `Breast-BUS-Ext` |
| `--checkpoint` | (required) | Path to `.pth` file |
| `--output_dir` | `./results` | Where to save outputs |
| `--batch_size` | `8` | Batch size |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--visualize` | off | Add `--visualize` to save overlay images |
| `--encoder_input_size` | `256` | Encoder input resolution |
| `--low_image_size` | `128` | Low-res mask resolution |
| `--seed` | `300` | Random seed |

## Step 3: View Results

### Output Structure

```
results/
├── Breast-BUSI-Ext/
│   ├── results.json           # Full results (summary + per-image)
│   ├── per_image_metrics.csv  # Per-image metrics table
│   └── vis/
│       ├── overlay/           # [Original | GT | Prediction] images
│       └── pred_masks/        # Raw binary prediction masks (0/255)
└── Breast-BUS-Ext/
    └── ...
```

### Console Output Example

```
======================================================================
  Results for Breast-BUSI-Ext
======================================================================
  Images evaluated : 437
  Inference speed  : 42.3 images/sec
  Mean loss        : 0.1234

  Metric              Mean      Std
  ------------------------------
  Dice (%)           78.52     15.23
  IoU (%)            67.31     18.45
  HD                 32.10     25.67
  Accuracy (%)       95.12      3.45
  Sensitivity(%)     80.23     16.78
  Specificity(%)     97.56      2.34
```

### Evaluation Metrics

| Metric | Formula | Range |
|--------|---------|-------|
| **Dice** | 2TP / (FP + 2TP + FN) | 0-100% |
| **IoU** | TP / (FP + TP + FN) | 0-100% |
| **Hausdorff Distance** | max surface distance (Manhattan) | 0-inf |
| **Accuracy** | (TP + TN) / (TP + FP + FN + TN) | 0-100% |
| **Sensitivity** | TP / (TP + FN) | 0-100% |
| **Specificity** | TN / (FP + TN) | 0-100% |

## Quick-Start (Copy & Paste)

Run everything end-to-end:

```bash
cd "/Volumes/Autzoko/MS Thesis/SAMUS"
conda activate SAMUS

# Step 1: Preprocess
python preprocess_datasets.py \
    --busi_dir "/Users/langtian/Desktop/NYU/MS Thesis/3D SAM Foundation Model/Med3D/Data/BUSI" \
    --bus_dir  "/Users/langtian/Desktop/NYU/MS Thesis/3D SAM Foundation Model/Med3D/Data/BUS" \
    --output_dir ./data/processed

# Step 2: Inference on BUSI
python inference_autosamus.py \
    --data_path ./data/processed \
    --dataset Breast-BUSI-Ext \
    --checkpoint ./checkpoints/AutoSAMUS.pth \
    --output_dir ./results \
    --visualize

# Step 3: Inference on BUS
python inference_autosamus.py \
    --data_path ./data/processed \
    --dataset Breast-BUS-Ext \
    --checkpoint ./checkpoints/AutoSAMUS.pth \
    --output_dir ./results \
    --visualize
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'monai'` | `pip install monai==1.1.0` |
| `ModuleNotFoundError: No module named 'hausdorff'` | `pip install hausdorff==0.2.6` |
| `ModuleNotFoundError: No module named 'einops'` | `pip install einops==0.6.1` |
| `ModuleNotFoundError: No module named 'batchgenerators'` | `pip install batchgenerators==0.25` |
| `CUDA out of memory` | Reduce `--batch_size` to 2 or 1 |
| `Split file not found` | Run `preprocess_datasets.py` first |
| `RuntimeError: ... expected ... got ...` on MPS | Use `--device cpu` (MPS may lack some ops) |
| Checkpoint loading fails with key mismatch | The script handles `module.` prefix stripping and partial loading automatically. Ensure the checkpoint is the official SAMUS release. |
