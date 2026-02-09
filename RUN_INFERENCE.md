# Training & Inference Guide: SAMUS / AutoSAMUS on BUSI, BUS & BUSBRA

Complete guide to **train** AutoSAMUS and **run inference** with SAMUS or AutoSAMUS on the **BUSI**, **BUS**, and **BUSBRA** breast ultrasound datasets.

## Overview

| Stage | Script | What it does |
|-------|--------|-------------|
| 1. Preprocess | `preprocess_datasets.py` | Converts raw BUSI/BUS/BUSBRA into SAMUS format with train/val/test splits |
| 2. Checkpoint | (manual download) | Download the pre-trained SAMUS weights |
| 3. Train | `train.py` | Fine-tunes AutoSAMUS on the training split |
| 4. Inference | `inference_autosamus.py` | Runs SAMUS or AutoSAMUS on the test split, evaluates, and visualizes |

## Prerequisites

### 1. Environment Setup

```bash
conda create -n SAMUS python=3.8
conda activate SAMUS
```

**Step A -- Install PyTorch first** (must be done before `pip install -r requirements.txt`):

```bash
# CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

For other hardware:

```bash
# CPU only (macOS / no GPU)
pip install torch torchvision torchaudio

# Apple Silicon (MPS)
pip install torch torchvision torchaudio
```

**Step B -- Install remaining dependencies:**

```bash
cd "/Volumes/Autzoko/MS Thesis/SAMUS"
pip install -r requirements.txt
```

> **Note:** PyTorch is deliberately **not** listed in `requirements.txt` because CUDA wheels require the special `--index-url` flag. Always install PyTorch first (Step A), then run `pip install -r requirements.txt` (Step B).

### 2. Download Pre-trained SAMUS Checkpoint

Download the SAMUS pre-trained model from:

**Google Drive:** [SAMUS checkpoint](https://drive.google.com/file/d/1nQjMAvbPeolNpCxQyU_HTiOiB5704pkH/view?usp=sharing)

Place it in the `checkpoints/` directory:

```bash
mkdir -p checkpoints
mv ~/Downloads/SAMUS.pth checkpoints/SAMUS.pth
```

> This checkpoint contains the pre-trained SAMUS weights. It can be used directly for SAMUS inference, or as initialization for training AutoSAMUS.

## Step 1: Preprocess Datasets

Convert the raw datasets into the SAMUS-compatible format.

### For Training (with train/val/test splits)

```bash
cd "/Volumes/Autzoko/MS Thesis/SAMUS"

python preprocess_datasets.py \
    --busi_dir   "/Users/langtian/Desktop/NYU/MS Thesis/3D SAM Foundation Model/Med3D/Data/BUSI" \
    --bus_dir    "/Users/langtian/Desktop/NYU/MS Thesis/3D SAM Foundation Model/Med3D/Data/BUS" \
    --busbra_dir "/Users/langtian/Desktop/NYU/MS Thesis/3D SAM Foundation Model/Med3D/Data/BUSBRA" \
    --output_dir ./data/processed \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --seed 42
```

This creates a 70/15/15 train/val/test split (shuffled with seed 42).

### For Inference Only (all data in test split)

```bash
python preprocess_datasets.py \
    --busi_dir   "/Users/langtian/Desktop/NYU/MS Thesis/3D SAM Foundation Model/Med3D/Data/BUSI" \
    --bus_dir    "/Users/langtian/Desktop/NYU/MS Thesis/3D SAM Foundation Model/Med3D/Data/BUS" \
    --busbra_dir "/Users/langtian/Desktop/NYU/MS Thesis/3D SAM Foundation Model/Med3D/Data/BUSBRA" \
    --output_dir ./data/processed \
    --train_ratio 0.0 \
    --val_ratio 0.0
```

### Output Structure

```
data/processed/
├── MainPatient/
│   ├── class.json                      # {"Breast-BUSI-Ext": 2, ...}
│   ├── train.txt / val.txt / test.txt  # Combined splits
│   ├── train-Breast-BUSI-Ext.txt       # Per-dataset train split
│   ├── val-Breast-BUSI-Ext.txt         # Per-dataset val split
│   ├── test-Breast-BUSI-Ext.txt        # Per-dataset test split
│   └── ... (same for BUS-Ext, BUSBRA-Ext)
├── Breast-BUSI-Ext/
│   ├── img/     # Grayscale PNGs
│   └── label/   # Binary mask PNGs (0/1)
├── Breast-BUS-Ext/
│   ├── img/
│   └── label/
└── Breast-BUSBRA-Ext/
    ├── img/
    └── label/
```

### Preprocessor Options

| Flag | Default | Description |
|------|---------|-------------|
| `--busi_dir` | ... | Path to raw BUSI dataset |
| `--bus_dir` | ... | Path to raw BUS dataset |
| `--busbra_dir` | ... | Path to raw BUSBRA dataset |
| `--output_dir` | `./data/processed` | Output directory |
| `--train_ratio` | `0.7` | Fraction for training |
| `--val_ratio` | `0.15` | Fraction for validation |
| `--seed` | `42` | Shuffle seed for reproducible splits |

**What the preprocessor does:**
- **BUSI:** Reads images from `benign/` and `malignant/` folders only (the `normal` folder is excluded entirely since those are no-lesion samples); merges multiple mask files (`_mask.png`, `_mask_1.png`, etc.) via union; converts to grayscale; binarizes masks to 0/1.
- **BUS:** Reads from `original/` and `GT/`; converts 0/255 masks to 0/1.
- **BUSBRA:** Reads from `Images/` and `Masks/`; pairs `bus_XXXX-x.png` with `mask_XXXX-x.png`; binarizes masks to 0/1.

## Step 2: Train AutoSAMUS

AutoSAMUS extends SAMUS by adding a `prompt_generator` (cross-attention prompt embedding generator) and a `feature_adapter` (4-layer CNN). During training, **only these two modules are trained** -- the image encoder, prompt encoder, and mask decoder remain frozen from the SAMUS checkpoint.

### Training on BUSI

```bash
python train.py \
    --modelname AutoSAMUS \
    --task BUSI_EXT \
    --batch_size 8 \
    --base_lr 0.0001 \
    -keep_log True
```

### Training on BUS

```bash
python train.py \
    --modelname AutoSAMUS \
    --task BUS_EXT \
    --batch_size 8 \
    --base_lr 0.0001 \
    -keep_log True
```

### Training on BUSBRA

```bash
python train.py \
    --modelname AutoSAMUS \
    --task BUSBRA_EXT \
    --batch_size 8 \
    --base_lr 0.0001 \
    -keep_log True
```

### Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--modelname` | `SAMUS` | Model type: `SAMUS` or `AutoSAMUS` |
| `--task` | `US30K` | Dataset config: `BUSI_EXT`, `BUS_EXT`, or `BUSBRA_EXT` |
| `--batch_size` | `8` | Batch size per GPU |
| `--base_lr` | `0.0005` | Learning rate (use `0.0001` for AutoSAMUS) |
| `--warmup` | `False` | Enable learning rate warmup |
| `--warmup_period` | `250` | Number of warmup iterations |
| `-keep_log` | `False` | Save loss/dice logs to TensorBoard |
| `--n_gpu` | `1` | Number of GPUs |

### Training Config (in `utils/config.py`)

Each task (`BUSI_EXT`, `BUS_EXT`, `BUSBRA_EXT`) has a config class that sets:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `data_path` | `./data/processed` | Must match `--output_dir` from preprocessing |
| `load_path` | `./checkpoints/SAMUS.pth` | SAMUS checkpoint for AutoSAMUS initialization |
| `epochs` | `200` | Total training epochs |
| `learning_rate` | `1e-4` | Used when `--base_lr` not specified |
| `eval_mode` | `mask_slice` | Evaluation mode for binary segmentation |
| `save_path` | `./checkpoints/{TASK}/` | Where best/periodic checkpoints are saved |

> **Important:** The `load_path` in the config must point to the downloaded SAMUS checkpoint (`./checkpoints/SAMUS.pth`). This is how AutoSAMUS initializes its shared weights (image encoder, prompt encoder, mask decoder) before training the `prompt_generator` and `feature_adapter`.

### What Happens During Training

1. `_build_samus()` constructs the AutoSAMUS model
2. The SAMUS checkpoint is loaded via `load_from_pretrained()` (partial matching -- shared keys are loaded, AutoSAMUS-specific keys like `prompt_generator.*` and `feature_adapter.*` are randomly initialized)
3. All parameters in `image_encoder`, `prompt_encoder`, and `mask_decoder` are **frozen**
4. Only `prompt_generator` and `feature_adapter` are trained
5. Best model is saved based on validation Dice score

### Training Output

```
checkpoints/BUSI_EXT/
├── AutoSAMUS_01151430_42_0.8234.pth   # Best model (date_epoch_dice)
├── AutoSAMUS__0.pth                   # Periodic save (epoch 0)
└── AutoSAMUS__199.pth                 # Final epoch
```

## Step 3: Run Inference

The inference script supports both **SAMUS** (with the official checkpoint) and **AutoSAMUS** (with a trained checkpoint).

### With AutoSAMUS (trained checkpoint)

```bash
python inference_autosamus.py \
    --data_path ./data/processed \
    --dataset Breast-BUSI-Ext \
    --checkpoint ./checkpoints/BUSI_EXT/AutoSAMUS_best.pth \
    --modelname AutoSAMUS \
    --output_dir ./results \
    --batch_size 8 \
    --visualize
```

### With SAMUS (official checkpoint, no training needed)

```bash
python inference_autosamus.py \
    --data_path ./data/processed \
    --dataset Breast-BUSI-Ext \
    --checkpoint ./checkpoints/SAMUS.pth \
    --modelname SAMUS \
    --output_dir ./results \
    --batch_size 8 \
    --visualize
```

### Run on CPU (if no GPU available)

```bash
python inference_autosamus.py \
    --data_path ./data/processed \
    --dataset Breast-BUSI-Ext \
    --checkpoint ./checkpoints/SAMUS.pth \
    --modelname SAMUS \
    --device cpu \
    --batch_size 4 \
    --visualize
```

### Inference Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data_path` | `./data/processed` | Root of preprocessed data |
| `--dataset` | (required) | `Breast-BUSI-Ext`, `Breast-BUS-Ext`, or `Breast-BUSBRA-Ext` |
| `--checkpoint` | (required) | Path to `.pth` file |
| `--modelname` | `AutoSAMUS` | `SAMUS` or `AutoSAMUS` |
| `--output_dir` | `./results` | Where to save outputs |
| `--batch_size` | `8` | Batch size |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--visualize` | off | Add `--visualize` to save overlay images |
| `--encoder_input_size` | `256` | Encoder input resolution |
| `--low_image_size` | `128` | Low-res mask resolution |
| `--seed` | `300` | Random seed |

## Step 4: View Results

### Output Structure

```
results/
├── Breast-BUSI-Ext/
│   ├── results.json           # Full results (summary + per-image)
│   ├── per_image_metrics.csv  # Per-image metrics table
│   └── vis/
│       ├── overlay/           # [Original | GT | Prediction] images
│       └── pred_masks/        # Raw binary prediction masks (0/255)
├── Breast-BUS-Ext/
│   └── ...
└── Breast-BUSBRA-Ext/
    └── ...
```

### Console Output Example

```
======================================================================
  Results for Breast-BUSI-Ext
======================================================================
  Images evaluated : 66
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

## Quick-Start: Full Pipeline (Copy & Paste)

```bash
cd "/Volumes/Autzoko/MS Thesis/SAMUS"
conda activate SAMUS

# ── Step 1: Preprocess with train/val/test splits ──
python preprocess_datasets.py \
    --busi_dir   "/Users/langtian/Desktop/NYU/MS Thesis/3D SAM Foundation Model/Med3D/Data/BUSI" \
    --bus_dir    "/Users/langtian/Desktop/NYU/MS Thesis/3D SAM Foundation Model/Med3D/Data/BUS" \
    --busbra_dir "/Users/langtian/Desktop/NYU/MS Thesis/3D SAM Foundation Model/Med3D/Data/BUSBRA" \
    --output_dir ./data/processed \
    --train_ratio 0.7 \
    --val_ratio 0.15

# ── Step 2: Train AutoSAMUS on BUSI ──
python train.py --modelname AutoSAMUS --task BUSI_EXT --batch_size 8 --base_lr 0.0001 -keep_log True

# ── Step 3: Train AutoSAMUS on BUS ──
python train.py --modelname AutoSAMUS --task BUS_EXT --batch_size 8 --base_lr 0.0001 -keep_log True

# ── Step 4: Train AutoSAMUS on BUSBRA ──
python train.py --modelname AutoSAMUS --task BUSBRA_EXT --batch_size 8 --base_lr 0.0001 -keep_log True

# ── Step 5: Inference with trained AutoSAMUS ──
# (Replace checkpoint paths with your best checkpoint from training)
python inference_autosamus.py \
    --data_path ./data/processed \
    --dataset Breast-BUSI-Ext \
    --checkpoint ./checkpoints/BUSI_EXT/YOUR_BEST_CHECKPOINT.pth \
    --modelname AutoSAMUS \
    --output_dir ./results \
    --visualize

python inference_autosamus.py \
    --data_path ./data/processed \
    --dataset Breast-BUS-Ext \
    --checkpoint ./checkpoints/BUS_EXT/YOUR_BEST_CHECKPOINT.pth \
    --modelname AutoSAMUS \
    --output_dir ./results \
    --visualize

python inference_autosamus.py \
    --data_path ./data/processed \
    --dataset Breast-BUSBRA-Ext \
    --checkpoint ./checkpoints/BUSBRA_EXT/YOUR_BEST_CHECKPOINT.pth \
    --modelname AutoSAMUS \
    --output_dir ./results \
    --visualize
```

## SAMUS vs AutoSAMUS

| | SAMUS | AutoSAMUS |
|---|-------|-----------|
| **Prompt type** | Manual (click point + bbox) | Automatic (learned prompt generator) |
| **Extra modules** | None | `prompt_generator` + `feature_adapter` |
| **Trainable params** | Image encoder adapters | `prompt_generator` + `feature_adapter` only |
| **Official checkpoint** | Available | Must be trained from SAMUS checkpoint |
| **Use case** | Direct inference with official weights | Fine-tuned inference after training |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'monai'` | `pip install monai==1.1.0` |
| `ModuleNotFoundError: No module named 'hausdorff'` | `pip install hausdorff==0.2.6` |
| `ModuleNotFoundError: No module named 'einops'` | `pip install einops==0.6.1` |
| `ModuleNotFoundError: No module named 'batchgenerators'` | `pip install batchgenerators==0.25` |
| `ModuleNotFoundError: No module named 'numba'` | `pip install numba>=0.57` |
| `CUDA out of memory` | Reduce `--batch_size` to 2 or 1 |
| `Split file not found` | Run `preprocess_datasets.py` first |
| `RuntimeError: ... expected ... got ...` on MPS | Use `--device cpu` (MPS may lack some ops) |
| Checkpoint loading fails with key mismatch | The loader handles `module.` prefix stripping and partial loading automatically. Ensure the checkpoint is the official SAMUS release. |
| Training loss not decreasing | Check that `load_path` in config points to `./checkpoints/SAMUS.pth` |
| Empty train/val splits | Re-run `preprocess_datasets.py` with `--train_ratio 0.7 --val_ratio 0.15` |
