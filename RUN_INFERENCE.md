# Training & Inference Guide: SAMUS / AutoSAMUS

Complete guide covering:
1. **Training AutoSAMUS on US30K** (foundation model training from SAMUS checkpoint)
2. **Fine-tuning AutoSAMUS on target datasets** (BUSI, BUS, BUSBRA)
3. **Running inference** with SAMUS or AutoSAMUS

---

## Background: SAMUS vs AutoSAMUS

| | SAMUS | AutoSAMUS- | AutoSAMUS (full) |
|---|-------|------------|-------------------|
| **Prompt type** | Manual (click point) | Automatic (APG) | Automatic (APG) |
| **Extra modules** | None | `prompt_generator` + `feature_adapter` | `prompt_generator` + `feature_adapter` |
| **Frozen** | prompt_encoder, mask_decoder | image_encoder, prompt_encoder, mask_decoder | prompt_encoder, mask_decoder |
| **Trainable** | Image encoder adapters, CNN branch, upneck | APG + feature_adapter only (~8.86M) | APG + feature_adapter + SAMUS learnable parts |
| **Checkpoint** | Official release | Trained from SAMUS | Trained from SAMUS |
| **Flag** | `--modelname SAMUS` | `--modelname AutoSAMUS` | `--modelname AutoSAMUS --unfreeze_encoder` |

**Training pipeline (from the paper):**
1. Train SAMUS on US30K (already done -- official checkpoint provided)
2. Load trained SAMUS into AutoSAMUS (shared weights: image encoder, prompt encoder, mask decoder)
3. Fine-tune APG (`prompt_generator` + `feature_adapter`) on downstream tasks or the full US30K

---

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
cd "/path/to/SAMUS"
pip install -r requirements.txt
```

> **Note:** PyTorch is deliberately **not** listed in `requirements.txt` because CUDA wheels require the special `--index-url` flag. Always install PyTorch first (Step A), then run `pip install -r requirements.txt` (Step B).

### 2. Download Checkpoints

**SAMUS pre-trained checkpoint** (required for all AutoSAMUS training):

- [SAMUS checkpoint (Google Drive)](https://drive.google.com/file/d/1nQjMAvbPeolNpCxQyU_HTiOiB5704pkH/view?usp=sharing)

```bash
mkdir -p checkpoints
mv ~/Downloads/SAMUS.pth checkpoints/SAMUS.pth
```

**SAM ViT-B checkpoint** (only needed if training SAMUS from scratch):

- [sam_vit_b_01ec64.pth](https://github.com/facebookresearch/segment-anything)

```bash
mv ~/Downloads/sam_vit_b_01ec64.pth checkpoints/sam_vit_b_01ec64.pth
```

### 3. Download US30K Dataset

**Preprocessed US30K** (~30K ultrasound images, 7 public datasets):

- [US30K (Google Drive)](https://drive.google.com/file/d/13MUXQIyCXqNscIKTLRIEHKtpak6MJby_/view?usp=sharing)

Extract and note the path. Expected structure:

```
US30K/
├── MainPatient/
│   ├── class.json                          # {"Breast-BUSI": 2, "ThyroidNodule-TN3K": 2, ...}
│   ├── train.txt                           # format: class_id/dataset/filename
│   ├── val.txt
│   ├── test.txt
│   ├── train-Breast-BUSI.txt               # per-dataset splits
│   ├── val-Breast-BUSI.txt
│   ├── test-Breast-BUSI.txt
│   ├── train-ThyroidNodule-TN3K.txt
│   └── ...
├── Breast-BUSI/
│   ├── img/                                # grayscale PNGs
│   └── label/                              # binary mask PNGs
├── Breast-UDIAT/
│   ├── img/
│   └── label/
├── ThyroidNodule-TN3K/
│   ├── img/
│   └── label/
├── ThyroidNodule-DDTI/
│   ├── img/
│   └── label/
├── ThyroidNodule-TG3K/
│   ├── img/
│   └── label/
├── Echocardiography-CAMUS/
│   ├── img/
│   └── label/
└── Echocardiography-HMCQU/
    ├── img/
    └── label/
```

The split files use two formats:
- **Train/Val:** `class_id/dataset_folder/image_name` (e.g., `1/Breast-BUSI/benign0046`)
- **Test:** `dataset_folder/image_name` (e.g., `Breast-BUSI/benign0031`)

---

## Part A: Train AutoSAMUS on US30K

This trains AutoSAMUS as a foundation model on the full US30K dataset, starting from the SAMUS checkpoint.

### AutoSAMUS- (APG only, ~8.86M trainable params)

```bash
python train.py \
    --modelname AutoSAMUS \
    --task US30K \
    --data_path /path/to/US30K \
    --load_path ./checkpoints/SAMUS.pth \
    --batch_size 8 \
    --base_lr 0.0005 \
    -keep_log True
```

### AutoSAMUS (full, APG + SAMUS learnable parts)

```bash
python train.py \
    --modelname AutoSAMUS \
    --task US30K \
    --data_path /path/to/US30K \
    --load_path ./checkpoints/SAMUS.pth \
    --unfreeze_encoder \
    --batch_size 8 \
    --base_lr 0.0005 \
    -keep_log True
```

### What happens during training

1. `get_model("AutoSAMUS", ...)` builds the model and loads SAMUS checkpoint via `load_from_pretrained()` (partial key matching -- shared keys loaded, APG + feature_adapter randomly initialized)
2. AutoSAMUS freezes `image_encoder`, `prompt_encoder`, `mask_decoder`
3. Only `prompt_generator` (~8.49M) + `feature_adapter` (~0.37M) are trained
4. With `--unfreeze_encoder`: also unfreezes SAMUS adapters, CNN branch, relative position embeddings, and upneck
5. Best checkpoint saved based on validation Dice score

### Training output

```
checkpoints/SAMUS/
├── AutoSAMUS_02091430_42_0.7856.pth   # Best model (date_epoch_dice)
├── AutoSAMUS__0.pth                    # Epoch 0 save
└── AutoSAMUS__199.pth                  # Final epoch
```

### Config reference (`Config_US30K` in `utils/config.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `data_path` | `../../dataset/SAMUS/` | Override with `--data_path` |
| `load_path` | placeholder | Override with `--load_path ./checkpoints/SAMUS.pth` |
| `train_split` | `train` | Reads `MainPatient/train.txt` |
| `val_split` | `val` | Reads `MainPatient/val.txt` |
| `epochs` | 200 | |
| `classes` | 2 | Binary segmentation |
| `eval_mode` | `mask_slice` | Slice-level evaluation |
| `save_path` | `./checkpoints/SAMUS/` | |

---

## Part B: Fine-tune AutoSAMUS on Target Datasets (BUSI / BUS / BUSBRA)

### Step 1: Preprocess Target Datasets

```bash
python preprocess_datasets.py \
    --busi_dir   "/path/to/BUSI" \
    --bus_dir    "/path/to/BUS" \
    --busbra_dir "/path/to/BUSBRA" \
    --output_dir ./data/processed \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --seed 42
```

This creates 70/15/15 train/val/test splits. Output:

```
data/processed/
├── MainPatient/
│   ├── class.json
│   ├── train.txt / val.txt / test.txt
│   ├── train-Breast-BUSI-Ext.txt
│   ├── val-Breast-BUSI-Ext.txt
│   ├── test-Breast-BUSI-Ext.txt
│   └── ... (same for BUS-Ext, BUSBRA-Ext)
├── Breast-BUSI-Ext/
│   ├── img/
│   └── label/
├── Breast-BUS-Ext/
│   ├── img/
│   └── label/
└── Breast-BUSBRA-Ext/
    ├── img/
    └── label/
```

### Step 2: Train

```bash
# BUSI
python train.py \
    --modelname AutoSAMUS \
    --task BUSI_EXT \
    --batch_size 8 \
    --base_lr 0.0001 \
    -keep_log True

# BUS
python train.py \
    --modelname AutoSAMUS \
    --task BUS_EXT \
    --batch_size 8 \
    --base_lr 0.0001 \
    -keep_log True

# BUSBRA
python train.py \
    --modelname AutoSAMUS \
    --task BUSBRA_EXT \
    --batch_size 8 \
    --base_lr 0.0001 \
    -keep_log True
```

> The `BUSI_EXT` / `BUS_EXT` / `BUSBRA_EXT` configs in `utils/config.py` already set `load_path = "./checkpoints/SAMUS.pth"`. If you trained AutoSAMUS on US30K first and want to use that checkpoint instead, pass `--load_path ./checkpoints/SAMUS/YOUR_US30K_BEST.pth`.

### Preprocessor options

| Flag | Default | Description |
|------|---------|-------------|
| `--busi_dir` | ... | Path to raw BUSI dataset |
| `--bus_dir` | ... | Path to raw BUS dataset |
| `--busbra_dir` | ... | Path to raw BUSBRA dataset |
| `--output_dir` | `./data/processed` | Output directory |
| `--train_ratio` | `0.7` | Fraction for training |
| `--val_ratio` | `0.15` | Fraction for validation |
| `--seed` | `42` | Shuffle seed for reproducible splits |

---

## Part C: Run Inference

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

### On CPU (if no GPU)

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

### Output

```
results/
├── Breast-BUSI-Ext/
│   ├── results.json           # Full results (summary + per-image)
│   ├── per_image_metrics.csv  # Per-image metrics table
│   └── vis/
│       ├── overlay/           # [Original | GT | Prediction] images
│       └── pred_masks/        # Raw binary prediction masks (0/255)
```

### Inference options

| Flag | Default | Description |
|------|---------|-------------|
| `--data_path` | `./data/processed` | Root of preprocessed data |
| `--dataset` | (required) | `Breast-BUSI-Ext`, `Breast-BUS-Ext`, or `Breast-BUSBRA-Ext` |
| `--checkpoint` | (required) | Path to `.pth` file |
| `--modelname` | `AutoSAMUS` | `SAMUS` or `AutoSAMUS` |
| `--output_dir` | `./results` | Where to save outputs |
| `--batch_size` | `8` | Batch size |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--visualize` | off | Add to save overlay images |
| `--seed` | `300` | Random seed |

---

## Training Options Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--modelname` | `SAMUS` | Model: `SAM`, `SAMUS`, or `AutoSAMUS` |
| `--task` | `US30K` | Config name: `US30K`, `TN3K`, `BUSI`, `CAMUS`, `BUSI_EXT`, `BUS_EXT`, `BUSBRA_EXT` |
| `--data_path` | (from config) | Override dataset root path |
| `--load_path` | (from config) | Override checkpoint path for model initialization |
| `--unfreeze_encoder` | off | Unfreeze SAMUS learnable parts for full AutoSAMUS |
| `--batch_size` | `8` | Batch size per GPU |
| `--base_lr` | `0.0005` | Learning rate (use `0.0001` for fine-tuning) |
| `--n_gpu` | `1` | Number of GPUs |
| `--warmup` | `False` | Enable learning rate warmup |
| `--warmup_period` | `250` | Warmup iterations |
| `-keep_log` | `False` | Save TensorBoard logs |
| `--sam_ckpt` | `checkpoints/sam_vit_b_01ec64.pth` | SAM checkpoint (for training SAMUS from scratch) |

---

## Quick-Start: Complete Pipeline

```bash
cd "/path/to/SAMUS"
conda activate SAMUS

# ── 1. Train AutoSAMUS on US30K ──
python train.py \
    --modelname AutoSAMUS \
    --task US30K \
    --data_path /path/to/US30K \
    --load_path ./checkpoints/SAMUS.pth \
    --batch_size 8 \
    --base_lr 0.0005 \
    -keep_log True

# ── 2. Preprocess target datasets ──
python preprocess_datasets.py \
    --busi_dir   "/path/to/BUSI" \
    --bus_dir    "/path/to/BUS" \
    --busbra_dir "/path/to/BUSBRA" \
    --output_dir ./data/processed \
    --train_ratio 0.7 \
    --val_ratio 0.15

# ── 3. Fine-tune AutoSAMUS on target datasets ──
python train.py --modelname AutoSAMUS --task BUSI_EXT --batch_size 8 --base_lr 0.0001 -keep_log True
python train.py --modelname AutoSAMUS --task BUS_EXT --batch_size 8 --base_lr 0.0001 -keep_log True
python train.py --modelname AutoSAMUS --task BUSBRA_EXT --batch_size 8 --base_lr 0.0001 -keep_log True

# ── 4. Run inference ──
python inference_autosamus.py \
    --data_path ./data/processed \
    --dataset Breast-BUSI-Ext \
    --checkpoint ./checkpoints/BUSI_EXT/YOUR_BEST.pth \
    --modelname AutoSAMUS \
    --output_dir ./results \
    --visualize
```

---

## Evaluation Metrics

| Metric | Formula | Range |
|--------|---------|-------|
| **Dice** | 2TP / (FP + 2TP + FN) | 0-100% |
| **IoU** | TP / (FP + TP + FN) | 0-100% |
| **Hausdorff Distance** | max surface distance (Manhattan) | 0-inf |
| **Accuracy** | (TP + TN) / (TP + FP + FN + TN) | 0-100% |
| **Sensitivity** | TP / (TP + FN) | 0-100% |
| **Specificity** | TN / (FP + TN) | 0-100% |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'monai'` | `pip install monai==1.1.0` |
| `ModuleNotFoundError: No module named 'hausdorff'` | `pip install hausdorff==0.2.6` |
| `ModuleNotFoundError: No module named 'einops'` | `pip install einops==0.6.1` |
| `ModuleNotFoundError: No module named 'batchgenerators'` | `pip install batchgenerators==0.25` |
| `ModuleNotFoundError: No module named 'numba'` | `pip install numba>=0.57` |
| `CUDA out of memory` | Reduce `--batch_size` to 2 or 1 |
| `Split file not found` | Run `preprocess_datasets.py` first (for target datasets) or check `--data_path` (for US30K) |
| `RuntimeError: expected ... got ...` on MPS | Use `--device cpu` (MPS may lack some ops) |
| `FileNotFoundError` on checkpoint | Check `--load_path` points to the SAMUS checkpoint |
| Checkpoint key mismatch | `load_from_pretrained()` handles `module.` prefix stripping and partial loading automatically |
| Training loss not decreasing | Verify `--load_path` or config `load_path` points to `./checkpoints/SAMUS.pth` |
| Empty train/val splits | Re-run `preprocess_datasets.py` with `--train_ratio 0.7 --val_ratio 0.15` |
