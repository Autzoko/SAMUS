# Training & Inference Guide: SAMUS / AutoSAMUS

Complete guide covering:
1. **Training AutoSAMUS on downstream tasks** (BUSI, BUS, BUSBRA)
2. **Running inference** with SAMUS or AutoSAMUS

---

## Background: SAMUS vs AutoSAMUS

### Architecture Comparison

| | SAMUS | AutoSAMUS- | AutoSAMUS (full) |
|---|-------|------------|-------------------|
| **Prompt type** | Manual (click point) | Automatic (APG) | Automatic (APG) |
| **Extra modules** | None | `prompt_generator` + `feature_adapter` | `prompt_generator` + `feature_adapter` |
| **Frozen** | prompt_encoder, mask_decoder | image_encoder, prompt_encoder, mask_decoder | prompt_encoder, mask_decoder |
| **Trainable** | Image encoder adapters, CNN branch, upneck | APG + feature_adapter only (~8.86M) | APG + feature_adapter + SAMUS learnable parts |
| **Flag** | `--modelname SAMUS` | `--modelname AutoSAMUS` | `--modelname AutoSAMUS --unfreeze_encoder` |

### Training Pipeline (from the paper)

The paper describes a **two-stage** process:

1. **Stage 1: Train SAMUS on US30K** -- Trains the image encoder adaptations (CNN branch, adapters, rel_pos, upneck) on the full US30K dataset (~30K images, 7 datasets). Uses manual click prompts. **This is already done -- the official SAMUS checkpoint is provided.**

2. **Stage 2: Fine-tune AutoSAMUS on individual downstream tasks** -- Loads trained SAMUS weights into AutoSAMUS. The APG (`prompt_generator` + `feature_adapter`) replaces the manual prompt encoder and is trained from scratch on each target dataset separately.

> **Important:** The paper does NOT train AutoSAMUS on US30K. AutoSAMUS is only fine-tuned on individual downstream tasks (DDTI, UDIAT, HMC-QU in the paper; BUSI, BUS, BUSBRA in our case). This is because US30K contains CAMUS multi-class data where the same image appears with 3 different target masks (LV, MYO, LA) -- AutoSAMUS has no prompt to disambiguate which structure to segment.

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

**SAMUS pre-trained checkpoint** (required for AutoSAMUS training):

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

---

## Part A: Train AutoSAMUS

Two training methods are available. Choose based on your goal:

| | Method 1: Per-Dataset | Method 2: US30K (no CAMUS) |
|---|---|---|
| **Task flag** | `--task BUSI_EXT` / `BUS_EXT` / `BUSBRA_EXT` | `--task US30K_NOCAMUS` |
| **Training data** | Single target dataset (~100-500 images) | All US30K except CAMUS (~6,690 images) |
| **Best for** | Max performance on that specific dataset | Zero-shot inference on unseen datasets |
| **Generalization** | Low (task-specific) | Higher (multi-domain) |

### Method 1: Train on a Single Downstream Dataset

#### Step 1a: Preprocess Target Datasets

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

#### Step 1b: Train AutoSAMUS- (APG only, recommended first)

Only the `prompt_generator` (~8.49M) + `feature_adapter` (~0.37M) are trainable. The entire SAMUS encoder/decoder is frozen.

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

#### Step 1c (Optional): Train AutoSAMUS full (APG + encoder parts)

Add `--unfreeze_encoder` to also fine-tune the SAMUS learnable encoder parts (CNN branch, adapters, rel_pos embeddings, upneck). Uses differential learning rates: encoder parts get 10x lower lr to prevent corrupting the pre-trained weights.

```bash
python train.py \
    --modelname AutoSAMUS \
    --task BUSI_EXT \
    --unfreeze_encoder \
    --batch_size 8 \
    --base_lr 0.0001 \
    -keep_log True
```

---

### Method 2: Train on US30K (minus CAMUS)

This trains AutoSAMUS on all US30K datasets except CAMUS, creating a more general model. CAMUS is excluded because it has multi-class masks (LV/MYO/LA for the same image) which AutoSAMUS cannot disambiguate without a manual prompt.

Remaining datasets after removing CAMUS (~6,690 train / ~1,098 val / ~2,402 test):
- ThyroidNodule-TN3K, ThyroidNodule-TG3K
- Breast-BUSI
- Aponeurosis-FALLMUDRyan, Fascicle-FALLMUDRyan
- Echocardiography-HMCQU (test only)
- and others (test only)

#### Step 2a: Generate Filtered Split Files

```bash
python prepare_us30k_no_camus.py --data_path ./US30K
```

This creates `train_no_camus.txt`, `val_no_camus.txt`, `test_no_camus.txt` in `US30K/MainPatient/`. Only needs to be run once.

#### Step 2b: Train AutoSAMUS- (APG only)

```bash
python train.py \
    --modelname AutoSAMUS \
    --task US30K_NOCAMUS \
    --data_path ./US30K \
    --batch_size 8 \
    --base_lr 0.0001 \
    -keep_log True
```

#### Step 2c (Optional): Train AutoSAMUS full

```bash
python train.py \
    --modelname AutoSAMUS \
    --task US30K_NOCAMUS \
    --data_path ./US30K \
    --unfreeze_encoder \
    --batch_size 8 \
    --base_lr 0.0001 \
    -keep_log True
```

---

### What happens during training (both methods)

1. `get_model("AutoSAMUS", ...)` builds the model and loads SAMUS checkpoint via `load_from_pretrained()` (shared keys loaded; APG + feature_adapter randomly initialized)
2. AutoSAMUS freezes `image_encoder`, `prompt_encoder`, `mask_decoder`
3. Only `prompt_generator` + `feature_adapter` are trained (AutoSAMUS-)
4. With `--unfreeze_encoder`: also unfreezes SAMUS adapters, CNN branch, rel_pos, upneck (AutoSAMUS full)
5. With `--unfreeze_encoder`: differential LR applied -- encoder params at `base_lr * 0.1`, new params at `base_lr`
6. Best checkpoint saved based on validation Dice score

### Training output

```
# Method 1
checkpoints/BUSI_EXT/
├── AutoSAMUS_02091430_42_0.7856.pth   # Best model (date_epoch_dice)
├── AutoSAMUS__0.pth                    # Epoch 0 save
└── AutoSAMUS__199.pth                  # Final epoch

# Method 2
checkpoints/US30K_NOCAMUS/
├── AutoSAMUS_02091430_42_0.7856.pth
├── AutoSAMUS__0.pth
└── AutoSAMUS__199.pth
```

### Config reference (`utils/config.py`)

| Parameter | BUSI_EXT | BUS_EXT | BUSBRA_EXT | US30K_NOCAMUS |
|-----------|----------|---------|------------|---------------|
| `data_path` | `./data/processed` | `./data/processed` | `./data/processed` | `./US30K` |
| `load_path` | `./checkpoints/SAMUS.pth` | `./checkpoints/SAMUS.pth` | `./checkpoints/SAMUS.pth` | `./checkpoints/SAMUS.pth` |
| `epochs` | 200 | 200 | 200 | 200 |
| `classes` | 2 | 2 | 2 | 2 |
| `eval_mode` | `mask_slice` | `mask_slice` | `mask_slice` | `mask_slice` |

> Paper uses lr=1e-4 and 400 epochs for downstream fine-tuning. Our configs use 200 epochs as default; increase with `--epochs` if needed.

---

## Part B: Run Inference

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
| `--task` | `US30K` | Config name: `US30K`, `TN3K`, `BUSI`, `CAMUS`, `BUSI_EXT`, `BUS_EXT`, `BUSBRA_EXT`, `US30K_NOCAMUS` |
| `--data_path` | (from config) | Override dataset root path |
| `--load_path` | (from config) | Override checkpoint path for model initialization |
| `--unfreeze_encoder` | off | Unfreeze SAMUS learnable parts for full AutoSAMUS |
| `--batch_size` | `8` | Batch size per GPU |
| `--base_lr` | `0.0005` | Learning rate (use `0.0001` for downstream fine-tuning) |
| `--n_gpu` | `1` | Number of GPUs |
| `--warmup` | `False` | Enable learning rate warmup |
| `--warmup_period` | `250` | Warmup iterations |
| `-keep_log` | `False` | Save TensorBoard logs |
| `--sam_ckpt` | `checkpoints/sam_vit_b_01ec64.pth` | SAM checkpoint (for training SAMUS from scratch) |

---

## Quick-Start: Complete Pipeline

### Option A: Per-dataset training (best accuracy on target dataset)

```bash
cd "/path/to/SAMUS"
conda activate SAMUS

# -- 1. Preprocess target datasets --
python preprocess_datasets.py \
    --busi_dir   "/path/to/BUSI" \
    --bus_dir    "/path/to/BUS" \
    --busbra_dir "/path/to/BUSBRA" \
    --output_dir ./data/processed \
    --train_ratio 0.7 \
    --val_ratio 0.15

# -- 2. Train AutoSAMUS on downstream tasks (using SAMUS checkpoint) --
python train.py --modelname AutoSAMUS --task BUSI_EXT --batch_size 8 --base_lr 0.0001 -keep_log True
python train.py --modelname AutoSAMUS --task BUS_EXT --batch_size 8 --base_lr 0.0001 -keep_log True
python train.py --modelname AutoSAMUS --task BUSBRA_EXT --batch_size 8 --base_lr 0.0001 -keep_log True

# -- 3. Run inference --
python inference_autosamus.py \
    --data_path ./data/processed \
    --dataset Breast-BUSI-Ext \
    --checkpoint ./checkpoints/BUSI_EXT/YOUR_BEST.pth \
    --modelname AutoSAMUS \
    --output_dir ./results \
    --visualize
```

### Option B: US30K foundation model (better generalization to unseen datasets)

```bash
cd "/path/to/SAMUS"
conda activate SAMUS

# -- 1. Generate filtered split files (one-time) --
python prepare_us30k_no_camus.py --data_path ./US30K

# -- 2. Train AutoSAMUS on US30K minus CAMUS --
python train.py --modelname AutoSAMUS --task US30K_NOCAMUS --data_path ./US30K --batch_size 8 --base_lr 0.0001 -keep_log True

# -- 3. Run inference on any dataset --
python inference_autosamus.py \
    --data_path ./data/processed \
    --dataset Breast-BUSI-Ext \
    --checkpoint ./checkpoints/US30K_NOCAMUS/YOUR_BEST.pth \
    --modelname AutoSAMUS \
    --output_dir ./results \
    --visualize
```

---

## Part C: Comprehensive Evaluation

`evaluate.py` evaluates one or more checkpoints on both in-distribution (BUSI from US30K) and out-of-distribution (BUSBRA, unseen) data, producing a comparison table.

### Evaluate SAMUS baseline

```bash
python evaluate.py \
    --checkpoints ./checkpoints/SAMUS.pth \
    --labels SAMUS_baseline \
    --modelname SAMUS \
    --busi_data_path ./US30K \
    --busbra_raw "/path/to/BUSBRA" \
    --output_dir ./eval_results \
    --visualize
```

### Compare multiple AutoSAMUS checkpoints

```bash
python evaluate.py \
    --checkpoints \
        ./checkpoints/BUSI_EXT/best.pth \
        ./checkpoints/US30K_NOCAMUS/best.pth \
    --labels "BUSI_APG" "US30K_APG" \
    --modelname AutoSAMUS \
    --busi_data_path ./US30K \
    --busbra_raw "/path/to/BUSBRA" \
    --output_dir ./eval_results
```

### Evaluation output

```
eval_results/
├── summary.json              # All results in structured JSON
├── comparison.csv            # Comparison table (checkpoint x dataset x metrics)
├── .busbra_eval/             # Preprocessed BUSBRA data (auto-generated, cached)
├── BUSI_APG/
│   ├── Breast-BUSI/
│   │   ├── per_image_metrics.csv
│   │   └── vis/              # If --visualize
│   └── Breast-BUSBRA-Raw/
│       ├── per_image_metrics.csv
│       └── vis/
└── US30K_APG/
    ├── Breast-BUSI/
    └── Breast-BUSBRA-Raw/
```

### Evaluate options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoints` | (required) | One or more `.pth` files |
| `--labels` | (auto) | Human-readable name per checkpoint |
| `--modelname` | `AutoSAMUS` | `SAMUS` or `AutoSAMUS` |
| `--busi_data_path` | `./US30K` | Path containing Breast-BUSI test data |
| `--no_busi` | off | Skip BUSI evaluation |
| `--busbra_raw` | (none) | Path to raw BUSBRA dataset (auto-preprocessed) |
| `--output_dir` | `./eval_results` | Where to save results |
| `--batch_size` | `8` | Batch size |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--visualize` | off | Save overlay images |

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
| `Split file not found` | Run `preprocess_datasets.py` first or check `--data_path` |
| `RuntimeError: expected ... got ...` on MPS | Use `--device cpu` (MPS may lack some ops) |
| `FileNotFoundError` on checkpoint | Check `--load_path` points to `./checkpoints/SAMUS.pth` |
| Checkpoint key mismatch | `load_from_pretrained()` handles `module.` prefix stripping and partial loading automatically |
| Training loss not decreasing | Verify config `load_path` points to `./checkpoints/SAMUS.pth` |
| Low val dice on US30K | AutoSAMUS should NOT be trained on US30K (CAMUS multi-class conflict); use downstream task configs instead |
| Empty train/val splits | Re-run `preprocess_datasets.py` with `--train_ratio 0.7 --val_ratio 0.15` |
| `Split file not found` for US30K_NOCAMUS | Run `python prepare_us30k_no_camus.py --data_path ./US30K` first |
