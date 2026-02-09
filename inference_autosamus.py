"""
AutoSAMUS Inference Script for BUSI and BUS Datasets.

End-to-end pipeline:
  1. Load preprocessed data (output of preprocess_datasets.py)
  2. Load pre-trained AutoSAMUS checkpoint
  3. Run inference (no manual prompts needed — AutoSAMUS generates them automatically)
  4. Evaluate with Dice, IoU, Hausdorff Distance, Accuracy, Sensitivity, Specificity
  5. Save visualization overlays and raw prediction masks

Usage:
  python inference_autosamus.py \
      --data_path ./data/processed \
      --dataset Breast-BUSI-Ext \
      --checkpoint ./checkpoints/AutoSAMUS.pth \
      --output_dir ./results

See RUN_INFERENCE.md for full instructions.
"""

import os
import sys
import argparse
import json
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch.utils.data import DataLoader
from torch.autograd import Variable

# SAMUS project imports
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.generate_prompts import get_click_prompt
from utils.loss_functions.sam_loss import get_criterion
import utils.metrics as metrics
from hausdorff import hausdorff_distance


# ─────────────────────────────────────────────────────────────────────────────
# Configuration helper: mimics the Config classes in utils/config.py
# ─────────────────────────────────────────────────────────────────────────────
class InferenceConfig:
    """Minimal config object compatible with the SAMUS evaluation pipeline."""

    def __init__(self, args):
        self.data_path = args.data_path
        self.data_subpath = os.path.join(args.data_path, args.dataset)
        self.save_path = args.output_dir
        self.result_path = os.path.join(args.output_dir, "vis")
        self.load_path = args.checkpoint

        self.classes = 2
        self.img_size = 256
        self.batch_size = args.batch_size
        self.test_split = f"test-{args.dataset}"
        self.crop = None

        self.device = args.device
        self.gray = "yes"
        self.img_channel = 1
        self.eval_mode = "mask_slice"
        self.pre_trained = True
        self.mode = "val"  # triggers full metric reporting
        self.visual = args.visualize
        self.modelname = "AutoSAMUS"


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────
def save_visualization(seg, gt, image_path, save_dir, filename, img_size=256):
    """
    Save a side-by-side visualization:
      [original image] | [GT overlay] | [prediction overlay]
    """
    os.makedirs(save_dir, exist_ok=True)

    # Read original image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        # Fallback: try grayscale
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img_bgr = cv2.resize(img_bgr, (img_size, img_size))

    # Colors (BGR)
    pred_color = np.array([244, 164, 96])   # light blue
    gt_color = np.array([144, 255, 144])     # light green

    # Prediction overlay
    pred_overlay = img_bgr.copy()
    pred_mask = seg[0, :, :]
    for c in range(3):
        pred_overlay[:, :, c] = np.where(
            pred_mask == 1, pred_color[c], pred_overlay[:, :, c]
        )
    pred_blend = cv2.addWeighted(img_bgr, 0.4, pred_overlay, 0.6, 0)

    # GT overlay
    gt_overlay = img_bgr.copy()
    gt_mask = gt[0, :, :]
    for c in range(3):
        gt_overlay[:, :, c] = np.where(
            gt_mask == 1, gt_color[c], gt_overlay[:, :, c]
        )
    gt_blend = cv2.addWeighted(img_bgr, 0.4, gt_overlay, 0.6, 0)

    # Concatenate: original | GT | prediction
    canvas = np.hstack([img_bgr, gt_blend, pred_blend])
    cv2.imwrite(os.path.join(save_dir, filename), canvas)


def save_pred_mask(seg, save_dir, filename):
    """Save raw binary prediction mask as 0/255 PNG."""
    os.makedirs(save_dir, exist_ok=True)
    mask = (seg[0, :, :] * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, filename), mask)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(model, dataloader, opt, args, criterion):
    """
    Run inference and compute per-image metrics.
    Returns a list of per-image result dicts and aggregated statistics.
    """
    model.eval()
    device = opt.device

    results = []
    all_dice, all_iou, all_hd, all_acc, all_se, all_sp = [], [], [], [], [], []
    total_loss = 0.0
    total_time = 0.0
    num_batches = 0

    for batch_idx, datapack in enumerate(dataloader):
        imgs = datapack['image'].to(dtype=torch.float32, device=device)
        masks_low = datapack['low_mask'].to(dtype=torch.float32, device=device)
        label = datapack['label'].to(dtype=torch.float32, device=device)
        image_filenames = datapack['image_name']

        pt = get_click_prompt(datapack, opt)

        with torch.no_grad():
            t0 = time.time()
            pred = model(imgs, pt)
            total_time += time.time() - t0

        # Loss
        val_loss = criterion(pred, masks_low)
        total_loss += val_loss.item()
        num_batches += 1

        # Post-process predictions
        gt = label.detach().cpu().numpy()[:, 0, :, :]  # (B, H, W)
        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()         # (B, C, H, W)
        seg = predict[:, 0, :, :] > 0.5                  # (B, H, W)

        b, h, w = seg.shape
        for j in range(b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j + 1] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j + 1] == 1] = 255

            dice_i = metrics.dice_coefficient(pred_i, gt_i)
            iou_i, acc_i, se_i, sp_i = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            hd_i = hausdorff_distance(pred_i[0], gt_i[0], distance="manhattan")

            result = {
                "filename": image_filenames[j],
                "dice": float(dice_i),
                "iou": float(iou_i),
                "hd": float(hd_i),
                "acc": float(acc_i),
                "se": float(se_i),
                "sp": float(sp_i),
            }
            results.append(result)
            all_dice.append(dice_i)
            all_iou.append(iou_i)
            all_hd.append(hd_i)
            all_acc.append(acc_i)
            all_se.append(se_i)
            all_sp.append(sp_i)

            # Visualization
            if opt.visual:
                img_subpath = os.path.join(opt.data_subpath, "img", image_filenames[j])
                vis_dir = os.path.join(opt.result_path, "overlay")
                save_visualization(seg[j:j + 1], gt[j:j + 1], img_subpath, vis_dir,
                                   image_filenames[j], img_size=h)
                mask_dir = os.path.join(opt.result_path, "pred_masks")
                save_pred_mask(seg[j:j + 1], mask_dir, image_filenames[j])

    # Aggregate
    n = len(results)
    fps = n / total_time if total_time > 0 else 0
    stats = {
        "num_images": n,
        "fps": fps,
        "mean_loss": total_loss / max(num_batches, 1),
        "dice_mean": np.mean(all_dice) * 100,
        "dice_std": np.std(all_dice) * 100,
        "iou_mean": np.mean(all_iou) * 100,
        "iou_std": np.std(all_iou) * 100,
        "hd_mean": np.mean(all_hd),
        "hd_std": np.std(all_hd),
        "acc_mean": np.mean(all_acc) * 100,
        "acc_std": np.std(all_acc) * 100,
        "se_mean": np.mean(all_se) * 100,
        "se_std": np.std(all_se) * 100,
        "sp_mean": np.mean(all_sp) * 100,
        "sp_std": np.std(all_sp) * 100,
    }
    return results, stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AutoSAMUS Inference on BUSI/BUS")
    parser.add_argument("--data_path", type=str, default="./data/processed",
                        help="Root of the SAMUS-formatted dataset")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["Breast-BUSI-Ext", "Breast-BUS-Ext", "Breast-BUSBRA-Ext"],
                        help="Which dataset to run inference on")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the AutoSAMUS .pth checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda or cpu (auto-detected if omitted)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save overlay visualizations")
    parser.add_argument("--encoder_input_size", type=int, default=256,
                        help="Image input size for the encoder (default: 256)")
    parser.add_argument("--low_image_size", type=int, default=128,
                        help="Low-res mask size (default: 128)")
    parser.add_argument("--seed", type=int, default=300,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    # Reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    # Config
    opt = InferenceConfig(args)
    out_dir = os.path.join(args.output_dir, args.dataset)
    opt.result_path = os.path.join(out_dir, "vis")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f"  AutoSAMUS Inference")
    print(f"  Dataset     : {args.dataset}")
    print(f"  Data path   : {args.data_path}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Device      : {args.device}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Visualize   : {args.visualize}")
    print("=" * 70)

    # ── Verify data exists ───────────────────────────────────────────────
    split_file = os.path.join(args.data_path, "MainPatient",
                              f"test-{args.dataset}.txt")
    if not os.path.isfile(split_file):
        print(f"\n[ERROR] Split file not found: {split_file}")
        print("Have you run preprocess_datasets.py first?")
        sys.exit(1)

    with open(split_file) as f:
        n_samples = sum(1 for line in f if line.strip())
    print(f"\nFound {n_samples} test samples in {split_file}")

    # ── Data loader ──────────────────────────────────────────────────────
    tf_val = JointTransform2D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=opt.img_size,
        crop=None,
        p_flip=0,
        color_jitter_params=None,
        long_mask=True,
    )
    val_dataset = ImageToImage2D(
        args.data_path,
        opt.test_split,
        tf_val,
        img_size=args.encoder_input_size,
        class_id=1,
    )
    dataloader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,  # safe default for all platforms
        pin_memory=(args.device == "cuda"),
    )
    print(f"DataLoader ready: {len(val_dataset)} images, {len(dataloader)} batches\n")

    # ── Build model ──────────────────────────────────────────────────────
    # AutoSAMUS loads the full checkpoint (encoder + prompt generator +
    # decoder) directly via build_samus → _build_samus, which internally
    # calls load_from_pretrained to handle multi-GPU prefix stripping.
    print("Loading AutoSAMUS model...")

    # Create a namespace that get_model expects
    class ModelArgs:
        encoder_input_size = args.encoder_input_size
        low_image_size = args.low_image_size
        sam_ckpt = args.checkpoint  # not used for AutoSAMUS, but keep for API
        vit_name = "vit_b"

    model_args = ModelArgs()
    model = get_model("AutoSAMUS", args=model_args, opt=opt)
    model.to(args.device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters    : {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")

    # ── Loss function (for reporting val loss) ───────────────────────────
    criterion = get_criterion(modelname="AutoSAMUS", opt=opt)

    # ── Run inference + evaluation ───────────────────────────────────────
    print(f"\nRunning inference on {args.dataset}...")
    results, stats = run_evaluation(model, dataloader, opt, model_args, criterion)

    # ── Print results ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  Results for {args.dataset}")
    print("=" * 70)
    print(f"  Images evaluated : {stats['num_images']}")
    print(f"  Inference speed  : {stats['fps']:.1f} images/sec")
    print(f"  Mean loss        : {stats['mean_loss']:.4f}")
    print()
    print(f"  {'Metric':<14} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*30}")
    print(f"  {'Dice (%)':<14} {stats['dice_mean']:>8.2f} {stats['dice_std']:>8.2f}")
    print(f"  {'IoU (%)':<14} {stats['iou_mean']:>8.2f} {stats['iou_std']:>8.2f}")
    print(f"  {'HD':<14} {stats['hd_mean']:>8.2f} {stats['hd_std']:>8.2f}")
    print(f"  {'Accuracy (%)':<14} {stats['acc_mean']:>8.2f} {stats['acc_std']:>8.2f}")
    print(f"  {'Sensitivity(%)':<14} {stats['se_mean']:>8.2f} {stats['se_std']:>8.2f}")
    print(f"  {'Specificity(%)':<14} {stats['sp_mean']:>8.2f} {stats['sp_std']:>8.2f}")
    print()

    # ── Save results to JSON ─────────────────────────────────────────────
    summary = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "stats": stats,
        "per_image": results,
    }
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Detailed results saved to: {json_path}")

    # ── Save CSV summary ─────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "per_image_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("filename,dice,iou,hd,acc,se,sp\n")
        for r in results:
            f.write(f"{r['filename']},{r['dice']:.4f},{r['iou']:.4f},"
                    f"{r['hd']:.2f},{r['acc']:.4f},{r['se']:.4f},{r['sp']:.4f}\n")
    print(f"Per-image CSV saved to : {csv_path}")

    if opt.visual:
        vis_dir = os.path.join(opt.result_path, "overlay")
        mask_dir = os.path.join(opt.result_path, "pred_masks")
        print(f"Visualizations saved to: {vis_dir}")
        print(f"Prediction masks saved : {mask_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
