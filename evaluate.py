"""
Comprehensive evaluation for SAMUS / AutoSAMUS.

Evaluates one or more checkpoints on:
  1. BUSI test data (from US30K, 128 images -- in-distribution)
  2. BUSBRA (raw unseen dataset, ~1875 images -- out-of-distribution)

Usage:
  # Evaluate SAMUS baseline on both datasets
  python evaluate.py \
      --checkpoints ./checkpoints/SAMUS.pth \
      --labels SAMUS_baseline \
      --modelname SAMUS \
      --busbra_raw "/path/to/BUSBRA"

  # Compare multiple AutoSAMUS checkpoints
  python evaluate.py \
      --checkpoints \
          ./checkpoints/BUSI_EXT/best.pth \
          ./checkpoints/US30K_NOCAMUS/best.pth \
      --labels "BUSI_APG" "US30K_APG" \
      --modelname AutoSAMUS \
      --busbra_raw "/path/to/BUSBRA"

  # Skip BUSI (only test on BUSBRA)
  python evaluate.py \
      --checkpoints ./checkpoints/SAMUS.pth \
      --labels SAMUS \
      --modelname SAMUS \
      --no_busi \
      --busbra_raw "/path/to/BUSBRA"
"""

import os
import sys
import argparse
import json
import time
import random
import glob
import shutil
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader

from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.generate_prompts import get_click_prompt
import utils.metrics as metrics
from hausdorff import hausdorff_distance


# ─────────────────────────────────────────────────────────────────────────────
# BUSBRA preprocessing
# ─────────────────────────────────────────────────────────────────────────────
BUSBRA_DATASET_NAME = "Breast-BUSBRA-Raw"


def preprocess_busbra(busbra_dir, output_dir):
    """Convert raw BUSBRA dataset into SAMUS evaluation format.

    Raw BUSBRA layout:
        Images/bus_XXXX-Y.png
        Masks/mask_XXXX-Y.png

    Output (SAMUS format):
        {output_dir}/Breast-BUSBRA-Raw/img/bus_XXXX-Y.png
        {output_dir}/Breast-BUSBRA-Raw/label/bus_XXXX-Y.png
        {output_dir}/MainPatient/test-Breast-BUSBRA-Raw.txt
        {output_dir}/MainPatient/class.json
    """
    img_src = os.path.join(busbra_dir, "Images")
    mask_src = os.path.join(busbra_dir, "Masks")
    if not os.path.isdir(img_src) or not os.path.isdir(mask_src):
        print(f"[ERROR] BUSBRA directory must contain Images/ and Masks/ subdirs")
        sys.exit(1)

    out_img = os.path.join(output_dir, BUSBRA_DATASET_NAME, "img")
    out_label = os.path.join(output_dir, BUSBRA_DATASET_NAME, "label")
    mp_dir = os.path.join(output_dir, "MainPatient")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_label, exist_ok=True)
    os.makedirs(mp_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(img_src, "bus_*.png")))
    entries = []
    skipped = 0

    for img_path in image_files:
        basename = os.path.basename(img_path)          # bus_0001-l.png
        stem = os.path.splitext(basename)[0]            # bus_0001-l
        mask_name = "mask_" + stem[4:] + ".png"         # mask_0001-l.png
        mask_path = os.path.join(mask_src, mask_name)

        if not os.path.isfile(mask_path):
            skipped += 1
            continue

        # Copy image as-is (data loader handles resizing)
        shutil.copy2(img_path, os.path.join(out_img, basename))

        # Copy mask -- ensure binary (0 or 1 pixel values)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            skipped += 1
            continue
        mask[mask > 0] = 1
        cv2.imwrite(os.path.join(out_label, basename), mask)

        entries.append(f"{BUSBRA_DATASET_NAME}/{stem}")

    # Test split file (all images are test -- unseen dataset)
    split_path = os.path.join(mp_dir, f"test-{BUSBRA_DATASET_NAME}.txt")
    with open(split_path, "w") as f:
        f.write("\n".join(entries) + "\n")

    # class.json
    class_json = os.path.join(mp_dir, "class.json")
    cls = {}
    if os.path.isfile(class_json):
        with open(class_json) as f:
            cls = json.load(f)
    cls[BUSBRA_DATASET_NAME] = 2
    with open(class_json, "w") as f:
        json.dump(cls, f, indent=2)

    print(f"  Preprocessed {len(entries)} BUSBRA images ({skipped} skipped)")
    return len(entries)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation core
# ─────────────────────────────────────────────────────────────────────────────
class EvalConfig:
    """Minimal config object for the SAMUS data loader and model."""
    def __init__(self, data_path, dataset, device, load_path):
        self.data_path = data_path
        self.data_subpath = os.path.join(data_path, dataset)
        self.load_path = load_path
        self.classes = 2
        self.img_size = 256
        self.batch_size = 8
        self.test_split = f"test-{dataset}"
        self.crop = None
        self.device = device
        self.gray = "yes"
        self.img_channel = 1
        self.eval_mode = "mask_slice"
        self.pre_trained = True
        self.mode = "val"
        self.visual = False
        self.modelname = "SAM"


def build_model(modelname, checkpoint, device, encoder_input_size=256,
                low_image_size=128):
    """Build and load a SAMUS or AutoSAMUS model."""

    class ModelArgs:
        pass

    margs = ModelArgs()
    margs.encoder_input_size = encoder_input_size
    margs.low_image_size = low_image_size
    margs.vit_name = "vit_b"
    margs.sam_ckpt = checkpoint  # used by SAMUS model builder

    opt = type("Opt", (), {"load_path": checkpoint})()
    model = get_model(modelname, args=margs, opt=opt)
    model.to(device)
    model.eval()
    return model


def run_eval(model, dataloader, device, visualize_dir=None, data_subpath=None):
    """Run inference and return per-image metrics."""
    model.eval()

    # Config-like object for get_click_prompt
    class PromptOpt:
        pass
    popt = PromptOpt()
    popt.device = device

    results = []
    total_time = 0.0

    for batch_idx, datapack in enumerate(dataloader):
        imgs = datapack["image"].to(dtype=torch.float32, device=device)
        label = datapack["label"].to(dtype=torch.float32, device=device)
        image_filenames = datapack["image_name"]
        pt = get_click_prompt(datapack, popt)

        with torch.no_grad():
            t0 = time.time()
            pred = model(imgs, pt)
            total_time += time.time() - t0

        gt = label.detach().cpu().numpy()[:, 0, :, :]
        predict = torch.sigmoid(pred["masks"]).detach().cpu().numpy()
        seg = predict[:, 0, :, :] > 0.5

        b, h, w = seg.shape
        for j in range(b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j : j + 1] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j : j + 1] == 1] = 255

            dice_i = metrics.dice_coefficient(pred_i, gt_i)
            iou_i, acc_i, se_i, sp_i = metrics.sespiou_coefficient2(
                pred_i, gt_i, all=False
            )
            hd_i = hausdorff_distance(pred_i[0], gt_i[0], distance="manhattan")

            results.append(
                {
                    "filename": image_filenames[j],
                    "dice": float(dice_i),
                    "iou": float(iou_i),
                    "hd": float(hd_i),
                    "acc": float(acc_i),
                    "se": float(se_i),
                    "sp": float(sp_i),
                }
            )

            # Save visualization overlay
            if visualize_dir and data_subpath:
                _save_vis(
                    seg[j : j + 1],
                    gt[j : j + 1],
                    os.path.join(data_subpath, "img", image_filenames[j]),
                    visualize_dir,
                    image_filenames[j],
                    h,
                )

        print(
            f"\r  batch [{batch_idx + 1}/{len(dataloader)}]", end="", flush=True
        )

    print()

    n = len(results)
    dices = [r["dice"] for r in results]
    ious = [r["iou"] for r in results]
    hds = [r["hd"] for r in results]
    accs = [r["acc"] for r in results]
    ses = [r["se"] for r in results]
    sps = [r["sp"] for r in results]

    stats = {
        "num_images": n,
        "fps": n / total_time if total_time > 0 else 0,
        "dice_mean": float(np.mean(dices) * 100),
        "dice_std": float(np.std(dices) * 100),
        "iou_mean": float(np.mean(ious) * 100),
        "iou_std": float(np.std(ious) * 100),
        "hd_mean": float(np.mean(hds)),
        "hd_std": float(np.std(hds)),
        "acc_mean": float(np.mean(accs) * 100),
        "acc_std": float(np.std(accs) * 100),
        "se_mean": float(np.mean(ses) * 100),
        "se_std": float(np.std(ses) * 100),
        "sp_mean": float(np.mean(sps) * 100),
        "sp_std": float(np.std(sps) * 100),
    }
    return results, stats


def _save_vis(seg, gt, img_path, save_dir, filename, img_size):
    """Save [original | GT overlay | prediction overlay] image."""
    os.makedirs(save_dir, exist_ok=True)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img_bgr = cv2.resize(img_bgr, (img_size, img_size))

    pred_color = np.array([244, 164, 96])
    gt_color = np.array([144, 255, 144])

    pred_ov = img_bgr.copy()
    for c in range(3):
        pred_ov[:, :, c] = np.where(seg[0] == 1, pred_color[c], pred_ov[:, :, c])
    pred_blend = cv2.addWeighted(img_bgr, 0.4, pred_ov, 0.6, 0)

    gt_ov = img_bgr.copy()
    for c in range(3):
        gt_ov[:, :, c] = np.where(gt[0] == 1, gt_color[c], gt_ov[:, :, c])
    gt_blend = cv2.addWeighted(img_bgr, 0.4, gt_ov, 0.6, 0)

    canvas = np.hstack([img_bgr, gt_blend, pred_blend])
    cv2.imwrite(os.path.join(save_dir, filename), canvas)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────────────
def print_table(all_stats, datasets):
    """Print a formatted comparison table across checkpoints and datasets."""
    metric_keys = [
        ("dice", "Dice (%)"),
        ("iou", "IoU (%)"),
        ("hd", "HD"),
        ("acc", "Acc (%)"),
        ("se", "Sen (%)"),
        ("sp", "Spec (%)"),
    ]

    for ds_name, ds_type in datasets:
        print()
        print("=" * 72)
        print(f"  Dataset: {ds_name}  [{ds_type}]")
        print("=" * 72)

        # Header
        labels = list(all_stats.keys())
        col_w = 20
        header = f"  {'Metric':<12}"
        for lbl in labels:
            header += f" {lbl:>{col_w}}"
        print(header)
        print("  " + "-" * (12 + (col_w + 1) * len(labels)))

        for key, display in metric_keys:
            row = f"  {display:<12}"
            for lbl in labels:
                s = all_stats[lbl].get(ds_name)
                if s is None:
                    row += f" {'N/A':>{col_w}}"
                else:
                    mean = s[f"{key}_mean"]
                    std = s[f"{key}_std"]
                    row += f" {mean:>{col_w - 10}.2f} +/- {std:<5.2f}"
            print(row)

        # Num images
        row = f"  {'# images':<12}"
        for lbl in labels:
            s = all_stats[lbl].get(ds_name)
            if s is None:
                row += f" {'N/A':>{col_w}}"
            else:
                row += f" {s['num_images']:>{col_w}}"
        print(row)

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive SAMUS/AutoSAMUS Evaluation"
    )
    parser.add_argument(
        "--checkpoints", nargs="+", required=True, help="Checkpoint .pth file(s)"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Human-readable label for each checkpoint (must match --checkpoints count)",
    )
    parser.add_argument(
        "--modelname",
        default="AutoSAMUS",
        choices=["SAMUS", "AutoSAMUS"],
        help="Model architecture",
    )
    parser.add_argument(
        "--busi_data_path",
        default="./US30K",
        help="Path to SAMUS-format data containing Breast-BUSI (default: ./US30K)",
    )
    parser.add_argument(
        "--no_busi",
        action="store_true",
        help="Skip BUSI evaluation",
    )
    parser.add_argument(
        "--busbra_raw",
        default=None,
        help="Path to raw BUSBRA dataset (Images/ + Masks/). Preprocessed automatically.",
    )
    parser.add_argument("--output_dir", default="./eval_results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--seed", type=int, default=300)
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    # Auto-generate labels
    if args.labels is None:
        args.labels = []
        for c in args.checkpoints:
            # Use parent dir name + filename stem, e.g. "BUSI_EXT/best"
            parent = os.path.basename(os.path.dirname(c))
            stem = os.path.splitext(os.path.basename(c))[0]
            args.labels.append(f"{parent}/{stem}" if parent else stem)

    if len(args.labels) != len(args.checkpoints):
        print("[ERROR] --labels count must match --checkpoints count")
        sys.exit(1)

    # Reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Prepare evaluation datasets ──────────────────────────────────────
    datasets = []  # list of (name, data_path, type_str)

    # BUSI from US30K
    if not args.no_busi:
        busi_split = os.path.join(
            args.busi_data_path, "MainPatient", "test-Breast-BUSI.txt"
        )
        if os.path.isfile(busi_split):
            with open(busi_split) as f:
                n = sum(1 for line in f if line.strip())
            print(f"[BUSI] Found {n} test images in {args.busi_data_path}")
            datasets.append(("Breast-BUSI", args.busi_data_path, "in-distribution"))
        else:
            print(f"[BUSI] Warning: split not found at {busi_split}, skipping")

    # BUSBRA from raw
    if args.busbra_raw:
        busbra_eval_dir = os.path.join(args.output_dir, ".busbra_eval")
        # Only re-preprocess if not already done
        split_check = os.path.join(
            busbra_eval_dir,
            "MainPatient",
            f"test-{BUSBRA_DATASET_NAME}.txt",
        )
        if os.path.isfile(split_check):
            with open(split_check) as f:
                n = sum(1 for line in f if line.strip())
            print(f"[BUSBRA] Using cached preprocessing ({n} images)")
        else:
            print(f"[BUSBRA] Preprocessing raw data from {args.busbra_raw}")
            n = preprocess_busbra(args.busbra_raw, busbra_eval_dir)
        datasets.append((BUSBRA_DATASET_NAME, busbra_eval_dir, "unseen"))

    if not datasets:
        print("[ERROR] No datasets to evaluate. Provide --busi_data_path or --busbra_raw.")
        sys.exit(1)

    # ── Print evaluation plan ────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  Evaluation Plan")
    print("=" * 72)
    print(f"  Model       : {args.modelname}")
    print(f"  Device      : {args.device}")
    print(f"  Checkpoints : {len(args.checkpoints)}")
    for lbl, ckpt in zip(args.labels, args.checkpoints):
        print(f"    {lbl}: {ckpt}")
    print(f"  Datasets    : {len(datasets)}")
    for name, path, dtype in datasets:
        print(f"    {name} [{dtype}]: {path}")
    print("=" * 72)
    print()

    # ── Evaluate ─────────────────────────────────────────────────────────
    all_stats = {}   # {label: {dataset_name: stats_dict}}
    all_details = {} # {label: {dataset_name: per_image_list}}

    for ckpt_path, label in zip(args.checkpoints, args.labels):
        all_stats[label] = {}
        all_details[label] = {}
        print(f"Loading checkpoint: {label} ({ckpt_path})")
        model = build_model(args.modelname, ckpt_path, args.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        for ds_name, ds_path, ds_type in datasets:
            print(f"\n  Evaluating on {ds_name} [{ds_type}]...")

            # Data loader
            tf_val = JointTransform2D(
                img_size=256,
                low_img_size=128,
                ori_size=256,
                crop=None,
                p_flip=0,
                color_jitter_params=None,
                long_mask=True,
            )
            val_dataset = ImageToImage2D(
                ds_path, f"test-{ds_name}", tf_val, img_size=256, class_id=1
            )
            dataloader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=(args.device == "cuda"),
            )

            vis_dir = None
            data_subpath = None
            if args.visualize:
                vis_dir = os.path.join(args.output_dir, label, ds_name, "vis")
                data_subpath = os.path.join(ds_path, ds_name)

            results, stats = run_eval(
                model, dataloader, args.device, vis_dir, data_subpath
            )

            all_stats[label][ds_name] = stats
            all_details[label][ds_name] = results

            # Save per-image CSV
            csv_dir = os.path.join(args.output_dir, label, ds_name)
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, "per_image_metrics.csv")
            with open(csv_path, "w") as f:
                f.write("filename,dice,iou,hd,acc,se,sp\n")
                for r in results:
                    f.write(
                        f"{r['filename']},{r['dice']:.4f},{r['iou']:.4f},"
                        f"{r['hd']:.2f},{r['acc']:.4f},{r['se']:.4f},{r['sp']:.4f}\n"
                    )

            print(
                f"  -> Dice: {stats['dice_mean']:.2f}% +/- {stats['dice_std']:.2f}%  "
                f"IoU: {stats['iou_mean']:.2f}%  HD: {stats['hd_mean']:.2f}"
            )

        # Free model memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Print comparison table ───────────────────────────────────────────
    ds_info = [(name, dtype) for name, _, dtype in datasets]
    print_table(all_stats, ds_info)

    # ── Save summary JSON ────────────────────────────────────────────────
    summary = {
        "modelname": args.modelname,
        "device": args.device,
        "seed": args.seed,
        "checkpoints": {
            lbl: ckpt for lbl, ckpt in zip(args.labels, args.checkpoints)
        },
        "datasets": {name: dtype for name, _, dtype in datasets},
        "results": {
            lbl: {
                ds: {
                    "stats": all_stats[lbl][ds],
                    "per_image_csv": os.path.join(lbl, ds, "per_image_metrics.csv"),
                }
                for ds in all_stats[lbl]
            }
            for lbl in all_stats
        },
    }
    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {json_path}")

    # ── Save comparison CSV ──────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "comparison.csv")
    with open(csv_path, "w") as f:
        f.write("checkpoint,dataset,type,dice_mean,dice_std,iou_mean,iou_std,"
                "hd_mean,hd_std,acc_mean,acc_std,se_mean,se_std,sp_mean,sp_std,"
                "num_images\n")
        for lbl in all_stats:
            for ds_name, _, ds_type in datasets:
                s = all_stats[lbl].get(ds_name)
                if s:
                    f.write(
                        f"{lbl},{ds_name},{ds_type},"
                        f"{s['dice_mean']:.2f},{s['dice_std']:.2f},"
                        f"{s['iou_mean']:.2f},{s['iou_std']:.2f},"
                        f"{s['hd_mean']:.2f},{s['hd_std']:.2f},"
                        f"{s['acc_mean']:.2f},{s['acc_std']:.2f},"
                        f"{s['se_mean']:.2f},{s['se_std']:.2f},"
                        f"{s['sp_mean']:.2f},{s['sp_std']:.2f},"
                        f"{s['num_images']}\n"
                    )
    print(f"Comparison CSV saved to: {csv_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
