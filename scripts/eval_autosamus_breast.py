"""
Evaluate AutoSAMUS on breast ultrasound datasets with medical metrics.

Computes per-image Dice, IoU, HD95, Precision, Recall, Specificity,
then reports mean and std across the dataset.

Usage:
    cd SAMUS

    # Evaluate on all breast val sets (model trained on AllBreast)
    python scripts/eval_autosamus_breast.py \
        --checkpoint checkpoints/AllBreast/AutoSAMUS_best.pth \
        --eval-datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM

    # Evaluate on a single dataset
    python scripts/eval_autosamus_breast.py \
        --checkpoint checkpoints/AllBreast/AutoSAMUS_best.pth \
        --eval-datasets BUSI

    # Use SAMUS checkpoint directly (baseline without finetuning)
    python scripts/eval_autosamus_breast.py \
        --checkpoint checkpoints/samus_pretrained.pth \
        --eval-datasets BUSI
"""

import argparse
import json
import os
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion

warnings.filterwarnings('ignore')

# ─── Metric computation ───────────────────────────────────────────────────────

def compute_surface_distances(pred_mask, gt_mask):
    """Compute HD95 and ASD between pred and gt boundaries."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return float('inf'), float('inf')

    gt_dist_map = distance_transform_edt(~gt)
    pred_dist_map = distance_transform_edt(~pred)

    pred_boundary = pred & ~binary_erosion(pred)
    gt_boundary = gt & ~binary_erosion(gt)

    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return float('inf'), float('inf')

    pred_to_gt = gt_dist_map[pred_boundary]
    gt_to_pred = pred_dist_map[gt_boundary]

    hd95 = max(np.percentile(pred_to_gt, 95), np.percentile(gt_to_pred, 95))
    asd = (pred_to_gt.mean() + gt_to_pred.mean()) / 2.0
    return hd95, asd


def compute_metrics(pred_mask, gt_mask, smooth=1e-7):
    """Compute all medical segmentation metrics for a single image."""
    pred = pred_mask.astype(np.float32).flatten()
    gt = gt_mask.astype(np.float32).flatten()

    tp = (pred * gt).sum()
    fp = pred.sum() - tp
    fn = gt.sum() - tp
    tn = ((1 - pred) * (1 - gt)).sum()

    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    accuracy = (tp + tn + smooth) / (tp + fp + fn + tn + smooth)

    hd95, asd = compute_surface_distances(pred_mask, gt_mask)

    return {
        'Dice': dice, 'IoU': iou, 'HD95': hd95, 'ASD': asd,
        'Precision': precision, 'Recall': recall,
        'Specificity': specificity, 'Accuracy': accuracy,
    }


# ─── Data loading (SAMUS format) ──────────────────────────────────────────────

def load_split_data(data_root, split_name):
    """Load image/mask paths from a SAMUS split file.

    Returns list of (img_path, mask_path, filename).
    """
    split_file = os.path.join(data_root, 'MainPatient', f'{split_name}.txt')
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    class_file = os.path.join(data_root, 'MainPatient', 'class.json')
    with open(class_file, 'r') as f:
        class_dict = json.load(f)

    items = []
    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('/')
            class_id, sub_path, filename = parts[0], parts[1], parts[2]
            img_path = os.path.join(data_root, sub_path, 'img', filename + '.png')
            mask_path = os.path.join(data_root, sub_path, 'label', filename + '.png')
            items.append((img_path, mask_path, filename))
    return items


# ─── Model building ───────────────────────────────────────────────────────────

def build_model(checkpoint_path, device, encoder_input_size=256):
    """Build AutoSAMUS model and load checkpoint."""
    import sys
    samus_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if samus_root not in sys.path:
        sys.path.insert(0, samus_root)

    from models.segment_anything_samus_autoprompt.build_samus import autosamus_model_registry

    class Args:
        def __init__(self):
            self.encoder_input_size = encoder_input_size
            self.low_image_size = encoder_input_size // 2

    args = Args()
    model = autosamus_model_registry['vit_b'](args=args, checkpoint=checkpoint_path)
    model = model.to(device)
    model.eval()
    return model


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_dataset(model, data_root, split_name, device, img_size=256):
    """Evaluate model on a dataset split."""
    items = load_split_data(data_root, split_name)
    total = len(items)
    all_metrics = []

    for i, (img_path, mask_path, filename) in enumerate(items):
        # Load and preprocess image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        # Resize to model input size
        img_resized = cv2.resize(img, (img_size, img_size),
                                 interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (img_size, img_size),
                                  interpolation=cv2.INTER_NEAREST)

        # Binarize GT mask
        gt_mask = (mask_resized > 0).astype(np.uint8)

        if gt_mask.sum() == 0:
            continue

        # Prepare input tensor: grayscale -> (1, 1, H, W) float32 [0, 1]
        img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)

        # Dummy point prompt (AutoSAMUS ignores it, but forward signature requires it)
        pt = torch.zeros(1, 1, 2, device=device)
        pt_label = torch.ones(1, 1, dtype=torch.int, device=device)
        pt_prompt = (pt, pt_label)

        # Forward pass
        with torch.no_grad():
            outputs = model(img_tensor, pt_prompt)

        # Get prediction mask at original resolution (256x256)
        pred_logits = outputs['masks']  # (1, 1, 256, 256)
        pred_prob = torch.sigmoid(pred_logits)
        pred_mask = (pred_prob[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

        # Compute metrics
        metrics = compute_metrics(pred_mask, gt_mask)
        metrics['file_name'] = filename
        all_metrics.append(metrics)

        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{total}...")

    print(f"    Processed {total}/{total}. Done.")
    return all_metrics


def print_results(metrics_list, dataset_name=""):
    """Print aggregated metrics."""
    if not metrics_list:
        print(f"  No valid results for {dataset_name}")
        return

    header = f" {dataset_name} " if dataset_name else ""
    print(f"\n{'='*60}")
    print(f" Results{header}({len(metrics_list)} images)")
    print(f"{'='*60}")

    metric_names = ['Dice', 'IoU', 'HD95', 'ASD', 'Precision', 'Recall',
                    'Specificity', 'Accuracy']

    for name in metric_names:
        values = [m[name] for m in metrics_list if np.isfinite(m[name])]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            if name in ('HD95', 'ASD'):
                print(f"  {name:15s}: {mean:8.2f} +/- {std:.2f}")
            else:
                print(f"  {name:15s}: {mean:8.4f} +/- {std:.4f}  ({mean*100:.2f}%)")
        else:
            skipped = len(metrics_list) - len(values)
            print(f"  {name:15s}: N/A ({skipped} images had inf values)")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate AutoSAMUS on breast ultrasound datasets')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to AutoSAMUS checkpoint')
    parser.add_argument('--data-root', default='SAMUS_DATA',
                        help='Path to SAMUS_DATA directory')
    parser.add_argument('--eval-datasets', nargs='+',
                        default=['BUSI', 'BUSBRA', 'BUS', 'BUS_UC', 'BUS_UCLM'],
                        help='Datasets to evaluate on')
    parser.add_argument('--device', default='cuda:0',
                        help='Device (default: cuda:0)')
    parser.add_argument('--save-json', default=None,
                        help='Save per-image results to JSON file')
    args = parser.parse_args()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Datasets: {args.eval_datasets}")
    print(f"Device: {args.device}")

    model = build_model(args.checkpoint, args.device)

    all_results = {}
    combined_metrics = []

    for ds_name in args.eval_datasets:
        split_name = f'val-{ds_name}'
        print(f"\nEvaluating on {ds_name} ({split_name})...")

        metrics = evaluate_dataset(model, args.data_root, split_name, args.device)
        all_results[ds_name] = metrics
        combined_metrics.extend(metrics)
        print_results(metrics, ds_name)

    if len(args.eval_datasets) > 1:
        print_results(combined_metrics, "COMBINED")

    if args.save_json:
        for ds_metrics in all_results.values():
            for m in ds_metrics:
                for k, v in m.items():
                    if isinstance(v, (np.floating, np.integer)):
                        m[k] = float(v)
        os.makedirs(os.path.dirname(args.save_json) or '.', exist_ok=True)
        with open(args.save_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nPer-image results saved to {args.save_json}")


if __name__ == '__main__':
    main()
