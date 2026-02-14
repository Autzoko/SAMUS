"""
Preprocess 5 breast ultrasound datasets for SAMUS/AutoSAMUS training.

Uses UltraSAM's preprocessed data (images + COCO annotations) as the source,
since those have consistent filenames and train/val splits.

Output format (SAMUS standard):
    SAMUS_DATA/
    ├── MainPatient/
    │   ├── class.json
    │   ├── train-AllBreast.txt  (combined)
    │   ├── val-AllBreast.txt
    │   ├── train-BUSI.txt       (per-dataset)
    │   ├── val-BUSI.txt
    │   └── ...
    ├── BUSI/
    │   ├── img/*.png   (256x256 grayscale)
    │   └── label/*.png (256x256 binary mask, 0/255)
    └── ...

Usage:
    cd SAMUS
    python scripts/preprocess_breast_for_samus.py \
        --ultrasam-data ../UltraSam/UltraSAM_DATA \
        --save-dir SAMUS_DATA
"""

import argparse
import json
import os

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_util


IMG_SIZE = 256

DATASETS = ['BUSI', 'BUSBRA', 'BUS', 'BUS_UC', 'BUS_UCLM']

# Map output dataset names to UltraSAM_DATA directory names and COCO annotation prefix
DS_DIR_MAP = {
    'BUSI': ('BUSI', 'BUSI'),
    'BUSBRA': ('BUSBRA', 'BUSBRA'),
    'BUS': ('BUS', 'BUS'),
    'BUS_UC': ('BUS_UC', 'BUS_UC'),
    'BUS_UCLM': ('BUS_UCLM', 'BUS_UCLM'),
}


def process_dataset_split(ultrasam_dir, save_dir, ds_name, split):
    """Process one dataset split (train or val).

    Args:
        ultrasam_dir: Path to UltraSAM_DATA root
        save_dir: Output SAMUS_DATA root
        ds_name: Dataset name (e.g., 'BUSI')
        split: 'train' or 'val'

    Returns:
        List of split IDs in SAMUS format: '1/{ds_name}/{stem}'
    """
    ds_dir, coco_prefix = DS_DIR_MAP[ds_name]
    ann_file = os.path.join(
        ultrasam_dir, ds_dir, 'annotations', f'{split}.{coco_prefix}__coco.json')

    if not os.path.exists(ann_file):
        print(f"  Warning: {ann_file} not found, skipping")
        return []

    img_src_dir = os.path.join(ultrasam_dir, ds_dir, 'images')
    img_save_dir = os.path.join(save_dir, ds_name, 'img')
    lbl_save_dir = os.path.join(save_dir, ds_name, 'label')
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(lbl_save_dir, exist_ok=True)

    coco = COCO(ann_file)
    img_ids = sorted(coco.getImgIds())
    split_ids = []
    skipped = 0

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        fname = img_info['file_name']
        h, w = img_info['height'], img_info['width']
        stem = os.path.splitext(fname)[0]

        # Load image
        img_path = os.path.join(img_src_dir, fname)
        if not os.path.exists(img_path):
            skipped += 1
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue

        # Build union mask from all annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        if len(anns) == 0:
            skipped += 1
            continue

        gt_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            rle = coco.annToRLE(ann)
            m = mask_util.decode(rle)
            gt_mask = np.maximum(gt_mask, m)

        # Skip empty masks
        if gt_mask.sum() == 0:
            skipped += 1
            continue

        # Resize to 256x256
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                                 interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(gt_mask, (IMG_SIZE, IMG_SIZE),
                                  interpolation=cv2.INTER_NEAREST)

        # Binarize mask to 0/255 for SAMUS (label loader reads grayscale,
        # then does mask[mask > 1] = 1 for binary segmentation)
        mask_resized[mask_resized > 0] = 255

        # Save
        cv2.imwrite(os.path.join(img_save_dir, stem + '.png'), img_resized)
        cv2.imwrite(os.path.join(lbl_save_dir, stem + '.png'), mask_resized)

        split_ids.append(f"1/{ds_name}/{stem}")

    if skipped > 0:
        print(f"    Skipped {skipped} images (missing file or empty mask)")

    return split_ids


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess breast datasets from UltraSAM format to SAMUS format')
    parser.add_argument('--ultrasam-data', required=True,
                        help='Path to UltraSAM_DATA directory')
    parser.add_argument('--save-dir', default='SAMUS_DATA',
                        help='Output directory for SAMUS format data')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.save_dir, 'MainPatient'), exist_ok=True)

    print("Converting UltraSAM data to SAMUS format...")
    print(f"  Source: {args.ultrasam_data}")
    print(f"  Output: {args.save_dir}")
    print()

    all_train, all_val = [], []

    for ds_name in DATASETS:
        print(f"Processing {ds_name}...")

        train_ids = process_dataset_split(
            args.ultrasam_data, args.save_dir, ds_name, 'train')
        val_ids = process_dataset_split(
            args.ultrasam_data, args.save_dir, ds_name, 'val')

        print(f"  {ds_name}: {len(train_ids)} train, {len(val_ids)} val")

        all_train.extend(train_ids)
        all_val.extend(val_ids)

        # Write per-dataset split files
        with open(os.path.join(args.save_dir, 'MainPatient',
                               f'train-{ds_name}.txt'), 'w') as f:
            f.write('\n'.join(train_ids) + '\n')
        with open(os.path.join(args.save_dir, 'MainPatient',
                               f'val-{ds_name}.txt'), 'w') as f:
            f.write('\n'.join(val_ids) + '\n')
        print()

    # Write combined split files
    for name, ids in [('train-AllBreast', all_train), ('val-AllBreast', all_val),
                      ('train', all_train), ('val', all_val)]:
        with open(os.path.join(args.save_dir, 'MainPatient', f'{name}.txt'), 'w') as f:
            f.write('\n'.join(ids) + '\n')

    # Write class.json (all binary: background + lesion)
    class_dict = {ds: 2 for ds in DATASETS}
    with open(os.path.join(args.save_dir, 'MainPatient', 'class.json'), 'w') as f:
        json.dump(class_dict, f, indent=2)

    print("=" * 60)
    print(f"Total: {len(all_train)} train, {len(all_val)} val")
    print(f"Data saved to {args.save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
