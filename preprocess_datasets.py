"""
Preprocess BUSI and BUS datasets into the SAMUS-compatible format.

SAMUS expects:
    dataset_root/
    ├── MainPatient/
    │   ├── class.json          # {"DatasetName": num_classes}
    │   ├── test-DatasetName.txt  # lines: DatasetName/image_id
    │   └── ...
    ├── DatasetName/
    │   ├── img/   *.png  (grayscale)
    │   └── label/ *.png  (binary mask, 0/1 pixel values)

This script converts the raw BUSI and BUS datasets into that layout.

BUSI raw format:
    BUSI/{benign,malignant,normal}/
        image.png  +  image_mask.png  (+ optional image_mask_1.png, ...)

BUS (Dataset B) raw format:
    BUS/original/*.png   (grayscale images)
    BUS/GT/*.png         (grayscale masks, pixel values 0/255)
"""

import os
import sys
import json
import shutil
import argparse
import cv2
import numpy as np
from glob import glob


def process_busi(raw_dir, out_root, test_ratio=1.0):
    """
    Convert the raw BUSI dataset.
    - Merges multiple mask files (_mask.png, _mask_1.png, ...) via union.
    - Skips 'normal' images that have empty masks (all zeros).
    - Renames images to a clean sequential format: benignXXXX, malignantXXXX.
    - If test_ratio==1.0, ALL images go into the test split (inference-only).
    """
    dataset_name = "Breast-BUSI-Ext"
    img_out = os.path.join(out_root, dataset_name, "img")
    label_out = os.path.join(out_root, dataset_name, "label")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(label_out, exist_ok=True)

    entries = []  # (clean_name,)
    idx = 0

    for category in ["benign", "malignant", "normal"]:
        cat_dir = os.path.join(raw_dir, category)
        if not os.path.isdir(cat_dir):
            print(f"  [WARN] Category dir not found: {cat_dir}, skipping.")
            continue

        # Collect unique base names (without _mask suffix)
        all_files = sorted(os.listdir(cat_dir))
        base_names = set()
        for f in all_files:
            if f.endswith(".png") and "_mask" not in f:
                base_names.add(f[:-4])  # strip .png

        for base in sorted(base_names):
            img_file = os.path.join(cat_dir, base + ".png")
            if not os.path.isfile(img_file):
                continue

            # Collect all mask files for this image
            mask_files = []
            primary_mask = os.path.join(cat_dir, base + "_mask.png")
            if os.path.isfile(primary_mask):
                mask_files.append(primary_mask)
            # Additional masks: _mask_1.png, _mask_2.png, ...
            for extra in sorted(glob(os.path.join(cat_dir, base + "_mask_*.png"))):
                if extra not in mask_files:
                    mask_files.append(extra)

            if not mask_files:
                continue

            # Merge masks via union
            merged_mask = None
            for mf in mask_files:
                m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                if merged_mask is None:
                    merged_mask = np.zeros_like(m, dtype=np.uint8)
                merged_mask = np.maximum(merged_mask, m)

            if merged_mask is None:
                continue

            # Binarize: anything > 0 becomes 1
            merged_mask[merged_mask > 0] = 1

            # Skip normal images with completely empty masks
            if category == "normal" and merged_mask.sum() == 0:
                continue

            # Read image as grayscale
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            clean_name = f"{category}{idx:04d}"
            cv2.imwrite(os.path.join(img_out, clean_name + ".png"), img)
            cv2.imwrite(os.path.join(label_out, clean_name + ".png"), merged_mask)
            entries.append(clean_name)
            idx += 1

    print(f"  BUSI: {len(entries)} images processed -> {dataset_name}/")
    return dataset_name, entries


def process_bus(raw_dir, out_root, test_ratio=1.0):
    """
    Convert the raw BUS (Dataset B) dataset.
    Masks are 0/255 grayscale -> convert to 0/1.
    """
    dataset_name = "Breast-BUS-Ext"
    img_out = os.path.join(out_root, dataset_name, "img")
    label_out = os.path.join(out_root, dataset_name, "label")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(label_out, exist_ok=True)

    entries = []
    img_dir = os.path.join(raw_dir, "original")
    gt_dir = os.path.join(raw_dir, "GT")

    if not os.path.isdir(img_dir) or not os.path.isdir(gt_dir):
        print(f"  [ERROR] BUS dataset directories not found: {img_dir} or {gt_dir}")
        return dataset_name, entries

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

    for f in img_files:
        gt_file = os.path.join(gt_dir, f)
        if not os.path.isfile(gt_file):
            print(f"  [WARN] No GT for {f}, skipping.")
            continue

        img = cv2.imread(os.path.join(img_dir, f), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        # Binarize mask
        mask[mask > 0] = 1

        clean_name = f[:-4]  # already clean like 000001
        cv2.imwrite(os.path.join(img_out, clean_name + ".png"), img)
        cv2.imwrite(os.path.join(label_out, clean_name + ".png"), mask)
        entries.append(clean_name)

    print(f"  BUS: {len(entries)} images processed -> {dataset_name}/")
    return dataset_name, entries


def write_split_files(out_root, dataset_name, entries, train_ratio=0.0, val_ratio=0.0):
    """
    Write MainPatient split files.
    For inference-only, all samples go to the test split.
    """
    main_dir = os.path.join(out_root, "MainPatient")
    os.makedirs(main_dir, exist_ok=True)

    n = len(entries)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_entries = entries[:n_train]
    val_entries = entries[n_train:n_train + n_val]
    test_entries = entries[n_train + n_val:]

    # Write dataset-specific split files
    def write_split(split_name, split_entries, include_class=True):
        fname = f"{split_name}-{dataset_name}.txt"
        with open(os.path.join(main_dir, fname), "w") as f:
            for i, name in enumerate(split_entries, 1):
                if include_class and "test" not in split_name:
                    f.write(f"1/{dataset_name}/{name}\n")
                else:
                    f.write(f"{dataset_name}/{name}\n")

    write_split("train", train_entries)
    write_split("val", val_entries)
    write_split("test", test_entries, include_class=False)

    return {
        "train": train_entries,
        "val": val_entries,
        "test": test_entries,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess BUSI/BUS datasets for SAMUS")
    parser.add_argument("--busi_dir", type=str,
                        default="/Users/langtian/Desktop/NYU/MS Thesis/3D SAM Foundation Model/Med3D/Data/BUSI",
                        help="Path to raw BUSI dataset")
    parser.add_argument("--bus_dir", type=str,
                        default="/Users/langtian/Desktop/NYU/MS Thesis/3D SAM Foundation Model/Med3D/Data/BUS",
                        help="Path to raw BUS dataset")
    parser.add_argument("--output_dir", type=str,
                        default="./data/processed",
                        help="Output directory for SAMUS-formatted data")
    args = parser.parse_args()

    out_root = args.output_dir
    os.makedirs(out_root, exist_ok=True)

    print("=" * 60)
    print("Preprocessing datasets for SAMUS/AutoSAMUS inference")
    print("=" * 60)

    class_dict = {}
    all_test = {}

    # Process BUSI
    if os.path.isdir(args.busi_dir):
        print(f"\n[1/2] Processing BUSI from: {args.busi_dir}")
        ds_name, entries = process_busi(args.busi_dir, out_root)
        if entries:
            class_dict[ds_name] = 2  # binary: background + foreground
            splits = write_split_files(out_root, ds_name, entries)
            all_test[ds_name] = splits["test"]
    else:
        print(f"\n[1/2] BUSI directory not found: {args.busi_dir}, skipping.")

    # Process BUS
    if os.path.isdir(args.bus_dir):
        print(f"\n[2/2] Processing BUS from: {args.bus_dir}")
        ds_name, entries = process_bus(args.bus_dir, out_root)
        if entries:
            class_dict[ds_name] = 2
            splits = write_split_files(out_root, ds_name, entries)
            all_test[ds_name] = splits["test"]
    else:
        print(f"\n[2/2] BUS directory not found: {args.bus_dir}, skipping.")

    # Write class.json
    main_dir = os.path.join(out_root, "MainPatient")
    os.makedirs(main_dir, exist_ok=True)
    with open(os.path.join(main_dir, "class.json"), "w") as f:
        json.dump(class_dict, f, indent=4)

    # Write combined test.txt (union of all datasets)
    combined_test = []
    for ds, names in all_test.items():
        for name in names:
            combined_test.append(f"{ds}/{name}")
    with open(os.path.join(main_dir, "test.txt"), "w") as f:
        for line in combined_test:
            f.write(line + "\n")

    # Also write empty train.txt and val.txt so the loader doesn't crash
    for split in ["train", "val"]:
        fpath = os.path.join(main_dir, f"{split}.txt")
        if not os.path.exists(fpath):
            open(fpath, "w").close()

    print(f"\nDone! Processed data saved to: {os.path.abspath(out_root)}")
    print(f"  class.json: {class_dict}")
    total = sum(len(v) for v in all_test.values())
    print(f"  Total test images: {total}")
    print()


if __name__ == "__main__":
    main()
