"""Generate US30K split files with CAMUS entries removed.

CAMUS is a multi-class cardiac dataset where the same image appears with
3 different target masks (LV, MYO, LA).  AutoSAMUS has no manual prompt to
disambiguate which structure to segment, so training on CAMUS creates
contradictory supervision.  This script filters CAMUS out, keeping all
other US30K datasets for a cleaner AutoSAMUS foundation-model training.

Usage:
    python prepare_us30k_no_camus.py --data_path ./US30K
"""

import argparse
import os


CAMUS_KEYWORD = "Echocardiography-CAMUS"

SPLITS = {
    "train.txt": "train_no_camus.txt",
    "val.txt":   "val_no_camus.txt",
    "test.txt":  "test_no_camus.txt",
}


def filter_split(src_path, dst_path):
    with open(src_path, "r") as f:
        lines = f.readlines()
    original = len(lines)
    filtered = [l for l in lines if CAMUS_KEYWORD not in l]
    with open(dst_path, "w") as f:
        f.writelines(filtered)
    removed = original - len(filtered)
    print(f"  {os.path.basename(src_path)}: {original} -> {len(filtered)} entries ({removed} CAMUS removed)")
    return original, len(filtered)


def main():
    parser = argparse.ArgumentParser(description="Filter CAMUS from US30K splits")
    parser.add_argument("--data_path", type=str, default="./US30K",
                        help="Root directory of US30K dataset (contains MainPatient/)")
    args = parser.parse_args()

    mp_dir = os.path.join(args.data_path, "MainPatient")
    if not os.path.isdir(mp_dir):
        print(f"Error: {mp_dir} does not exist. Check --data_path.")
        return

    print(f"Filtering CAMUS from US30K splits in {mp_dir}")
    total_orig, total_kept = 0, 0
    for src_name, dst_name in SPLITS.items():
        src = os.path.join(mp_dir, src_name)
        dst = os.path.join(mp_dir, dst_name)
        if not os.path.isfile(src):
            print(f"  Warning: {src} not found, skipping")
            continue
        orig, kept = filter_split(src, dst)
        total_orig += orig
        total_kept += kept

    print(f"Done. Total: {total_orig} -> {total_kept} entries")


if __name__ == "__main__":
    main()
