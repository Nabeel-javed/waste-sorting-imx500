"""Merge the auto-labeled own dataset into the YOLO training dataset.

- Stratified per-class split of own images into train/val.
- Own train images are duplicated (oversampled) so the small own set carries
  meaningful weight against the much larger public base.
- Idempotent: re-running first removes all previously merged own_* files.

Usage:
    python src/merge_own_dataset.py --own data/own_dataset --dataset dataset --oversample 3
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge own auto-labeled images into the YOLO dataset.")
    parser.add_argument("--own", default="data/own_dataset", help="Own dataset root (images/, labels/).")
    parser.add_argument("--dataset", default="dataset", help="YOLO dataset root.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Per-class fraction held out for val.")
    parser.add_argument("--oversample", type=int, default=3, help="Copies of each own train image.")
    parser.add_argument("--seed", type=int, default=42, help="Split shuffle seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    own_images = Path(args.own) / "images"
    own_labels = Path(args.own) / "labels"
    dataset = Path(args.dataset)

    # Remove any previously merged own_* files so re-runs do not accumulate.
    removed = 0
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            for f in (dataset / sub / split).glob("own_*"):
                f.unlink()
                removed += 1
    if removed:
        print(f"Removed {removed} previously merged own_* files.")

    class_names = ("plastic_bottle", "can", "paper", "cardboard", "glass_jar", "food_wrapper")
    by_class: dict[str, list[Path]] = defaultdict(list)
    for image_path in sorted(own_images.glob("own_*.jpg")):
        matched = next((c for c in class_names if image_path.stem.startswith(f"own_{c}_")), None)
        if matched and (own_labels / f"{image_path.stem}.txt").exists():
            by_class[matched].append(image_path)

    rng = random.Random(args.seed)
    totals = {"train": 0, "val": 0}
    for class_name, files in sorted(by_class.items()):
        rng.shuffle(files)
        n_val = max(1, round(len(files) * args.val_fraction))
        val_files, train_files = files[:n_val], files[n_val:]

        for image_path in val_files:
            label_path = own_labels / f"{image_path.stem}.txt"
            shutil.copy2(image_path, dataset / "images" / "val" / image_path.name)
            shutil.copy2(label_path, dataset / "labels" / "val" / label_path.name)
            totals["val"] += 1

        for image_path in train_files:
            label_path = own_labels / f"{image_path.stem}.txt"
            for copy_index in range(args.oversample):
                suffix = "" if copy_index == 0 else f"_dup{copy_index}"
                shutil.copy2(image_path, dataset / "images" / "train" / f"{image_path.stem}{suffix}.jpg")
                shutil.copy2(label_path, dataset / "labels" / "train" / f"{image_path.stem}{suffix}.txt")
                totals["train"] += 1

        print(f"{class_name:15s} total={len(files):3d} -> train={len(train_files)} (x{args.oversample}) val={n_val}")

    print(f"\nMerged into {dataset}: +{totals['train']} train files, +{totals['val']} val files.")
    for split in ("train", "val"):
        count = len(list((dataset / "images" / split).glob("*.jpg")))
        print(f"{split} now has {count} images total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
