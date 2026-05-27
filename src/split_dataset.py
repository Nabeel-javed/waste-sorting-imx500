from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split raw images and optional YOLO labels into train/val/test.")
    parser.add_argument("--src-images", default="dataset/images/raw", help="Raw image folder.")
    parser.add_argument("--src-labels", default="dataset/labels/raw", help="Raw label folder.")
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        default=(0.7, 0.2, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--require-labels", action="store_true", help="Skip images without matching .txt labels.")
    return parser.parse_args()


def ensure_split_dirs() -> None:
    for split in SPLITS:
        Path("dataset/images", split).mkdir(parents=True, exist_ok=True)
        Path("dataset/labels", split).mkdir(parents=True, exist_ok=True)


def split_for_index(index: int, total: int, ratios: tuple[float, float, float]) -> str:
    train_cutoff = int(total * ratios[0])
    val_cutoff = train_cutoff + int(total * ratios[1])
    if index < train_cutoff:
        return "train"
    if index < val_cutoff:
        return "val"
    return "test"


def main() -> int:
    args = parse_args()
    ratios = tuple(args.ratios)
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        print("--ratios must contain three values that sum to 1.0")
        return 1

    src_images = Path(args.src_images)
    src_labels = Path(args.src_labels)
    if not src_images.exists():
        print(f"Image source folder not found: {src_images}")
        return 1

    ensure_split_dirs()
    images = [path for path in src_images.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    random.Random(args.seed).shuffle(images)

    counts = {split: 0 for split in SPLITS}
    skipped = 0

    for index, image_path in enumerate(images):
        label_path = src_labels / f"{image_path.stem}.txt"
        if not label_path.exists() and args.require_labels:
            print(f"Skipping {image_path.name}: missing label {label_path.name}")
            skipped += 1
            continue

        split = split_for_index(index, len(images), ratios)
        destination_image = Path("dataset/images", split, image_path.name)
        shutil.move(str(image_path), destination_image)

        if label_path.exists():
            destination_label = Path("dataset/labels", split, label_path.name)
            shutil.move(str(label_path), destination_label)

        counts[split] += 1

    print("Split summary")
    for split in SPLITS:
        print(f"  {split}: {counts[split]}")
    print(f"  skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

