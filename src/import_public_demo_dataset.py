from __future__ import annotations

import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any


DATASET_URLS = {
    "train": "https://huggingface.co/datasets/keremberke/garbage-object-detection/resolve/main/data/train.zip",
    "valid": "https://huggingface.co/datasets/keremberke/garbage-object-detection/resolve/main/data/valid.zip",
    "test": "https://huggingface.co/datasets/keremberke/garbage-object-detection/resolve/main/data/test.zip",
}

CLASS_REMAP = {
    "plastic": 0,  # plastic_bottle, temporary broad-class proxy
    "metal": 1,  # can, temporary broad-class proxy
    "paper": 2,
    "cardboard": 3,
    "glass": 4,  # glass_jar, temporary broad-class proxy
    "biodegradable": 5,  # food_wrapper, temporary demo proxy
}

PROJECT_CLASS_NAMES = {
    0: "plastic_bottle",
    1: "can",
    2: "paper",
    3: "cardboard",
    4: "glass_jar",
    5: "food_wrapper",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a temporary public garbage-detection dataset for the intermediate demo."
    )
    parser.add_argument("--download-dir", default="data/public/keremberke-garbage", help="Download/cache folder.")
    parser.add_argument("--dataset-dir", default="dataset", help="Project YOLO dataset folder.")
    parser.add_argument("--max-train-per-class", type=int, default=120, help="Approximate max train images per class.")
    parser.add_argument("--max-val-per-class", type=int, default=30, help="Approximate max val images per class.")
    parser.add_argument("--max-test-per-class", type=int, default=30, help="Approximate max test images per class.")
    parser.add_argument("--full-train", action="store_true", help="Download/use the larger public train.zip split.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed.")
    parser.add_argument("--keep-existing", action="store_true", help="Do not clear existing train/val/test folders first.")
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using cached {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    marker = output_dir / ".extracted"
    if marker.exists():
        print(f"Using extracted {output_dir}")
        return
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    marker.write_text("ok\n", encoding="utf-8")


def clear_project_split_dirs(dataset_dir: Path) -> None:
    for group in ("images", "labels"):
        for split in ("train", "val", "test"):
            split_dir = dataset_dir / group / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for path in split_dir.iterdir():
                if path.name != ".gitkeep":
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()


def coco_to_yolo_line(annotation: dict[str, Any], image: dict[str, Any], class_id: int) -> str:
    x, y, width, height = [float(value) for value in annotation["bbox"]]
    image_width = float(image["width"])
    image_height = float(image["height"])
    x_center = (x + width / 2.0) / image_width
    y_center = (y + height / 2.0) / image_height
    norm_width = width / image_width
    norm_height = height / image_height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"


def load_coco(split_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[int, str], dict[int, list[dict[str, Any]]]]:
    coco_path = split_dir / "_annotations.coco.json"
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    images = {int(image["id"]): image for image in data["images"]}
    categories = {int(category["id"]): str(category["name"]) for category in data["categories"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    return images, categories, annotations_by_image


def choose_images(
    images: dict[int, dict[str, Any]],
    categories: dict[int, str],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    max_per_class: int,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    image_ids = list(images.keys())
    rng.shuffle(image_ids)
    counts: dict[int, int] = defaultdict(int)
    chosen: list[int] = []

    for image_id in image_ids:
        project_classes = {
            CLASS_REMAP[categories[int(annotation["category_id"])]]
            for annotation in annotations_by_image.get(image_id, [])
            if categories[int(annotation["category_id"])] in CLASS_REMAP
        }
        if not project_classes:
            continue
        if all(counts[class_id] >= max_per_class for class_id in project_classes):
            continue
        chosen.append(image_id)
        for class_id in project_classes:
            counts[class_id] += 1
        if all(counts[class_id] >= max_per_class for class_id in PROJECT_CLASS_NAMES):
            break

    return chosen


def write_yolo_split(
    source_dir: Path,
    dataset_dir: Path,
    output_split: str,
    max_per_class: int,
    seed: int,
) -> dict[str, int]:
    images, categories, annotations_by_image = load_coco(source_dir)
    chosen_ids = choose_images(images, categories, annotations_by_image, max_per_class, seed)
    image_output_dir = dataset_dir / "images" / output_split
    label_output_dir = dataset_dir / "labels" / output_split
    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    counts = {"images": 0, "labels": 0, "boxes": 0}
    for image_id in chosen_ids:
        image = images[image_id]
        source_image = source_dir / image["file_name"]
        if not source_image.exists() or source_image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        lines: list[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            category_name = categories[int(annotation["category_id"])]
            if category_name not in CLASS_REMAP:
                continue
            lines.append(coco_to_yolo_line(annotation, image, CLASS_REMAP[category_name]))
        if not lines:
            continue

        destination_image = image_output_dir / source_image.name
        destination_label = label_output_dir / f"{source_image.stem}.txt"
        shutil.copy2(source_image, destination_image)
        destination_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts["images"] += 1
        counts["labels"] += 1
        counts["boxes"] += len(lines)

    return counts


def main() -> int:
    args = parse_args()
    download_dir = Path(args.download_dir)
    dataset_dir = Path(args.dataset_dir)

    requested_splits = ("train", "valid", "test") if args.full_train else ("valid", "test")
    for split in requested_splits:
        url = DATASET_URLS[split]
        zip_path = download_dir / f"{split}.zip"
        extract_dir = download_dir / split
        download_file(url, zip_path)
        extract_zip(zip_path, extract_dir)

    if not args.keep_existing:
        clear_project_split_dirs(dataset_dir)

    train_counts = write_yolo_split(
        download_dir / ("train" if args.full_train else "valid"),
        dataset_dir,
        "train",
        args.max_train_per_class,
        args.seed,
    )
    val_counts = write_yolo_split(
        download_dir / "test",
        dataset_dir,
        "val",
        args.max_val_per_class,
        args.seed + 1,
    )
    test_counts = write_yolo_split(
        download_dir / "test",
        dataset_dir,
        "test",
        args.max_test_per_class,
        args.seed + 2,
    )

    print("\nTemporary public demo dataset created")
    for split, counts in (("train", train_counts), ("val", val_counts), ("test", test_counts)):
        print(f"  {split}: {counts['images']} images, {counts['boxes']} boxes")
    print("\nClass remap used for the demo branch:")
    print("  plastic -> plastic_bottle")
    print("  metal -> can")
    print("  paper -> paper")
    print("  cardboard -> cardboard")
    print("  glass -> glass_jar")
    print("  biodegradable -> food_wrapper")
    print("\nThis is only for the intermediate demo. Replace it with your own collected dataset for final work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
