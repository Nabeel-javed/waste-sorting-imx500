"""Convert extra public datasets (drinking-waste, TACO-YOLO) into the project dataset.

Adds TRAIN data only - the existing val split stays fixed so metrics remain
comparable across training runs.

Sources (downloaded/unzipped under --extra-root):
  drinking/Images_of_Waste/YOLO_imgs/   image+label pairs; class from filename
                                        prefix (AluCan, Glass, HDPEM, PET)
  taco_yolo/{train,valid}/              Roboflow TACO 18-class YOLO export;
                                        only classes that map cleanly are kept

Usage (typically on the training server):
    python src/convert_extra_public.py --extra-root data/public_extra --dataset dataset \
        --drinking-cap 1000
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

# Project ids: 0 plastic_bottle, 1 can, 2 paper (incl. cardboard), 3 glass_jar, 4 food_wrapper
DRINKING_PREFIX_MAP = {
    "alucan": 1,
    "glass": 3,
    "hdpem": 0,
    "pet": 0,
}

# Roboflow TACO 18-class ids -> project ids ('Bottle' is skipped: mixes glass+plastic)
TACO_ID_MAP = {
    4: 1,   # Can -> can
    5: 2,   # Carton -> paper (cardboard merged into paper)
    11: 2,  # Paper -> paper
    12: 4,  # Plastic bag - wrapper -> food_wrapper
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert extra public datasets into project train split.")
    parser.add_argument("--extra-root", default="data/public_extra", help="Folder with unzipped downloads.")
    parser.add_argument("--dataset", default="dataset", help="Project YOLO dataset root.")
    parser.add_argument("--drinking-cap", type=int, default=1000, help="Max drinking-waste images per class.")
    return parser.parse_args()


def sanitize(stem: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")


def convert_drinking(extra_root: Path, images_out: Path, labels_out: Path, cap: int) -> dict[str, int]:
    source = extra_root / "drinking" / "Images_of_Waste" / "YOLO_imgs"
    counts: dict[int, int] = defaultdict(int)
    added = 0
    for image_path in sorted(source.glob("*.jpg")):
        prefix_match = re.match(r"([A-Za-z]+)", image_path.stem)
        class_id = DRINKING_PREFIX_MAP.get((prefix_match.group(1) if prefix_match else "").lower())
        label_path = image_path.with_suffix(".txt")
        if class_id is None or not label_path.exists():
            continue
        if counts[class_id] >= cap:
            continue
        lines = []
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                lines.append(" ".join([str(class_id), *parts[1:]]))
        if not lines:
            continue
        stem = f"drink_{sanitize(image_path.stem)}"
        shutil.copy2(image_path, images_out / f"{stem}.jpg")
        (labels_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        counts[class_id] += 1
        added += 1
    return {"images": added, **{f"class_{k}": v for k, v in sorted(counts.items())}}


def convert_taco(extra_root: Path, images_out: Path, labels_out: Path) -> dict[str, int]:
    counts: dict[int, int] = defaultdict(int)
    added = 0
    for split in ("train", "valid"):
        image_dir = extra_root / "taco_yolo" / split / "images"
        label_dir = extra_root / "taco_yolo" / split / "labels"
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.glob("*.jpg")):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            lines = []
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5 and int(parts[0]) in TACO_ID_MAP:
                    lines.append(" ".join([str(TACO_ID_MAP[int(parts[0])]), *parts[1:]]))
            if not lines:
                continue
            stem = f"taco_{sanitize(image_path.stem)[:60]}"
            shutil.copy2(image_path, images_out / f"{stem}.jpg")
            (labels_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            for line in lines:
                counts[int(line.split()[0])] += 1
            added += 1
    return {"images": added, **{f"class_{k}_boxes": v for k, v in sorted(counts.items())}}


def main() -> int:
    args = parse_args()
    extra_root = Path(args.extra_root)
    images_out = Path(args.dataset) / "images" / "train"
    labels_out = Path(args.dataset) / "labels" / "train"

    # Idempotent re-runs: clear previously converted files first.
    removed = 0
    for out_dir in (images_out, labels_out):
        for prefix in ("drink_", "taco_"):
            for f in out_dir.glob(f"{prefix}*"):
                f.unlink()
                removed += 1
    if removed:
        print(f"Removed {removed} previously converted files.")

    print("drinking-waste:", convert_drinking(extra_root, images_out, labels_out, args.drinking_cap))
    print("taco:", convert_taco(extra_root, images_out, labels_out))
    print(f"train total now: {len(list(images_out.glob('*.jpg')))} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
