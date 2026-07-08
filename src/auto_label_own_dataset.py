"""Semi-automatic labeling of self-collected images with YOLO-World.

Input: a folder of per-class subfolders (one dominant object per image), e.g.

    trash_project 2/
      cane/       -> can
      Plastic/    -> plastic_bottle
      glass/      -> glass_jar
      paper/      -> paper
      cardboard/  -> cardboard

For each class folder, YOLO-World (open-vocabulary detector) is prompted with
text descriptions of that class and the resulting boxes become YOLO-format
labels with the project class id. Annotated previews are written so a human can
review every label quickly. Images where nothing is detected either get a
near-full-frame fallback box (classes that fill the frame, e.g. TrashNet-style
paper/cardboard shots) or are flagged for manual review.

Usage:
    python src/auto_label_own_dataset.py \
        --source "/Users/Nabeel/Downloads/trash_project 2" \
        --out data/own_dataset
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from ultralytics import YOLOWorld


PROJECT_CLASS_IDS = {
    "plastic_bottle": 0,
    "can": 1,
    "paper": 2,
    "cardboard": 3,
    "glass_jar": 4,
    "food_wrapper": 5,
}


@dataclass
class FolderSpec:
    project_class: str
    prompts: list[str]
    # Classes that often fill the whole frame get a fallback box when YOLO-World
    # finds nothing; hand-held object classes are flagged for review instead.
    full_frame_fallback: bool = False
    review: list[str] = field(default_factory=list)


FOLDER_SPECS: dict[str, FolderSpec] = {
    "cane": FolderSpec("can", ["aluminum can", "tin can", "beverage can"]),
    "metal": FolderSpec("can", ["aluminum can", "tin can", "beverage can", "crushed can", "metal container"]),
    "plastic": FolderSpec("plastic_bottle", ["plastic bottle", "water bottle", "plastic container bottle"]),
    "glass": FolderSpec("glass_jar", ["glass bottle", "glass jar", "bottle", "jar"]),
    "paper": FolderSpec("paper", ["sheet of paper", "paper document", "flyer", "newspaper"], full_frame_fallback=True),
    "cardboard": FolderSpec("cardboard", ["cardboard box", "piece of cardboard", "carton"], full_frame_fallback=True),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label per-class image folders with YOLO-World.")
    parser.add_argument("--source", required=True, help="Folder containing per-class subfolders.")
    parser.add_argument("--out", default="data/own_dataset", help="Output root (images/, labels/, previews/).")
    parser.add_argument("--model", default="yolov8s-worldv2.pt", help="YOLO-World weights.")
    parser.add_argument("--conf", type=float, default=0.15, help="Minimum confidence for a box.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size.")
    parser.add_argument("--device", default="mps", help="Inference device (mps/cpu/cuda).")
    parser.add_argument("--max-boxes", type=int, default=5, help="Max boxes kept per image.")
    parser.add_argument("--prefix", default="own", help="Output filename prefix (e.g. own, trashnet).")
    return parser.parse_args()


def yolo_line(class_id: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    clamp = lambda v: max(0.0, min(1.0, v))
    return f"{class_id} {clamp(cx):.6f} {clamp(cy):.6f} {clamp(bw):.6f} {clamp(bh):.6f}"


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    out_root = Path(args.out)
    images_dir = out_root / "images"
    labels_dir = out_root / "labels"
    previews_dir = out_root / "previews"
    for d in (images_dir, labels_dir, previews_dir):
        d.mkdir(parents=True, exist_ok=True)

    model = YOLOWorld(args.model)
    stats: dict[str, dict[str, int]] = {}

    for folder in sorted(source.iterdir()):
        if not folder.is_dir():
            continue
        spec = FOLDER_SPECS.get(folder.name.lower())
        if spec is None:
            print(f"Skipping unknown folder: {folder.name}")
            continue

        class_id = PROJECT_CLASS_IDS[spec.project_class]
        model.set_classes(spec.prompts)
        counters = {"images": 0, "detected": 0, "fallback": 0, "review": 0, "boxes": 0}
        stats[spec.project_class] = counters

        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for image_path in files:
            counters["images"] += 1
            # Name by source stem: unique by construction and traceable back to the original.
            safe_stem = "".join(c if c.isalnum() else "_" for c in image_path.stem.lower())
            stem = f"{args.prefix}_{spec.project_class}_{safe_stem}"
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Unreadable image skipped: {image_path}")
                continue
            h, w = frame.shape[:2]

            result = model.predict(
                str(image_path), conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False
            )[0]

            boxes = []
            if len(result.boxes):
                ranked = sorted(result.boxes, key=lambda b: float(b.conf[0]), reverse=True)
                top_conf = float(ranked[0].conf[0])
                for box in ranked[: args.max_boxes]:
                    conf = float(box.conf[0])
                    # Keep the top box plus clearly-confident companions only.
                    if conf >= max(args.conf, 0.5 * top_conf):
                        boxes.append((*[float(v) for v in box.xyxy[0].tolist()], conf))

            lines = []
            if boxes:
                counters["detected"] += 1
                counters["boxes"] += len(boxes)
                for x1, y1, x2, y2, _ in boxes:
                    lines.append(yolo_line(class_id, x1, y1, x2, y2, w, h))
            elif spec.full_frame_fallback:
                counters["fallback"] += 1
                margin_w, margin_h = w * 0.02, h * 0.02
                boxes = [(margin_w, margin_h, w - margin_w, h - margin_h, 0.0)]
                lines.append(yolo_line(class_id, *boxes[0][:4], w, h))
            else:
                counters["review"] += 1
                spec.review.append(image_path.name)

            if lines:
                shutil.copy2(image_path, images_dir / f"{stem}.jpg")
                (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

            preview = frame.copy()
            for x1, y1, x2, y2, conf in boxes:
                color = (0, 255, 0) if conf > 0 else (0, 165, 255)
                cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                tag = f"{spec.project_class} {conf:.2f}" if conf > 0 else f"{spec.project_class} FALLBACK"
                cv2.putText(preview, tag, (int(x1) + 4, max(24, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            if not boxes:
                cv2.putText(preview, "NO DETECTION - REVIEW", (12, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)
            class_preview_dir = previews_dir / spec.project_class
            class_preview_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(class_preview_dir / f"{stem}.jpg"), preview)

    print("\n=== Auto-label summary ===")
    for class_name, counters in stats.items():
        print(
            f"{class_name:15s} images={counters['images']:3d} detected={counters['detected']:3d} "
            f"fallback={counters['fallback']:3d} needs_review={counters['review']:3d} boxes={counters['boxes']:3d}"
        )
    for folder_name, spec in FOLDER_SPECS.items():
        for name in spec.review:
            print(f"REVIEW ({spec.project_class}): {name}")
    print(f"\nLabels: {labels_dir}\nPreviews for human review: {previews_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
