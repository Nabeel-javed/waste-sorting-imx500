from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.bin_logic import format_recommendation, is_known_class
from src.utils import create_output_dirs, draw_label, iter_image_files


KNOWN_COLOR = (0, 255, 0)
UNKNOWN_COLOR = (0, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO waste detection on an image or image folder.")
    parser.add_argument("--model", required=True, help="Path to weights, e.g. models/best.pt.")
    parser.add_argument("--source", required=True, help="Image file or directory.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--out", default="examples/sample_images/out", help="Output file or directory.")
    return parser.parse_args()


def class_name_for(result, class_id: int) -> str:
    names = getattr(result, "names", {})
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(names[class_id])


def annotate_image(image, result) -> list[str]:
    detections: list[str] = []
    for box in result.boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = class_name_for(result, class_id)
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        color = KNOWN_COLOR if is_known_class(class_name) else UNKNOWN_COLOR
        label = format_recommendation(class_name, confidence)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        draw_label(image, label, x1, y1, color=color)
        detections.append(label)
    return detections


def output_path_for(image_path: Path, out_arg: str, multiple: bool) -> Path:
    out_path = Path(out_arg)
    if not multiple and out_path.suffix:
        create_output_dirs(out_path.parent)
        return out_path
    create_output_dirs(out_path)
    return out_path / f"{image_path.stem}_annotated{image_path.suffix}"


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        print(f"Source not found: {source}")
        return 1

    image_paths = list(iter_image_files(source))
    if not image_paths:
        print(f"No images found in: {source}")
        return 1

    model = YOLO(args.model)
    multiple = len(image_paths) > 1 or source.is_dir()

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        result = model(image, conf=args.conf, verbose=False)[0]
        detections = annotate_image(image, result)
        save_path = output_path_for(image_path, args.out, multiple)
        cv2.imwrite(str(save_path), image)

        print(f"\n{image_path} -> {save_path}")
        if detections:
            for detection in detections:
                print(f"  {detection}")
        else:
            print("  No object detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
