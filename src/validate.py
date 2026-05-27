from __future__ import annotations

import argparse
from typing import Any

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained YOLO waste sorting detector.")
    parser.add_argument("--model", required=True, help="Path to trained weights, e.g. models/best.pt.")
    parser.add_argument("--data", default="dataset/data.yaml", help="YOLO data.yaml path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    return parser.parse_args()


def as_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def main() -> int:
    args = parse_args()
    metrics = YOLO(args.model).val(data=args.data, imgsz=args.imgsz)
    box = getattr(metrics, "box", None)

    if box is None:
        print("Validation finished, but no detection metrics were returned.")
        return 0

    print("\nValidation metrics")
    print(f"mAP50:    {as_float(getattr(box, 'map50', 0.0)):.4f}")
    print(f"mAP50-95: {as_float(getattr(box, 'map', 0.0)):.4f}")
    print(f"Precision:{as_float(getattr(box, 'mp', 0.0)):.4f}")
    print(f"Recall:   {as_float(getattr(box, 'mr', 0.0)):.4f}")

    names = getattr(metrics, "names", {}) or {}
    if names and hasattr(box, "class_result"):
        print("\nPer-class metrics")
        print(f"{'class':20s} {'P':>8s} {'R':>8s} {'mAP50':>8s} {'mAP50-95':>10s}")
        for class_id, class_name in sorted(names.items()):
            try:
                precision, recall, map50, map_all = box.class_result(int(class_id))
            except Exception:
                continue
            print(
                f"{str(class_name):20s} "
                f"{as_float(precision):8.4f} "
                f"{as_float(recall):8.4f} "
                f"{as_float(map50):8.4f} "
                f"{as_float(map_all):10.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

