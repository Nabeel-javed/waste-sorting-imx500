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


def sequence_value(values: Any, index: int, default: float = 0.0) -> float:
    if values is None:
        return default
    try:
        return as_float(values[index])
    except Exception:
        return default


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
    if names:
        print("\nPer-class metrics")
        print(f"{'class':20s} {'P':>8s} {'R':>8s} {'mAP50':>8s} {'mAP50-95':>10s}")
        class_indexes = [int(class_id) for class_id in getattr(box, "ap_class_index", [])]
        metrics_by_class: dict[int, tuple[float, float, float, float]] = {}
        for metric_index, class_id in enumerate(class_indexes):
            metrics_by_class[class_id] = (
                sequence_value(getattr(box, "p", None), metric_index),
                sequence_value(getattr(box, "r", None), metric_index),
                sequence_value(getattr(box, "ap50", None), metric_index),
                sequence_value(getattr(box, "ap", None), metric_index),
            )

        for class_id, class_name in sorted(names.items()):
            precision, recall, map50, map_all = metrics_by_class.get(int(class_id), (0.0, 0.0, 0.0, 0.0))
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
