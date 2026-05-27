from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO waste sorting detector.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model, e.g. yolov8n.pt or yolo11n.pt.")
    parser.add_argument("--data", default="dataset/data.yaml", help="YOLO data.yaml path.")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--name", default="waste_sorting_yolov8n", help="Ultralytics run name.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = YOLO(args.model)
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
    )

    save_dir = Path(getattr(result, "save_dir", Path("runs") / "detect" / args.name))
    best_path = save_dir / "weights" / "best.pt"
    print(f"Training complete. Best weights should be at: {best_path}")
    print("Copy them into models/best.pt, or pass this path directly to inference scripts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

