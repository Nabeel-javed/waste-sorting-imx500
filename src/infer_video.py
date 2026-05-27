from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.bin_logic import format_recommendation, is_known_class
from src.utils import create_output_dirs, draw_label


KNOWN_COLOR = (0, 255, 0)
UNKNOWN_COLOR = (0, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO waste detection on a video file.")
    parser.add_argument("--model", required=True, help="Path to weights, e.g. models/best.pt.")
    parser.add_argument("--source", required=True, help="Input video path.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--out", default="examples/demo_videos/annotated.mp4", help="Output video path.")
    return parser.parse_args()


def class_name_for(result, class_id: int) -> str:
    names = getattr(result, "names", {})
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(names[class_id])


def annotate_frame(frame, result) -> None:
    for box in result.boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = class_name_for(result, class_id)
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        color = KNOWN_COLOR if is_known_class(class_name) else UNKNOWN_COLOR
        label = format_recommendation(class_name, confidence)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, label, x1, y1, color=color)


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        print(f"Video not found: {source}")
        return 1

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        print(f"Cannot open video: {source}")
        return 1

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = Path(args.out)
    create_output_dirs(output_path.parent)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    model = YOLO(args.model)

    frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = model(frame, conf=args.conf, verbose=False)[0]
        annotate_frame(frame, result)
        writer.write(frame)
        frame_count += 1

    capture.release()
    writer.release()
    print(f"Processed {frame_count} frames. Output saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
