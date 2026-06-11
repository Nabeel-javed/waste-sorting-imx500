from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.bin_logic import format_recommendation, get_recommended_bin, is_known_class
from src.utils import create_output_dirs, draw_label, draw_status_panel
from src.zone_logic import (
    check_zone_correctness,
    draw_zones,
    expected_zone,
    get_object_center,
    get_zone_from_center,
)


KNOWN_COLOR = (0, 255, 0)
UNKNOWN_COLOR = (0, 255, 255)
WRONG_COLOR = (0, 0, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the waste sorting demo with Raspberry Pi Picamera2.")
    parser.add_argument("--model", required=True, help="Path to weights, e.g. models/best.pt.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size.")
    parser.add_argument("--width", type=int, default=640, help="Camera preview width.")
    parser.add_argument("--height", type=int, default=480, help="Camera preview height.")
    parser.add_argument("--zones", action="store_true", help="Enable virtual bin-zone checking.")
    parser.add_argument("--headless", action="store_true", help="Do not open a display window; print detections.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames. 0 means run forever.")
    parser.add_argument("--threads", type=int, default=2, help="Torch CPU thread count.")
    return parser.parse_args()


def class_name_for(result, class_id: int) -> str:
    names = getattr(result, "names", {})
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(names[class_id])


def draw_help_overlay(frame) -> None:
    lines = ["q / ESC: quit", "s: screenshot", "z: toggle zones", "h: help"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    padding = 10
    line_gap = 8
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    box_width = max(width for width, _ in sizes) + padding * 2
    box_height = sum(height for _, height in sizes) + line_gap * (len(lines) - 1) + padding * 2
    height, width = frame.shape[:2]
    x1 = max(0, width - box_width - 10)
    y1 = max(0, height - box_height - 10)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + box_width, y1 + box_height), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    y = y1 + padding
    for line, (_, text_height) in zip(lines, sizes):
        y += text_height
        cv2.putText(frame, line, (x1 + padding, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_gap


def validate_model_path(model_path: str) -> bool:
    if Path(model_path).exists():
        return True
    print(f"Model not found at {model_path}. Copy models/best.pt to the Pi first.")
    return False


def main() -> int:
    args = parse_args()
    if not validate_model_path(args.model):
        return 1

    try:
        import torch

        torch.set_num_threads(max(1, args.threads))
    except Exception:
        pass

    model = YOLO(args.model)
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"},
        buffer_count=4,
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    zone_mode = bool(args.zones)
    show_help = True
    window_name = "Waste Sorting Assistant - Pi"
    frame_count = 0

    try:
        while True:
            rgb_frame = picam2.capture_array()
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            result = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            detections: list[str] = []
            warnings: list[str] = []

            if zone_mode:
                draw_zones(frame)

            for box in result.boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = class_name_for(result, class_id)
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                bbox = (x1, y1, x2, y2)
                known = is_known_class(class_name)
                color = KNOWN_COLOR if known else UNKNOWN_COLOR
                label = format_recommendation(class_name, confidence)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label(frame, label, x1, y1, color=color)
                detections.append(label)

                if zone_mode and known:
                    center_x, center_y = get_object_center(bbox)
                    zone_name = get_zone_from_center(center_x, frame.shape[1])
                    if check_zone_correctness(class_name, zone_name):
                        draw_label(frame, "Correct bin", center_x, center_y, color=KNOWN_COLOR)
                    else:
                        target_zone = expected_zone(class_name)
                        recommended_bin = get_recommended_bin(class_name)
                        warnings.append(f"Wrong bin: {class_name} -> use {target_zone} zone ({recommended_bin})")
                        draw_label(frame, f"Wrong bin: use {target_zone}", center_x, center_y, color=WRONG_COLOR)

            panel_lines = [f"Objects: {len(detections)}", *(detections[:5] or ["No object detected."])]
            if warnings:
                panel_lines.extend(warnings[:3])
            draw_status_panel(frame, panel_lines[:9])

            if args.headless:
                print(" | ".join(panel_lines), flush=True)
            else:
                if show_help:
                    draw_help_overlay(frame)
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    output_dir = Path("examples") / "screenshots"
                    create_output_dirs(output_dir)
                    screenshot_path = output_dir / f"pi_screenshot_{int(time.time())}.png"
                    cv2.imwrite(str(screenshot_path), frame)
                    print(f"Saved screenshot: {screenshot_path}")
                elif key == ord("z"):
                    zone_mode = not zone_mode
                    print(f"Zone mode: {'on' if zone_mode else 'off'}")
                elif key == ord("h"):
                    show_help = not show_help

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        picam2.stop()
        if not args.headless:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
