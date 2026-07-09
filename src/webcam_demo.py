from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
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

# Same scan-mode tuning as the on-camera demo (src/imx500_camera_demo.py)
BANNER_HOLD_S = 1.5  # keep the last verdict on screen this long after votes stop
SCAN_COLOR = (0, 220, 255)
VOTE_WINDOW_S = 1.5   # rolling window of per-frame votes
MIN_VOTES = 6         # frames with a detection required before any verdict
VOTE_SHARE = 0.6      # winning class must hold this share of the votes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real-time webcam waste sorting demo.")
    parser.add_argument("--model", required=True, help="Path to weights, e.g. models/best.pt.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--zones", action="store_true", help="Enable virtual bin-zone checking.")
    return parser.parse_args()


def is_builtin_ultralytics_model(model_arg: str) -> bool:
    if "/" in model_arg or "\\" in model_arg:
        return False
    return re.match(r"^yolo(v?\d+)?[nslmx][\w-]*\.pt$", model_arg) is not None


def validate_model_arg(model_arg: str) -> bool:
    if Path(model_arg).exists() or is_builtin_ultralytics_model(model_arg):
        return True
    print(f"Model not found at {model_arg}. Train first or download yolov8n.pt.")
    return False


def class_name_for(result, class_id: int) -> str:
    names = getattr(result, "names", {})
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(names[class_id])


def draw_scan_banner(frame, banner: dict | None) -> None:
    """Big top-centre banner: what the camera sees and which bin it belongs in."""
    if banner:
        title, subtitle, color = banner["title"], banner["sub"], banner.get("color", KNOWN_COLOR)
    else:
        title, subtitle, color = "SCAN AN OBJECT", "Hold an item in front of the camera", (200, 200, 200)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(title, font, 1.1, 3)
    (sw, sh), _ = cv2.getTextSize(subtitle, font, 0.7, 2)
    width = max(tw, sw) + 40
    height = th + sh + 46
    x0 = max(0, (frame.shape[1] - width) // 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 10), (x0 + width, 10 + height), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x0, 10), (x0 + width, 10 + height), color, 2)

    cv2.putText(frame, title, (x0 + 20, 10 + th + 16), font, 1.1, color, 3, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (x0 + 20, 10 + th + sh + 32), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


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


def main() -> int:
    args = parse_args()
    if not validate_model_arg(args.model):
        return 1

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"Cannot open camera {args.camera}. Try --camera 1.")
        return 1

    model = YOLO(args.model)
    zone_mode = bool(args.zones)
    show_help = True
    window_name = "Waste Sorting Assistant"

    # Multi-frame voting state (same behaviour as the on-camera demo).
    votes: deque[tuple[float, str, float]] = deque()
    banner: dict | None = None
    last_verdict: str | None = None

    while True:
        ok, frame = capture.read()
        if not ok:
            print("Camera frame capture failed.")
            break

        result = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        detections: list[str] = []
        warnings: list[str] = []
        best_known: tuple[str, float] | None = None

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
            if known and (best_known is None or confidence > best_known[1]):
                best_known = (class_name, confidence)

            if zone_mode and known:
                center_x, center_y = get_object_center(bbox)
                zone_name = get_zone_from_center(center_x, frame.shape[1])
                if check_zone_correctness(class_name, zone_name):
                    draw_label(frame, "Correct bin", center_x, center_y, color=KNOWN_COLOR)
                else:
                    target_zone = expected_zone(class_name)
                    recommended_bin = get_recommended_bin(class_name)
                    warning = f"Wrong bin: {class_name} -> use {target_zone} zone"
                    warnings.append(f"{warning} ({recommended_bin})")
                    draw_label(frame, f"Wrong bin: use {target_zone}", center_x, center_y, color=WRONG_COLOR)

        if zone_mode:
            if detections:
                panel_lines = [f"Objects: {len(detections)}", *detections[:5]]
            else:
                panel_lines = ["Objects: 0", "No object detected."]
            if warnings:
                panel_lines.extend(warnings[:3])
            draw_status_panel(frame, panel_lines[:9])
        else:
            now = time.monotonic()
            if best_known:
                votes.append((now, best_known[0], best_known[1]))
            while votes and votes[0][0] < now - VOTE_WINDOW_S:
                votes.popleft()

            verdict = None
            if len(votes) >= MIN_VOTES:
                counts = Counter(cls for _, cls, _ in votes)
                top_class, top_n = counts.most_common(1)[0]
                if top_n / len(votes) >= VOTE_SHARE:
                    top_confs = [c for _, cls, c in votes if cls == top_class]
                    verdict = (top_class, sum(top_confs) / len(top_confs))

            if verdict:
                verdict_class, verdict_conf = verdict
                banner = {
                    "title": f"{verdict_class.replace('_', ' ').upper()}  ({verdict_conf:.0%})",
                    "sub": f"Put it in: {get_recommended_bin(verdict_class)}",
                    "expires": now + BANNER_HOLD_S,
                }
                last_verdict = verdict_class
            elif votes:
                held = banner and now <= banner["expires"] and last_verdict
                if not held:
                    banner = {
                        "title": "SCANNING...",
                        "sub": f"Hold the object steady ({min(len(votes), MIN_VOTES)}/{MIN_VOTES})",
                        "color": SCAN_COLOR,
                        "expires": now,
                    }
                    last_verdict = None
            elif banner and now > banner["expires"]:
                banner = None
                last_verdict = None
            draw_scan_banner(frame, banner)

        if show_help:
            draw_help_overlay(frame)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            output_dir = Path("examples") / "screenshots"
            create_output_dirs(output_dir)
            screenshot_path = output_dir / f"screenshot_{int(time.time())}.png"
            cv2.imwrite(str(screenshot_path), frame)
            print(f"Saved screenshot: {screenshot_path}")
        elif key == ord("z"):
            zone_mode = not zone_mode
            print(f"Zone mode: {'on' if zone_mode else 'off'}")
        elif key == ord("h"):
            show_help = not show_help

    capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
