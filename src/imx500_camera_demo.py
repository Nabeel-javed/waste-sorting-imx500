"""Real-time waste sorting on the Sony IMX500 AI camera accelerator.

Unlike src/pi_picamera_demo.py (which runs the YOLO .pt model on the Pi CPU),
this script loads a packaged IMX500 network (.rpk). Inference runs *inside* the
camera sensor. The Pi only reads the output tensors (boxes/scores/classes) from
the frame metadata and applies the same bin + zone recommendation logic.

Model output contract (Ultralytics YOLOv8/YOLO11 exported with format=imx):
    output_tensors[0] = boxes    (N, 4) as x0, y0, x1, y1 in input-tensor pixels
    output_tensors[1] = scores   (N,)
    output_tensors[2] = class_id (N,)
    output_tensors[3] = count    number of valid detections
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2, Preview
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

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
BANNER_HOLD_S = 1.5  # keep the last result on screen this long after it disappears


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Waste sorting on the Sony IMX500 AI camera.")
    parser.add_argument("--model", required=True, help="Path to the packaged IMX500 network .rpk.")
    parser.add_argument("--labels", default=None, help="Path to labels.txt. Defaults to configs class order.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--fps", type=int, default=None, help="Camera / inference frame rate.")
    parser.add_argument("--zones", action="store_true",
                        help="Use the virtual bin-zone overlay instead of the default scan banner.")
    parser.add_argument("--headless", action="store_true", help="No preview window; print detections.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 means run forever.")
    parser.add_argument("--debug-outputs", action="store_true", help="Print output tensor shapes once and continue.")
    return parser.parse_args()


def default_labels() -> list[str]:
    return ["plastic_bottle", "can", "paper", "cardboard", "glass_jar", "food_wrapper"]


def load_labels(labels_path: str | None) -> list[str]:
    if labels_path and Path(labels_path).exists():
        with open(labels_path, "r", encoding="utf-8") as handle:
            labels = [line.strip() for line in handle if line.strip()]
        if labels:
            return labels
    return default_labels()


class Detection:
    """A single detection mapped into display-stream pixel coordinates."""

    def __init__(self, box_xywh, class_id: int, conf: float, class_name: str) -> None:
        self.box = box_xywh  # (x, y, w, h) in display pixels
        self.class_id = class_id
        self.conf = conf
        self.class_name = class_name


def parse_detections(imx500, picam2, metadata, labels, threshold, debug=False):
    """Read output tensors from camera metadata and map them to Detection objects."""
    outputs = imx500.get_outputs(metadata, add_batch=False)
    if outputs is None:
        return None

    if debug:
        for i, tensor in enumerate(outputs):
            arr = np.array(tensor)
            print(f"[debug] output[{i}] shape={arr.shape} dtype={arr.dtype} "
                  f"min={arr.min() if arr.size else 'NA'} max={arr.max() if arr.size else 'NA'}",
                  flush=True)

    boxes, scores, classes = outputs[0], outputs[1], outputs[2]
    n_detections = int(np.array(outputs[3]).flatten()[0])
    if n_detections <= 0:
        return []

    input_w, input_h = imx500.get_input_size()
    boxes = np.array(boxes[:n_detections], dtype=np.float32)
    scores = np.array(scores[:n_detections], dtype=np.float32)
    classes = np.array(classes[:n_detections]).astype(int)

    # Boxes are x0, y0, x1, y1 in input-tensor pixels -> normalise to 0..1.
    boxes[:, [0, 2]] /= input_w
    boxes[:, [1, 3]] /= input_h

    detections: list[Detection] = []
    for box, score, class_id in zip(boxes, scores, classes):
        if score < threshold:
            continue
        x0, y0, x1, y1 = box
        # convert_inference_coords expects normalised (y0, x0, y1, x1).
        mapped = imx500.convert_inference_coords((y0, x0, y1, x1), metadata, picam2)
        name = labels[class_id] if 0 <= class_id < len(labels) else str(class_id)
        detections.append(Detection(mapped, int(class_id), float(score), name))
    return detections


def draw_scan_banner(frame, banner: dict | None) -> None:
    """Big top-centre banner: what the camera sees and which bin it belongs in."""
    if banner:
        title, subtitle, color = banner["title"], banner["sub"], KNOWN_COLOR
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


def draw_overlay(frame, detections, zone_mode: bool, banner: dict | None = None) -> None:
    if zone_mode:
        draw_zones(frame)

    lines: list[str] = []
    warnings: list[str] = []
    for det in detections or []:
        x, y, w, h = det.box
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        known = is_known_class(det.class_name)
        color = KNOWN_COLOR if known else UNKNOWN_COLOR
        label = format_recommendation(det.class_name, det.conf)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, label, x1, y1, color=color)
        lines.append(label)

        if zone_mode and known:
            center_x, center_y = get_object_center((x1, y1, x2, y2))
            zone_name = get_zone_from_center(center_x, frame.shape[1])
            if check_zone_correctness(det.class_name, zone_name):
                draw_label(frame, "Correct bin", center_x, center_y, color=KNOWN_COLOR)
            else:
                target_zone = expected_zone(det.class_name)
                recommended_bin = get_recommended_bin(det.class_name)
                warnings.append(f"Wrong bin: {det.class_name} -> {target_zone} ({recommended_bin})")
                draw_label(frame, f"Wrong bin: use {target_zone}", center_x, center_y, color=WRONG_COLOR)

    if zone_mode:
        panel = [f"Objects: {len(detections or [])}", *(lines[:5] or ["No object detected."])]
        if warnings:
            panel.extend(warnings[:3])
        draw_status_panel(frame, panel[:9])
    else:
        draw_scan_banner(frame, banner)


def main() -> int:
    args = parse_args()
    if not Path(args.model).exists():
        print(f"Model not found at {args.model}. Package the .rpk on the Pi first.")
        return 1

    labels = load_labels(args.labels)

    # IMX500 must be constructed before Picamera2 so firmware can be staged.
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    intrinsics.labels = labels
    if args.fps is not None:
        intrinsics.inference_rate = args.fps
    intrinsics.update_with_defaults()

    picam2 = Picamera2(imx500.camera_num)
    controls = {}
    if intrinsics.inference_rate:
        controls["FrameRate"] = intrinsics.inference_rate
    config = picam2.create_preview_configuration(controls=controls, buffer_count=12)

    imx500.show_network_fw_progress_bar()

    state = {
        "detections": None,
        "zone_mode": bool(args.zones),
        "banner": None,
        "debug_left": 1 if args.debug_outputs else 0,
    }

    def pre_callback(request):
        with MappedArray(request, "main") as m:
            draw_overlay(m.array, state["detections"], state["zone_mode"], state["banner"])

    if not args.headless:
        picam2.start(config, show_preview=Preview.QTGL)
        picam2.pre_callback = pre_callback
    else:
        picam2.start(config)

    frame_count = 0
    try:
        while True:
            metadata = picam2.capture_metadata()
            debug = state["debug_left"] > 0
            state["detections"] = parse_detections(
                imx500, picam2, metadata, labels, args.conf, debug=debug
            )
            if debug:
                state["debug_left"] -= 1

            # Scan-mode banner: show the best detection and hold it briefly so the
            # display does not flicker when confidence dips between frames.
            known = [d for d in state["detections"] or [] if is_known_class(d.class_name)]
            if known:
                best = max(known, key=lambda d: d.conf)
                state["banner"] = {
                    "title": f"{best.class_name.replace('_', ' ').upper()}  ({best.conf:.0%})",
                    "sub": f"Put it in: {get_recommended_bin(best.class_name)}",
                    "expires": time.monotonic() + BANNER_HOLD_S,
                }
            elif state["banner"] and time.monotonic() > state["banner"]["expires"]:
                state["banner"] = None

            if args.headless:
                dets = state["detections"] or []
                summary = f"Objects: {len(dets)} | " + ", ".join(
                    f"{d.class_name}:{d.conf:.2f}->{get_recommended_bin(d.class_name)}" for d in dets[:5]
                )
                print(summary or "No object detected.", flush=True)

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
