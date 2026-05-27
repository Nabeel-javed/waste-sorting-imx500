from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import create_output_dirs, draw_status_panel, next_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture raw dataset images from a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--out", default="dataset/images/raw", help="Output folder for captured images.")
    parser.add_argument("--prefix", default="img", help="Filename prefix.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.out)
    create_output_dirs(output_dir)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"Cannot open camera {args.camera}. Try --camera 1.")
        return 1

    index = next_index(output_dir, args.prefix, ".jpg")
    print("SPACE saves an image. ESC quits.")

    while True:
        ok, frame = capture.read()
        if not ok:
            print("Camera frame capture failed.")
            break

        draw_status_panel(
            frame,
            [
                f"Output: {output_dir}",
                f"Next: {args.prefix}_{index:04d}.jpg",
                "SPACE: save image",
                "ESC: quit",
            ],
        )
        cv2.imshow("Dataset Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:
            image_path = output_dir / f"{args.prefix}_{index:04d}.jpg"
            cv2.imwrite(str(image_path), frame)
            print(f"Saved {image_path}")
            index += 1

    capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
