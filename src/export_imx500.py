from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO model artifacts for IMX500 or fallback formats.")
    parser.add_argument("--model", default="models/best.pt", help="Path to trained weights.")
    parser.add_argument("--data", default="dataset/data.yaml", help="Dataset YAML for INT8 calibration/export.")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size.")
    parser.add_argument("--out", default="models/exported", help="Output root folder.")
    parser.add_argument(
        "--format",
        default="imx",
        choices=("imx", "onnx", "tflite", "saved_model"),
        help="Export format. imx is the primary IMX500 path.",
    )
    return parser.parse_args()


def replace_path(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found at {model_path}. Train first and copy best.pt to models/best.pt.")
        return 1

    export_result = YOLO(str(model_path)).export(format=args.format, data=args.data, imgsz=args.imgsz)
    exported_path = Path(export_result)
    output_root = Path(args.out)
    destination_dir = output_root / args.format

    if exported_path.exists():
        if exported_path.is_dir():
            replace_path(exported_path, destination_dir)
        else:
            destination_dir.mkdir(parents=True, exist_ok=True)
            replace_path(exported_path, destination_dir / exported_path.name)

    print(f"Export complete. Files saved under: {destination_dir}")
    if args.format == "imx":
        print("\nNext Pi-side steps:")
        print("scp models/exported/imx/packerOut.zip models/exported/imx/labels.txt pi@<pi-ip>:~/waste-sorting/")
        print("ssh pi@<pi-ip> 'imx500-package -i ~/waste-sorting/packerOut.zip -o ~/waste-sorting/rpk_out'")
        print("Then run the Raspberry Pi IMX500 object detection demo with the generated network.rpk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

