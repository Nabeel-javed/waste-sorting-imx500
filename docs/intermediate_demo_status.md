# Intermediate Demo Status

Date: 2026-06-10

Branch:

```bash
demo/public-dataset
```

## Current Status

The project is ready to show as a working desktop-first YOLO waste-sorting pipeline:

- Complete source-code skeleton.
- Dataset capture script for own images.
- Temporary public dataset importer.
- YOLO dataset structure.
- YOLO training script.
- YOLO validation script.
- Image/video/webcam inference scripts.
- Bin recommendation logic.
- Optional wrong-bin zone logic.
- IMX500 export wrapper.
- Documentation and presentation outline.

The current model is temporary and trained only for the intermediate demo. It should not be presented as the final course model.

## Public Demo Dataset

Source:

```text
keremberke/garbage-object-detection
https://huggingface.co/datasets/keremberke/garbage-object-detection
```

Imported with:

```bash
python src/import_public_demo_dataset.py
```

Result:

```text
train: 531 images, 4447 boxes
val:   108 images, 390 boxes
test:  108 images, 477 boxes
```

A stronger public-data demo dataset can be built with:

```bash
python src/import_public_demo_dataset.py \
  --full-train \
  --max-train-per-class 500 \
  --max-val-per-class 80 \
  --max-test-per-class 80
```

Result used for the improved intermediate demo model:

```text
train: 2757 images, 19238 boxes
val:   258 images, 849 boxes
test:  252 images, 924 boxes
```

Temporary class remap:

| Public class | Project class |
| --- | --- |
| `plastic` | `plastic_bottle` |
| `metal` | `can` |
| `paper` | `paper` |
| `cardboard` | `cardboard` |
| `glass` | `glass_jar` |
| `biodegradable` | `food_wrapper` |

Important limitation: this remap is broad and imperfect. It exists only to demonstrate the pipeline before our own collected dataset is ready.

## Demo Training Run

Command:

```bash
python src/train.py \
  --model yolov8n.pt \
  --data dataset/data.yaml \
  --epochs 5 \
  --imgsz 416 \
  --batch 8 \
  --device mps \
  --name waste_sorting_public_demo
```

Output:

```text
runs/detect/waste_sorting_public_demo/weights/best.pt
models/best.pt
```

## Validation Results

Command:

```bash
python src/validate.py --model models/best.pt --data dataset/data.yaml --imgsz 416
```

Result:

```text
mAP50:     0.1846
mAP50-95:  0.1220
Precision: 0.4321
Recall:    0.2723
```

Per-class:

| Class | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| `plastic_bottle` | 0.5650 | 0.0882 | 0.1606 | 0.0862 |
| `can` | 0.3606 | 0.4557 | 0.3736 | 0.2496 |
| `paper` | 1.0000 | 0.0000 | 0.0953 | 0.0736 |
| `cardboard` | 0.1732 | 0.4500 | 0.2002 | 0.1556 |
| `glass_jar` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `food_wrapper` | 0.0620 | 0.3673 | 0.0934 | 0.0449 |

The metrics are low because this is a short five-epoch run on a temporary public class remap. The useful part for the intermediate demo is that the full pipeline works end to end.

## Improved Demo Model

A stronger temporary model was trained from the larger public train split. Training was stopped after the metric improved enough for the intermediate demo.

Command:

```bash
python src/train.py \
  --model yolov8n.pt \
  --data dataset/data.yaml \
  --epochs 15 \
  --imgsz 512 \
  --batch 8 \
  --device mps \
  --name waste_sorting_public_demo_full15
```

Active model:

```text
models/best.pt
```

Backup of the earlier 5-epoch model:

```text
models/demo_backups/best_public_demo_5epochs.pt
```

Improved validation result:

```text
mAP50:     0.2480
mAP50-95:  0.1606
Precision: 0.3404
Recall:    0.3151
```

Per-class:

| Class | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| `plastic_bottle` | 0.3530 | 0.3677 | 0.2931 | 0.1864 |
| `can` | 0.3958 | 0.5017 | 0.4113 | 0.2824 |
| `paper` | 0.4741 | 0.3947 | 0.3657 | 0.2657 |
| `cardboard` | 0.0839 | 0.2500 | 0.0719 | 0.0330 |
| `glass_jar` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `food_wrapper` | 0.3953 | 0.0612 | 0.0978 | 0.0355 |

## Demo Commands

Image inference:

```bash
python src/infer_image.py --model models/best.pt --source examples/public_demo_inputs --conf 0.2 --out examples/public_demo_outputs
```

Webcam demo:

```bash
python src/webcam_demo.py --model models/best.pt --camera 0 --conf 0.25
```

Webcam demo with virtual bin zones:

```bash
python src/webcam_demo.py --model models/best.pt --camera 0 --conf 0.25 --zones
```

Keyboard controls:

- `q` or `ESC`: quit
- `s`: save screenshot
- `z`: toggle zone mode
- `h`: toggle help overlay

## Raspberry Pi / IMX500 Demo Status

Pi checked at:

```text
172.20.10.3
```

The Raspberry Pi can be reached over SSH:

```bash
ssh pi@172.20.10.3
```

The IMX500 camera is detected:

```bash
rpicam-hello --list-cameras
```

Expected camera line:

```text
0 : imx500 [4056x3040 10-bit RGGB]
```

Headless camera test:

```bash
rpicam-hello -t 1000 --nopreview
```

For the intermediate demo, the custom waste model should be shown on the MacBook desktop demo. The Pi can be shown as hardware progress: IMX500 connected, camera working, and stock IMX500 model runnable. This is honest because the custom YOLO-to-IMX500 packaging step still needs the Linux/Sony/Raspberry Pi export pipeline.

The direct custom export from macOS is blocked by Ultralytics:

```text
IMX: export failure: Export only supported on Linux.
```

Next step after the intermediate demo:

```bash
yolo export model=models/best.pt format=imx data=dataset/data.yaml imgsz=640
imx500-package -i packerOut.zip -o ./rpk_out
```

If a VNC live Pi proof is needed, use the patched stock IMX500 demo on the Pi:

```bash
ssh pi@172.20.10.3
export XDG_RUNTIME_DIR=/run/user/1000
export WAYLAND_DISPLAY=wayland-0
export DISPLAY=:0
python3 /home/pi/waste-sorting/imx500_stock_demo_qtgl.py \
  --model /usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk \
  --threshold 0.45
```

This runs a stock COCO detector on the IMX500, not the custom waste model. It is only for proving the AI camera path.

## Custom Model On Raspberry Pi CPU

The custom `models/best.pt` model can also run directly on the Raspberry Pi through Python. This uses the Raspberry Pi CPU with Picamera2 frames, not the IMX500 accelerator.

Pi setup used:

```bash
sudo apt-get install -y python3-torch python3-torchvision python3-yaml python3-pandas
python3 -m venv --system-site-packages /home/pi/waste-sorting-venv
/home/pi/waste-sorting-venv/bin/pip install --upgrade pip setuptools wheel
/home/pi/waste-sorting-venv/bin/pip install --no-deps ultralytics==8.4.56
```

Project copy on Pi:

```text
/home/pi/waste-sorting-project
```

Headless smoke test:

```bash
cd /home/pi/waste-sorting-project
/home/pi/waste-sorting-venv/bin/python src/pi_picamera_demo.py \
  --model models/best.pt \
  --conf 0.25 \
  --imgsz 416 \
  --zones \
  --headless \
  --max-frames 2
```

Live VNC demo command:

```bash
cd /home/pi/waste-sorting-project
export DISPLAY=:0
export XDG_RUNTIME_DIR=/run/user/1000
export WAYLAND_DISPLAY=wayland-0
/home/pi/waste-sorting-venv/bin/python src/pi_picamera_demo.py \
  --model models/best.pt \
  --conf 0.25 \
  --imgsz 416 \
  --zones
```

This is acceptable for showing that the same custom model has been copied to the Pi. It will be slower than the MacBook. The final embedded target remains the IMX500 `.rpk` export path.

## What To Tell The Professor

- The project codebase is complete enough for desktop testing.
- A temporary public dataset was used only for the intermediate demo because own-data collection is still in progress.
- The final project will replace this dataset with our own 100+ labeled images.
- The model is lightweight YOLOv8n, suitable for edge deployment work.
- The repository already includes the IMX500 export path:

  ```bash
  python src/export_imx500.py --model models/best.pt --data dataset/data.yaml --imgsz 640
  ```

- The final IMX500 deployment will follow the professor/Sony/Raspberry Pi conversion and packaging pipeline.
