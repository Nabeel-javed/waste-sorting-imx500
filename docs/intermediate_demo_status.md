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
