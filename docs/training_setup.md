# Training Setup

## Model

Recommended first model:

```bash
yolov8n.pt
```

Alternative:

```bash
yolo11n.pt
```

Both are lightweight Ultralytics YOLO detectors suitable for a desktop-first embedded prototype.

## Hyperparameters

| Parameter | Default |
| --- | --- |
| Epochs | 80 |
| Image size | 640 |
| Batch size | 16 |
| Initial weights | COCO-pretrained YOLO nano model |

## Command

```bash
python src/train.py --model yolov8n.pt --data dataset/data.yaml --epochs 80 --imgsz 640 --batch 16
```

## Transfer Learning Rationale

The project dataset is small, so training from a COCO-pretrained nano model is more practical than training from scratch. The pretrained backbone already contains general visual features, and fine-tuning adapts the model to the 6 waste classes.

## Augmentations

Ultralytics applies common object-detection augmentations during training. Record the final augmentation settings here if they are changed from defaults.

## Output

Expected training folder:

```text
runs/detect/waste_sorting_yolov8n/
  weights/
    best.pt
    last.pt
  results.png
  confusion_matrix.png
  PR_curve.png
  F1_curve.png
```

Copy the final model into `models/best.pt` after training:

```bash
cp runs/detect/waste_sorting_yolov8n/weights/best.pt models/best.pt
```

