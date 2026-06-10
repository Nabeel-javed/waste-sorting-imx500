# Temporary Public Dataset Demo

This branch uses a public dataset only for the intermediate demo.

Final course work should still use our own collected and labeled images. The public dataset is a temporary bridge so the project can show a working training and inference pipeline before the self-collected dataset is ready.

## Source

Dataset:

```text
keremberke/garbage-object-detection
https://huggingface.co/datasets/keremberke/garbage-object-detection
```

The Hugging Face dataset card says the dataset was exported from Roboflow and contains 10,464 images with COCO-format annotations. The repository downloads only the `valid.zip` and `test.zip` splits by default for a smaller demo run.

## Class Remap

The public dataset uses broad material classes, while the final project uses item-level classes. For the intermediate demo branch only, the importer applies this temporary remap:

| Public class | Project class |
| --- | --- |
| `plastic` | `plastic_bottle` |
| `metal` | `can` |
| `paper` | `paper` |
| `cardboard` | `cardboard` |
| `glass` | `glass_jar` |
| `biodegradable` | `food_wrapper` |

This is not semantically perfect. For example, `plastic` is broader than `plastic_bottle`, and `biodegradable` is not the same as `food_wrapper`. Use this only to demonstrate that the pipeline works.

## Build The Demo Dataset

```bash
python src/import_public_demo_dataset.py
```

Default output:

```text
dataset/images/train
dataset/images/val
dataset/images/test
dataset/labels/train
dataset/labels/val
dataset/labels/test
```

The downloaded zips and extracted source files are stored under:

```text
data/public/keremberke-garbage/
```

That folder is ignored by Git.

## Quick Training For Intermediate Demo

For a fast laptop run:

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

If `mps` fails, use CPU:

```bash
python src/train.py \
  --model yolov8n.pt \
  --data dataset/data.yaml \
  --epochs 3 \
  --imgsz 416 \
  --batch 4 \
  --device cpu \
  --name waste_sorting_public_demo_cpu
```

After training:

```bash
cp runs/detect/waste_sorting_public_demo/weights/best.pt models/best.pt
```

or pass the run path directly:

```bash
python src/webcam_demo.py --model runs/detect/waste_sorting_public_demo/weights/best.pt --camera 0 --conf 0.25 --zones
```

## What To Tell The Professor

- The project skeleton is complete and desktop-testable.
- The dataset and model shown in the intermediate demo are temporary public-data placeholders.
- The final dataset will be replaced with our own collected 100+ labeled images.
- The code already supports the final workflow: own dataset capture, YOLO training, validation, webcam demo, and IMX500 export.
