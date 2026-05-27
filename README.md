# Real-Time Waste Sorting Assistant Using YOLO and IMX500

A real-time edge-AI waste sorting assistant that detects common waste objects with YOLO, recommends the correct recycling bin, and prepares the trained detector for Sony IMX500 deployment.

The project is desktop-testable first. Train and debug the detector with Ultralytics YOLO on a laptop webcam, then export the trained model using the IMX500 path once the dataset and model are ready.

## Features

- 6-class waste detector: `plastic_bottle`, `can`, `paper`, `cardboard`, `glass_jar`, `food_wrapper`.
- YOLOv8n default training path, with YOLO11n or other Ultralytics detector weights supported by CLI.
- Image, video, and webcam inference scripts.
- Bin recommendation layer for every known class.
- Graceful fallback for unknown detections from stock models, shown as `Unknown / no bin mapping`.
- Optional virtual bin-zone mode for correct/wrong placement feedback.
- Dataset capture helper for collecting your own images.
- IMX500 export wrapper using Ultralytics `format=imx` as the primary deployment path.

## Requirement Mapping

| Course requirement | Where it is covered |
| --- | --- |
| Build an object detection network | `src/train.py` with Ultralytics YOLO |
| Lightweight embedded model | Default `yolov8n.pt`; `yolo11n.pt` supported |
| IMX500 deployment/export files | `src/export_imx500.py`, `docs/imx500_deployment_notes.md` |
| Collect part of dataset ourselves | `src/capture_dataset.py`, `docs/dataset_description.md` |
| Minimum 100 labeled images | Dataset plan in `docs/dataset_description.md` |
| YOLO annotation format | `dataset/data.yaml`, `docs/annotation_guide.md` |
| Desktop testability | `src/infer_image.py`, `src/infer_video.py`, `src/webcam_demo.py` |
| Documentation and presentation | `docs/`, `presentation/slides_outline.md` |

## Installation

```bash
cd ~/Desktop/waste-sorting-imx500
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Structure

```text
dataset/
  images/
    train/
    val/
    test/
    raw/
  labels/
    train/
    val/
    test/
    raw/
  data.yaml
```

Each YOLO label file contains one object per line:

```text
class_id x_center y_center width height
```

All box values are normalized between `0` and `1`.

## Dataset Workflow

### Preferred: Roboflow or CVAT pre-split export

1. Capture your own images:

   ```bash
   python src/capture_dataset.py --camera 0 --out dataset/images/raw --prefix waste
   ```

2. Upload `dataset/images/raw/` to Roboflow or CVAT.
3. Annotate using exactly the 6 classes in `configs/classes.yaml`.
4. Export as YOLOv8 format with a 70/20/10 train/val/test split.
5. Unzip into `dataset/`, replacing `images/{train,val,test}` and `labels/{train,val,test}`.
6. Keep this repository's `dataset/data.yaml` if the exported YAML has different paths.
7. Train the model.

### Optional: local annotation and local split

1. Capture images:

   ```bash
   python src/capture_dataset.py --camera 0
   ```

2. Annotate locally in LabelImg, CVAT, or makesense.ai and save YOLO `.txt` labels to `dataset/labels/raw/`.
3. Split the dataset:

   ```bash
   python src/split_dataset.py --require-labels
   ```

4. Train the model.

## Training

```bash
python src/train.py \
  --model yolov8n.pt \
  --data dataset/data.yaml \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --name waste_sorting_yolov8n
```

After training, either copy the best weights:

```bash
cp runs/detect/waste_sorting_yolov8n/weights/best.pt models/best.pt
```

or pass the training run path directly to the inference scripts.

## Validation

```bash
python src/validate.py --model models/best.pt --data dataset/data.yaml --imgsz 640
```

Report these metrics in `docs/evaluation_results.md`:

- mAP50
- mAP50-95
- precision
- recall
- per-class performance
- confusion matrix

## Image Inference

```bash
python src/infer_image.py --model models/best.pt --source examples/sample_images/
```

Annotated images are saved to `examples/sample_images/out/` by default.

## Video Inference

```bash
python src/infer_video.py \
  --model models/best.pt \
  --source examples/demo_videos/demo.mp4 \
  --out examples/demo_videos/annotated.mp4
```

## Webcam Demo

```bash
python src/webcam_demo.py --model models/best.pt --camera 0 --conf 0.35
```

With virtual bin zones:

```bash
python src/webcam_demo.py --model models/best.pt --camera 0 --conf 0.35 --zones
```

Keyboard controls:

- `q` or `ESC`: quit
- `s`: save screenshot to `examples/screenshots/`
- `z`: toggle zone mode
- `h`: show or hide help overlay

Before training, you can smoke-test the UI with a stock model:

```bash
python src/webcam_demo.py --model yolov8n.pt --camera 0 --conf 0.4
```

Stock COCO classes such as `bottle` or `person` are intentionally shown as `Unknown / no bin mapping`; this confirms the demo does not crash before the custom waste model exists.

## IMX500 Export

Primary path:

```bash
python src/export_imx500.py --model models/best.pt --data dataset/data.yaml --imgsz 640
```

Expected output folder:

```text
models/exported/imx/
  packerOut.zip
  labels.txt
  model_quantized.onnx
```

Then copy the IMX files to the Raspberry Pi and package:

```bash
scp models/exported/imx/packerOut.zip models/exported/imx/labels.txt pi@<pi-ip>:~/waste-sorting/
ssh pi@<pi-ip> 'imx500-package -i ~/waste-sorting/packerOut.zip -o ~/waste-sorting/rpk_out'
```

The final `.rpk` can be tested with the Raspberry Pi IMX500 object-detection demo from the installed `imx500-all` tooling. See `docs/imx500_deployment_notes.md`.

Fallback export notes for ONNX, TensorFlow/SavedModel, TFLite, and manual MCT conversion are in `docs/imx500_fallback_mct.md`.

## Tests

```bash
pytest tests/ -v
```

The unit tests cover deterministic bin mapping and zone logic. They do not require a trained model.

## Limitations

- Transparent plastic bottles and glass jars can look similar.
- Paper and cardboard can confuse the model when both are flat or similarly colored.
- Food wrappers are visually diverse and need many examples.
- Low light, reflections, occlusion, and small objects reduce accuracy.
- IMX500 deployment may impose model size, quantization, and operation constraints.

## Future Work

- Add more waste classes.
- Improve dataset diversity and class balance.
- Add real bin detection instead of fixed virtual zones.
- Add audio feedback.
- Add multilingual labels.
- Add an automatic sorting actuator.

