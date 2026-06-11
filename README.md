# Real-Time Waste Sorting Assistant Using YOLO and IMX500

A real-time edge-AI waste sorting assistant that detects common waste objects, recommends the correct recycling bin, and prepares the model path for Sony IMX500 AI camera deployment.

The project currently runs end-to-end on a MacBook and on a Raspberry Pi using the Pi camera. The Raspberry Pi version currently runs the custom YOLO `.pt` model on the Pi CPU. The final embedded target is to convert the trained model into the IMX500 `.rpk` format so inference can run on the Sony IMX500 accelerator.

## Current Project Status

This repository is ready for an intermediate demo.

Implemented:

- YOLO object detection training pipeline.
- Six waste classes.
- Image, video, webcam, and Raspberry Pi Picamera2 demo scripts.
- Recycling-bin recommendation logic.
- Optional virtual bin-zone checking.
- Dataset capture helper for collecting our own images.
- Temporary public dataset importer for early demonstration.
- Validation script and metric reporting.
- IMX500 export wrapper and deployment notes.
- Tests for bin mapping and zone logic.
- Documentation and presentation outline.

Important honesty note: the current trained model is a temporary intermediate-demo model. It proves the full system works, but it is not final-quality or 90%+ accurate yet. Accuracy should improve after collecting and labeling our own dataset from the actual demo setup.

## Project Goal

The system watches a table with waste objects, detects the object class, and tells the user which bin to use.

Example:

```text
Input camera frame -> YOLO detects plastic_bottle -> rule layer recommends Plastic / Packaging
```

Optional zone mode divides the camera frame into virtual table regions. If the object is in the wrong virtual bin zone, the program warns the user.

Example:

```text
plastic_bottle appears in paper zone -> Wrong bin: use Plastic / Packaging
```

## Target Classes

The custom project classes are:

| ID | Class |
| ---: | --- |
| 0 | `plastic_bottle` |
| 1 | `can` |
| 2 | `paper` |
| 3 | `cardboard` |
| 4 | `glass_jar` |
| 5 | `food_wrapper` |

## Bin Mapping

The detector predicts object class only. The bin recommendation is a separate rule layer stored in `configs/bins.yaml`.

| Detected class | Recommended bin |
| --- | --- |
| `plastic_bottle` | Plastic / Packaging |
| `can` | Metal / Recycling |
| `paper` | Paper |
| `cardboard` | Paper / Cardboard |
| `glass_jar` | Glass |
| `food_wrapper` | General Waste / Packaging |

This design keeps the neural network simple. The model learns visual object categories, and normal Python logic maps each class to the correct bin.

## Why YOLO?

YOLO was selected because the course requirement is object detection, not only image classification. A classifier would say "this frame contains a plastic bottle", but it would not localize the object. YOLO returns:

- object class,
- confidence score,
- bounding box coordinates.

That makes it suitable for:

- drawing boxes around waste objects,
- detecting multiple objects in one frame,
- checking object location against virtual bin zones,
- preparing a real-time camera demo.

## Underlying Model

The current project uses Ultralytics YOLO, with YOLOv8n as the default model.

Default training model:

```text
yolov8n.pt
```

Why YOLOv8n:

- It is lightweight.
- It trains quickly.
- It runs on a laptop webcam in real time.
- It is more realistic for edge deployment than large YOLO models.
- Ultralytics supports export paths including `format=imx`.

YOLO11n can also be used later:

```bash
python src/train.py --model yolo11n.pt --data dataset/data.yaml --epochs 80 --imgsz 640 --batch 16
```

For the intermediate demo, YOLOv8n is the safer choice because it is mature, lightweight, and already tested in this repository.

## System Architecture

High-level pipeline:

```text
Camera / image / video
        |
        v
YOLO object detector
        |
        v
Detections: class, confidence, bounding box
        |
        v
Bin logic: class -> recommended recycling bin
        |
        v
Optional zone logic: object center -> virtual bin zone -> correct/wrong
        |
        v
OpenCV display with boxes, labels, status panel, screenshots
```

Main modules:

| File | Purpose |
| --- | --- |
| `src/train.py` | Train YOLO model with Ultralytics. |
| `src/validate.py` | Evaluate trained model and report metrics. |
| `src/infer_image.py` | Run detection on image files or folders. |
| `src/infer_video.py` | Run detection on video files. |
| `src/webcam_demo.py` | Main MacBook/desktop webcam demo. |
| `src/pi_picamera_demo.py` | Raspberry Pi demo using Picamera2 camera frames. |
| `src/bin_logic.py` | Maps detected classes to bin recommendations. |
| `src/zone_logic.py` | Implements virtual left/center/right bin zones. |
| `src/capture_dataset.py` | Captures our own images for dataset collection. |
| `src/import_public_demo_dataset.py` | Imports temporary public demo dataset. |
| `src/export_imx500.py` | Exports trained YOLO model for IMX500 pipeline. |
| `src/split_dataset.py` | Optional local train/val/test split helper. |
| `src/utils.py` | Shared drawing, output, and config helpers. |

## Repository Structure

```text
waste-sorting-imx500/
  README.md
  requirements.txt
  configs/
    bins.yaml
    classes.yaml
  dataset/
    data.yaml
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
  src/
    train.py
    validate.py
    infer_image.py
    infer_video.py
    webcam_demo.py
    pi_picamera_demo.py
    bin_logic.py
    zone_logic.py
    capture_dataset.py
    import_public_demo_dataset.py
    export_imx500.py
    split_dataset.py
    utils.py
  models/
    best.pt
    demo_backups/
    exported/
  docs/
    annotation_guide.md
    dataset_description.md
    evaluation_results.md
    imx500_deployment_notes.md
    intermediate_demo_status.md
    public_dataset_demo.md
    technical_pipeline.md
    training_setup.md
  presentation/
    slides_outline.md
  tests/
    test_bin_logic.py
    test_zone_logic.py
```

## Dataset

### Final Dataset Plan

The final course dataset should be collected by us. The requirement is at least 100 labeled images, but the practical target is 150-250+ images.

Recommended image count:

| Class | Recommended images |
| --- | ---: |
| `plastic_bottle` | 30-40 |
| `can` | 30-40 |
| `paper` | 25-35 |
| `cardboard` | 30-40 |
| `glass_jar` | 25-35 |
| `food_wrapper` | 30-40 |

Images should include:

- different lighting,
- different backgrounds,
- different camera angles,
- different distances,
- object rotation,
- single-object scenes,
- multiple-object scenes,
- partial occlusion,
- objects placed in different virtual bin zones.

### YOLO Annotation Format

Labels use standard YOLO bounding-box format:

```text
class_id x_center y_center width height
```

All box values are normalized between `0` and `1`.

Example:

```text
0 0.512 0.481 0.220 0.610
```

This means class `0` (`plastic_bottle`) with normalized center and size.

### Dataset Folder Layout

```text
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  data.yaml
```

`dataset/data.yaml`:

```yaml
path: ./dataset
train: images/train
val: images/val
test: images/test

names:
  0: plastic_bottle
  1: can
  2: paper
  3: cardboard
  4: glass_jar
  5: food_wrapper
```

## How We Got A Temporary Dataset

For the intermediate demo, our own final dataset was not ready yet. To avoid showing only empty code, we used a temporary public waste-detection dataset.

Source:

```text
keremberke/garbage-object-detection
https://huggingface.co/datasets/keremberke/garbage-object-detection
```

Importer:

```bash
python src/import_public_demo_dataset.py
```

The source dataset uses broader garbage categories, so we remapped them into our project classes:

| Public class | Temporary project class |
| --- | --- |
| `plastic` | `plastic_bottle` |
| `metal` | `can` |
| `paper` | `paper` |
| `cardboard` | `cardboard` |
| `glass` | `glass_jar` |
| `biodegradable` | `food_wrapper` |

This remap is intentionally temporary and imperfect. For example, public `plastic` may include many plastic objects that are not bottles, and `biodegradable` is not the same as a food wrapper. This is the main reason current accuracy is limited.

Improved temporary split used for the demo model:

```text
train: 2757 images, 19238 boxes
val:   258 images, 849 boxes
test:  252 images, 924 boxes
```

## Current Model Accuracy

The active temporary model is:

```text
models/best.pt
```

It was trained from a larger public demo split:

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

Current validation result:

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

Interpretation:

- The software pipeline works.
- The model is good enough to demonstrate the concept.
- The model is not final quality.
- It should not be claimed as 90% accurate.
- Real accuracy improvement depends on collecting our own labeled images from the actual table/camera setup.

## Why Accuracy Is Not High Yet

The current model is weak because:

- The public dataset does not exactly match our six target classes.
- The class remap is broad and noisy.
- Some objects are visually similar, especially transparent bottles and glass jars.
- `food_wrapper` is visually diverse.
- `paper` and `cardboard` can look similar.
- The current model was trained quickly for an intermediate demo.
- The final dataset from our actual physical setup is not collected and labeled yet.

For the final project, we should collect images using the same camera angle, table, objects, lighting, and bin-zone layout used in the demo.

## Installation On MacBook

```bash
cd /Users/Nabeel/Desktop/waste-sorting-imx500
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train A Model

Recommended first training command:

```bash
python src/train.py \
  --model yolov8n.pt \
  --data dataset/data.yaml \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --name waste_sorting_yolov8n
```

If using Apple Silicon GPU:

```bash
python src/train.py \
  --model yolov8n.pt \
  --data dataset/data.yaml \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --device mps \
  --name waste_sorting_yolov8n
```

After training, copy the best weights:

```bash
cp runs/detect/waste_sorting_yolov8n/weights/best.pt models/best.pt
```

Or pass the trained path directly:

```bash
python src/webcam_demo.py --model runs/detect/waste_sorting_yolov8n/weights/best.pt --camera 0
```

## Validate A Model

```bash
python src/validate.py --model models/best.pt --data dataset/data.yaml --imgsz 640
```

Report:

- mAP50,
- mAP50-95,
- precision,
- recall,
- confusion matrix,
- per-class results.

## Run Image Inference

```bash
python src/infer_image.py \
  --model models/best.pt \
  --source examples/sample_images/
```

For current public demo examples:

```bash
python src/infer_image.py \
  --model models/best.pt \
  --source examples/public_demo_inputs \
  --conf 0.2 \
  --out examples/public_demo_outputs
```

## Run Video Inference

```bash
python src/infer_video.py \
  --model models/best.pt \
  --source examples/demo_videos/demo.mp4 \
  --out examples/demo_videos/annotated.mp4
```

## Run MacBook Webcam Demo

Main demo:

```bash
cd /Users/Nabeel/Desktop/waste-sorting-imx500
source .venv/bin/activate
python src/webcam_demo.py --model models/best.pt --camera 0 --conf 0.25 --zones
```

Controls:

```text
q or ESC = quit
s        = save screenshot
z        = toggle zone mode
h        = toggle help overlay
```

If there are too many false detections:

```bash
python src/webcam_demo.py --model models/best.pt --camera 0 --conf 0.40 --zones
```

If it misses objects:

```bash
python src/webcam_demo.py --model models/best.pt --camera 0 --conf 0.20 --zones
```

## Raspberry Pi Custom Model Demo

The project has also been copied to the Raspberry Pi:

```text
/home/pi/waste-sorting-project
```

The custom model runs on Raspberry Pi CPU using Picamera2 frames:

```text
Pi camera frame -> Picamera2 -> YOLO best.pt on CPU -> OpenCV overlay
```

This is not the final IMX500 accelerator path, but it proves that the same custom model can run on the Pi.

Run from the Raspberry Pi VNC terminal:

```bash
pkill -f pi_picamera_demo.py
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

If slow:

```bash
/home/pi/waste-sorting-venv/bin/python src/pi_picamera_demo.py \
  --model models/best.pt \
  --conf 0.25 \
  --imgsz 320 \
  --zones
```

Headless smoke test:

```bash
/home/pi/waste-sorting-venv/bin/python src/pi_picamera_demo.py \
  --model models/best.pt \
  --conf 0.25 \
  --imgsz 416 \
  --zones \
  --headless \
  --max-frames 2
```

## Raspberry Pi Setup Used

The Pi-side environment was prepared with:

```bash
sudo apt-get install -y python3-torch python3-torchvision python3-yaml python3-pandas
python3 -m venv --system-site-packages /home/pi/waste-sorting-venv
/home/pi/waste-sorting-venv/bin/pip install --upgrade pip setuptools wheel
/home/pi/waste-sorting-venv/bin/pip install --no-deps ultralytics==8.4.56
```

The IMX500 camera was verified with:

```bash
rpicam-hello --list-cameras
rpicam-hello -t 1000 --nopreview
```

Expected camera listing:

```text
0 : imx500 [4056x3040 10-bit RGGB]
```

## IMX500 Deployment Plan

The final embedded deployment should run the model on the Sony IMX500 AI camera, not only on the Raspberry Pi CPU.

The final path is:

```text
models/best.pt
        |
        v
Ultralytics IMX export on Linux
        |
        v
packerOut.zip + labels.txt
        |
        v
imx500-package on Raspberry Pi
        |
        v
custom .rpk model
        |
        v
Picamera2 IMX500 object detection demo
```

Primary export command:

```bash
yolo export model=models/best.pt format=imx data=dataset/data.yaml imgsz=640
```

Or through the project wrapper:

```bash
python src/export_imx500.py --model models/best.pt --data dataset/data.yaml --imgsz 640
```

Then on Raspberry Pi:

```bash
imx500-package -i packerOut.zip -o ./rpk_out
```

Known issue: direct export from macOS is not supported by Ultralytics for IMX export.

Observed error:

```text
IMX: export failure: Export only supported on Linux.
```

Therefore, the next final-deployment step should use:

- Ubuntu GPU machine, or
- Linux Docker environment, or
- Raspberry Pi/Linux environment if supported and fast enough.

## Tests

```bash
pytest tests/ -v
```

Tests cover:

- known class to bin mapping,
- unknown class fallback,
- zone boundary logic,
- correct/wrong zone checks.

## Demo Script For Professor

Suggested explanation:

```text
This is a real-time waste sorting assistant. We use YOLOv8n for object detection because it gives object class and bounding box location. A Python rule layer maps each detected class to a recycling bin. The optional zone mode checks whether the object is placed in the correct virtual bin area.

For the intermediate demo, we used a temporary public garbage dataset so that the full pipeline could be demonstrated before our own labeled dataset is complete. The current model is not final accuracy. The final version will use our own 100+ labeled images from the actual table setup.

The custom model runs on the MacBook and has also been copied to the Raspberry Pi where it runs with Picamera2. The IMX500 camera is detected and working. The final deployment step is converting the trained YOLO model to the IMX500 .rpk format using the Linux export and packaging pipeline.
```

## Limitations

- Current model is trained on a temporary public dataset, not final self-collected images.
- Current accuracy is not 90%+.
- Public class remapping is noisy.
- Glass and plastic can be visually similar.
- Paper and cardboard can be confused.
- Food wrappers are highly variable.
- Raspberry Pi `.pt` demo runs on CPU and may be slow.
- Final IMX500 `.rpk` deployment still needs Linux export and packaging.

## Future Work

- Collect and label our own dataset.
- Train with 80-100 epochs on the final dataset.
- Use Ubuntu GPU access for faster training and IMX export.
- Export and package a custom IMX500 `.rpk`.
- Improve lighting and fixed camera positioning.
- Add more classes.
- Add audio feedback.
- Add real physical bin detection instead of virtual zones.
- Add an automatic sorting actuator.
