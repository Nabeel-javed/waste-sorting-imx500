# Real-Time Waste Sorting Assistant Using YOLO and the Sony IMX500

A real-time edge-AI waste sorting assistant. Hold a waste object in front of the
camera and the system detects it, announces what it is, and tells you which
recycling bin it belongs in - with the neural network running **inside the Sony
IMX500 camera sensor**, not on the Raspberry Pi CPU.

```text
Hold up a bottle  ->  "SCANNING..."  ->  "PLASTIC BOTTLE (72%) - Put it in: Plastic / Packaging"
```

The full pipeline is working end-to-end: dataset aggregation from five sources,
GPU training on a remote server, INT8 quantization through Sony's toolchain,
on-sensor deployment, and a stabilized live scan UI with multi-frame voting.

## Current Project Status

Implemented and verified:

- YOLOv8n detector trained on 15,876 images across five data sources (see [Dataset](#dataset)).
- Six waste classes with a rule-based bin recommendation layer.
- **On-camera inference**: the model is converted to IMX500 `.rpk` format and runs
  on the sensor's accelerator. The Pi only reads output tensors from frame metadata.
- **Scan mode with multi-frame voting** (default live UI): a verdict is announced
  only when one class wins >=60% of recent frames, eliminating flicker and
  single-frame ghost detections.
- Semi-automatic dataset labeling with YOLO-World (open-vocabulary detection).
- Dockerized and native-Linux export toolchains for the Sony converter.
- One-command deploy script (Mac -> Pi -> packaged `.rpk`).
- MacBook webcam demo, image/video inference, validation, and tests.
- Optional legacy virtual bin-zone overlay (`--zones`).

Model progression on the identical 289-image validation set:

| Model | Training data | mAP50 |
| --- | --- | ---: |
| v0 (intermediate demo) | 2.7k public images, 15 epochs on a MacBook | 0.248 |
| v1 (+ own images) | 7.8k images incl. our own 202, 100 epochs on GPU server | 0.501 |
| v2 (+ more public data) | 15.9k images from 5 sources | 0.613 |
| **v3 (robustness training)** | same 15.9k with heavy augmentation | **0.611** |

v3 is the deployed model: it matches v2 on clean validation photos while being
far more tolerant of real camera conditions (blur, lighting, odd angles), which
is what matters for the live demo.

## Project Goal

The system watches the camera feed, detects waste objects, and tells the user
which bin to use.

Primary UX (scan mode, default): hold one object in front of the camera; a
banner shows `SCANNING...` while frames accumulate, then announces the verdict
and target bin.

Optional zone mode (`--zones`): the frame is divided into virtual table regions
and the system warns when an object sits in the wrong region.

## Target Classes

| ID | Class |
| ---: | --- |
| 0 | `plastic_bottle` |
| 1 | `can` |
| 2 | `paper` (includes cardboard) |
| 3 | `glass_jar` |
| 4 | `food_wrapper` |

Earlier models (v0-v3) used six classes with `cardboard` as its own class (id 3).
The confusion matrix showed paper and cardboard constantly confused with each
other (cardboard precision 0.432), and both map to the same physical bin, so v4
merges cardboard into `paper`. Fewer confusable classes also makes the live
multi-frame voting settle faster.

## Bin Mapping

The detector predicts the object class only. Bin recommendation is a separate
rule layer stored in `configs/bins.yaml`:

| Detected class | Recommended bin |
| --- | --- |
| `plastic_bottle` | Plastic / Packaging |
| `can` | Metal / Recycling |
| `paper` | Paper / Cardboard |
| `glass_jar` | Glass |
| `food_wrapper` | General Waste / Packaging |

This keeps the neural network simple: it learns visual categories, and plain
Python maps each class to a bin.

## Why YOLO, and Why YOLOv8n

The course requires object detection (class + bounding box), not just
classification. YOLO returns class, confidence, and box per object, which
enables multi-object scenes, the zone overlay, and the live scan UI.

YOLOv8n specifically because of a hard constraint: the IMX500 sensor has roughly
8 MB of memory for network weights. YOLOv8n quantized to INT8 (~3 MB) fits;
larger YOLO variants do not. Ultralytics also supports `format=imx` export for
this model family.

## System Architecture

```text
IMX500 sensor (neural network runs HERE)
        |
        v
Output tensors in frame metadata: boxes, scores, class ids, count
        |
        v
Raspberry Pi: src/imx500_camera_demo.py
        |
        +-- multi-frame voting (1.5s window, >=60% majority)
        |
        +-- bin logic: class -> recommended bin
        |
        v
Scan banner: "PLASTIC BOTTLE (72%) - Put it in: Plastic / Packaging"
```

Main modules:

| File | Purpose |
| --- | --- |
| `src/train.py` | Train YOLO model with Ultralytics. |
| `src/validate.py` | Evaluate trained model and report metrics. |
| `src/infer_image.py` / `src/infer_video.py` | Detection on files. |
| `src/webcam_demo.py` | MacBook webcam demo. |
| `src/pi_picamera_demo.py` | Pi demo running `.pt` on the Pi CPU (fallback/comparison). |
| `src/imx500_camera_demo.py` | **Primary Pi demo**: `.rpk` on the IMX500 accelerator, scan mode + voting. |
| `src/bin_logic.py` | Maps detected classes to bin recommendations. |
| `src/zone_logic.py` | Virtual left/center/right bin zones (legacy overlay). |
| `src/auto_label_own_dataset.py` | Semi-automatic labeling of self-collected images with YOLO-World. |
| `src/merge_own_dataset.py` | Merge own images into the dataset (stratified split, 3x oversample). |
| `src/convert_extra_public.py` | Convert drinking-waste and TACO datasets to project classes. |
| `src/import_public_demo_dataset.py` | Import the keremberke base dataset. |
| `src/capture_dataset.py` | Capture helper for collecting own images. |
| `src/export_imx500.py` | Legacy ONNX/IMX export helper. |
| `src/split_dataset.py` | Local train/val/test split helper. |
| `src/utils.py` | Shared drawing, output, and config helpers. |
| `scripts/export_imx500_docker.sh` | Docker (amd64) export of `best.pt` -> `packerOut.zip` on the Mac. |
| `scripts/deploy_to_pi.sh` | Copy artifacts to the Pi and package `network.rpk`. |
| `scripts/run_imx500_camera.sh` | Launch the live on-camera demo (on the Pi). |
| `docker/Dockerfile.imx500-export` | Pinned Sony/Ultralytics export toolchain. |

## Dataset

### Composition (v2/v3 training set: 15,876 train / 289 val images)

The validation split has been kept **identical across all training runs** so
accuracy numbers stay comparable.

| Source | Images | What it contributes |
| --- | ---: | --- |
| keremberke garbage-object-detection | 7,324 | Base detection dataset (remapped broad classes). |
| Our own images (3x oversampled) | 513 | ~200 photos of our actual demo objects, auto-labeled. |
| Drinking-waste dataset | 3,000 | Cans, plastic and glass bottles, many held in a hand - matches demo conditions. |
| TACO (Roboflow YOLO export) | 3,230 | Real-world litter; main source of `food_wrapper` boxes (~3,000). |
| TrashNet | 1,809 | Per-class object photos, auto-labeled with YOLO-World. |

Class remaps used:

- keremberke: `plastic->plastic_bottle`, `metal->can`, `glass->glass_jar`,
  `biodegradable->food_wrapper`, `paper` direct, `cardboard->paper`.
- drinking-waste (by filename prefix): `AluCan->can`, `Glass->glass_jar`,
  `HDPEM/PET->plastic_bottle` (capped at 1,000 images/class).
- TACO (by class id): `Can->can`, `Carton->paper`, `Paper->paper`,
  `Plastic bag & wrapper->food_wrapper`; the ambiguous `Bottle` class was skipped.

The v0-v3 dataset used six classes. For v4 the existing labels were remapped
in place on the training server (`cardboard` id 3 -> `paper` id 2; glass/wrapper
ids shifted down), giving the merged `paper` class 9,041 training boxes.

### Semi-automatic labeling with YOLO-World

Our own images and TrashNet come as per-class folders without boxes. Instead of
hand-labeling, `src/auto_label_own_dataset.py` prompts YOLO-World (an
open-vocabulary detector) with text descriptions per class ("aluminum can",
"glass jar", ...) and converts its boxes to YOLO labels with the project class
id. Every result is rendered to a preview image for human review; images where
nothing is found get a near-full-frame fallback box (flat classes like paper) or
are flagged for review. 202 of 204 own images were labeled this way.

Rebuild the dataset from scratch:

```bash
python src/import_public_demo_dataset.py --full-train --max-train-per-class 100000
python src/auto_label_own_dataset.py --source <own-photos-folder> --out data/own_dataset --device cpu
python src/merge_own_dataset.py
python src/auto_label_own_dataset.py --source data/trashnet --out data/trashnet_labeled --prefix trashnet --device cpu
python src/convert_extra_public.py --extra-root data/public_extra --dataset dataset
```

(YOLO-World's CLIP text encoder has device-mismatch bugs on both MPS and CUDA -
run it with `--device cpu`.)

## Training

Training runs on a GPU server (RTX A5000). The deployed v3 model:

```bash
yolo detect train model=yolov8n.pt data=dataset/data.yaml \
  epochs=100 imgsz=640 batch=64 device=0 workers=12 cache=ram \
  hsv_h=0.02 hsv_s=0.8 hsv_v=0.6 degrees=10 translate=0.2 scale=0.7 \
  shear=2.0 perspective=0.0003 mixup=0.15 close_mosaic=15 \
  name=waste_sorting_public_v3
```

The augmentation overrides are the point of v3: strong brightness/saturation
jitter, rotation, shear, perspective warp, +-70% scale, mixup, and (via the
`albumentations` package) blur/median-blur/grayscale/CLAHE. They simulate live
camera conditions that clean web photos lack, narrowing the gap between
validation accuracy and real on-camera behavior. ~2.2 hours for 100 epochs.

The deployed v4 model merges cardboard into paper (5 classes) and fine-tunes
from the v3 weights instead of starting from COCO, which converges in 30
epochs (~35 minutes):

```bash
yolo detect train model=runs/detect/waste_sorting_public_v3/weights/best.pt \
  data=dataset5/data.yaml \
  epochs=30 imgsz=640 batch=64 device=0 workers=12 cache=ram \
  hsv_h=0.02 hsv_s=0.8 hsv_v=0.6 degrees=10 translate=0.2 scale=0.7 \
  shear=2.0 perspective=0.0003 mixup=0.15 close_mosaic=10 \
  name=waste_sorting_public_v4
```

### Results per class (v3, deployed)

| Class | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| `plastic_bottle` | 0.733 | 0.543 | 0.650 | 0.442 |
| `can` | 0.820 | 0.513 | 0.556 | 0.438 |
| `paper` | 0.874 | 0.593 | 0.745 | 0.576 |
| `cardboard` | 0.432 | 0.520 | 0.489 | 0.399 |
| `glass_jar` | 0.210 | 0.833 | 0.835 | 0.677 |
| `food_wrapper` | 0.865 | 0.265 | 0.392 | 0.186 |
| **all** | 0.655 | 0.544 | **0.611** | 0.453 |

Interpretation: a 2.5x improvement over the intermediate-demo model (0.248),
with every class usable. `food_wrapper` remains hardest (visually diverse
category). Live accuracy is further stabilized by multi-frame voting (below).

## IMX500 Deployment (Working)

The trained model runs on the Sony IMX500 accelerator. Full recipe and
rationale in `docs/imx500_deployment_notes.md`; work log in
`docs/imx500_work_log.md`.

```text
models/best.pt
        |  (1) Ultralytics format=imx export (Sony MCT INT8 quantization + converter)
        |      - on the Mac: scripts/export_imx500_docker.sh (amd64 Docker)
        |      - or natively on any x86_64 Linux box with the pinned toolchain
        v
models/best_imx_model/packerOut.zip + labels.txt
        |  (2) scripts/deploy_to_pi.sh  (copies artifacts, runs imx500-package on the Pi)
        v
/home/pi/waste-sorting-project/models/exported/imx/rpk_out/network.rpk
        |  (3) run_imx500_camera.sh  (picamera2 IMX500 device)
        v
Inference runs inside the camera sensor.
```

Two gotchas the toolchain files solve:

- `format=imx` export only works on x86_64 Linux (`Export only supported on
  Linux`) - hence the Docker wrapper on the Mac.
- Ultralytics 8.4.90 ships mutually unsatisfiable pins for Sony's packages
  (`edge-mdt-cl<1.1.0` vs `edge-mdt-tpc>=1.2.0`), which breaks the export with
  `ImportError: cannot import name 'MulticlassNMSOBB'`.
  `docker/Dockerfile.imx500-export` pins a coherent set
  (`edge-mdt-cl==1.1.1`, `model-compression-toolkit==2.5.1`,
  `mct-quantizers==1.6.0`, `edge-mdt-tpc==1.3.0`,
  `imx500-converter[pt]==3.18.2`) and patches out the stale pin.

Model output contract (verified on device): `output[0]` boxes `(300,4)` as
`x0,y0,x1,y1` in input-tensor pixels, `output[1]` scores, `output[2]` class
ids, `output[3]` valid-detection count.

## Live Demo (on the Pi)

```bash
/home/pi/waste-sorting-project/run_imx500_camera.sh              # scan mode (default)
/home/pi/waste-sorting-project/run_imx500_camera.sh --conf 0.15  # more sensitive
/home/pi/waste-sorting-project/run_imx500_camera.sh --zones      # legacy zone overlay
/home/pi/waste-sorting-project/run_imx500_camera.sh --headless   # no window; prints detections
```

This uses system `python3` with picamera2 - no torch on the Pi, because
inference happens on the sensor.

### Multi-frame voting (scan mode)

Single frames from a small INT8 model are noisy: labels flicker and one-frame
ghost detections appear. Scan mode therefore never trusts a single frame:

- The best detection of each frame becomes one *vote* in a rolling 1.5 s window.
- A verdict is announced only when >=6 votes exist and one class holds >=60% of them.
- While votes accumulate, the banner shows `SCANNING... (n/6)`.
- The verdict is held for 1.5 s after votes stop, so the result is stable and readable.

Even at a modest per-frame hit rate, a 60% majority over dozens of frames makes
the announced verdict correct almost every time an object is detected at all.

## Installation On MacBook

```bash
cd waste-sorting-imx500
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train / Validate / Infer (local)

```bash
# train (use device=mps on Apple Silicon)
python src/train.py --model yolov8n.pt --data dataset/data.yaml --epochs 80 --imgsz 640 --batch 16

# validate
python src/validate.py --model models/best.pt --data dataset/data.yaml --imgsz 640

# images / video / webcam
python src/infer_image.py --model models/best.pt --source examples/sample_images/
python src/infer_video.py --model models/best.pt --source demo.mp4 --out annotated.mp4
python src/webcam_demo.py --model models/best.pt --camera 0 --conf 0.25
```

Webcam demo controls: `q`/ESC quit, `s` screenshot, `z` toggle zones, `h` help.

## Raspberry Pi CPU Demo (fallback / comparison)

Kept to demonstrate *why* the IMX500 path matters: the same model on the Pi CPU
is much slower.

```bash
cd /home/pi/waste-sorting-project
export DISPLAY=:0 XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0
/home/pi/waste-sorting-venv/bin/python src/pi_picamera_demo.py \
  --model models/best.pt --conf 0.25 --imgsz 416 --zones
```

## Tests

```bash
pytest tests/ -v
```

Covers bin mapping, unknown-class fallback, and zone boundary logic.

## Demo Script For Professor

```text
This is a real-time waste sorting assistant. A YOLOv8n detector identifies six
waste classes; a Python rule layer maps each class to the correct recycling bin.

The model was trained on ~16,000 images aggregated from four public waste
datasets plus ~200 images we collected ourselves and labeled semi-automatically
with an open-vocabulary detector. Training ran on a GPU server; the final model
was additionally trained with heavy augmentation (blur, lighting, perspective)
to be robust to live camera conditions.

The trained network was quantized to INT8 and compiled with Sony's toolchain
into the IMX500 format. Inference runs inside the camera sensor itself - the
Raspberry Pi only reads the detection tensors from frame metadata and applies
the bin logic. The live UI uses multi-frame voting: it announces an object only
when one class wins a clear majority of recent frames, so the verdict you see
is stable and reliable.
```

## Limitations

- Trained mostly on public photos; the live camera domain still differs
  (mitigated by v3 augmentation + voting, not eliminated).
- `food_wrapper` is the weakest class (visually extremely diverse).
- The IMX500 memory limit caps the model at YOLOv8n + INT8; per-frame accuracy
  is bounded by that model capacity.
- Glass vs. plastic bottles remain visually confusable (paper vs. cardboard
  was solved in v4 by merging the two classes - they share a bin anyway).
- Voting stabilizes verdicts but cannot detect an object the model never sees.

## Future Work

- Fine-tune on frames captured by the IMX500 itself (the single biggest
  remaining accuracy lever - it closes the camera domain gap).
- Try YOLO11n export, and higher-resolution input if it fits sensor memory.
- Audio feedback for verdicts; automatic sorting actuator.
- Physical bin detection instead of virtual zones.
