# IMX500 On-Camera Deployment — Work Log / Handoff

Goal: make the custom waste-sorting YOLO model run **on the Sony IMX500 camera
accelerator** on the Raspberry Pi, instead of running the `.pt` model on the Pi
CPU. Achieved and verified on device.

## Environment

- Dev machine: MacBook, Apple M4 Pro (`arm64`), Docker Desktop (aarch64 engine,
  amd64 emulation available).
- Target: Raspberry Pi (aarch64, Bookworm) with Sony IMX500 AI camera. Reachable
  as ssh host `pi`. Project copy at `/home/pi/waste-sorting-project`.
- Pi already had `imx500-tools` / `imx500-package`, `picamera2` with the IMX500
  device classes, numpy, cv2 (system python3).

## Starting state

- `models/best.pt`: YOLOv8n, 6 classes (plastic_bottle, can, paper, cardboard,
  glass_jar, food_wrapper), trained on a remapped public garbage dataset
  (temporary demo model, mAP50 ≈ 0.25).
- Pi ran `src/pi_picamera_demo.py`: Picamera2 frames → `YOLO(best.pt)` on the Pi
  **CPU**. The IMX500 accelerator was unused. No `.rpk` existed.
- Direct `yolo export format=imx` on macOS fails: "Export only supported on
  Linux." Sony's converter is x86_64-Linux only.

## Solution architecture

```
models/best.pt
   │ (1) Docker amd64 (emulated) on the Mac: Ultralytics format=imx
   │     → MCT INT8 quantization → Sony imxconv-pt
   ▼
models/best_imx_model/packerOut.zip + labels.txt
   │ (2) imx500-package on the Pi (arm64)
   ▼
models/exported/imx/rpk_out/network.rpk
   │ (3) src/imx500_camera_demo.py — picamera2 IMX500 device
   ▼
Inference on the sensor; Pi reads boxes/scores/classes from frame metadata.
```

## Step by step

### 1. Confirmed amd64 emulation
`docker run --rm --platform linux/amd64 alpine uname -m` → `x86_64`. Good, so the
x86-only Sony toolchain can run under emulation on the M4.

### 2. Built the export image and hit a real Ultralytics dependency bug
First image installed `ultralytics` + IMX deps unpinned. `yolo export format=imx`
failed at:
```
ImportError: cannot import name 'MulticlassNMSOBB' from 'edgemdt_cl.pytorch'
```
Root cause (this is the important part for anyone reproducing):

- Ultralytics 8.4.90's IMX exporter pins **both** `edge-mdt-cl<1.1.0` **and**
  `edge-mdt-tpc>=1.2.0`.
- `edge-mdt-tpc>=1.2.0` → `model-compression-toolkit>=2.5` → `edge-mdt-cl>=1.1`.
- So the two pins are mutually unsatisfiable. Ultralytics' runtime
  `check_requirements` AutoUpdate thrashes the env (downgrades `edge-mdt-cl` to
  1.0.0, which lacks `MulticlassNMSOBB` that the installed MCT imports).

### 3. Resolved with a coherent pinned toolchain + a one-line Ultralytics patch
`docker/Dockerfile.imx500-export` installs the newest **coherent** set and relaxes
Ultralytics' obsolete upper bound so AutoUpdate stays quiet:
```
ultralytics 8.4.90
edge-mdt-cl 1.1.1
model-compression-toolkit 2.5.1
mct-quantizers 1.6.0
edge-mdt-tpc 1.3.0
imx500-converter[pt] 3.18.2
# patch: sed 's/"edge-mdt-cl<1.1.0"/"edge-mdt-cl"/' in ultralytics imx.py + exporter.py
```
The image has a build-time healthcheck that imports the exact modules Ultralytics
and MCT use at runtime, so a broken combo fails the build, not the export.

Resolver notes learned the hard way:
- `imx500-converter[pt]==3.17.3` pins `mct-quantizers~=1.6.0` **and**
  `edge-mdt-cl~=1.0.0` → conflicts with the MCT ≥2.5 group. Use converter
  **3.18.2**, which pairs with `mct-quantizers 1.6.0` + `edge-mdt-cl 1.1.1`.
- That converter pairs with **MCT 2.5.1** (not 2.6.0; 2.6.0 wants
  `mct-quantizers 1.7.0`).

### 4. Exported
```
yolo export model=models/best.pt format=imx data=dataset/data.yaml imgsz=640
```
INT8 calibration ran over the 258 `val` images (~1.75 it/s under emulation; whole
export ~10 min). Output `models/best_imx_model/`: `packerOut.zip`, `labels.txt`
(class order = data.yaml), `model_imx.onnx`, `dnnParams.xml`.

### 5. Packaged on the Pi
```
scp packerOut.zip labels.txt pi:.../models/exported/imx/
ssh pi 'cd .../models/exported/imx && imx500-package -i packerOut.zip -o rpk_out'
```
→ `rpk_out/network.rpk` (3.05 MB).

### 6. Wrote the on-camera runtime — `src/imx500_camera_demo.py`
Uses `picamera2.devices.IMX500`. Inference runs on the sensor; Python reads output
tensors from frame metadata via `imx500.get_outputs(metadata)`. Reuses the
project's `bin_logic` and `zone_logic` unchanged. Uses **system python3** (no
torch/ultralytics on the Pi).

Output tensor contract for the Ultralytics YOLOv8 IMX export (verified on device
with `--debug-outputs`):
```
output[0] boxes    (300,4)  x0,y0,x1,y1 in input-tensor pixels; divide by 640
output[1] scores   (300,)
output[2] class_id (300,)   values 0..5 → the 6 classes
output[3] count    (1,)     number of valid detections
```
`imx500.convert_inference_coords` expects normalized `(y0,x0,y1,x1)`, so the
decoder normalizes by input size then reorders `[1,0,3,2]` before mapping to
display pixels. (Decode logic cross-checked against Sony modlib's
`pp_od_yolo_ultralytics`.)

### 7. Verified on device
- Headless (`--headless --max-frames 20 --debug-outputs`): rpk loads, firmware
  uploads (~65s), loop runs on the accelerator, tensor shapes match the contract
  above (scores all 0 because the scene was empty — no physical object placed).
- Live preview (`--zones`, QTGL, DISPLAY=:0): libcamera configures the IMX500,
  streams 640x480, ran 40 frames, exited cleanly. No torch involved.

## Files added / changed

Added:
- `src/imx500_camera_demo.py` — on-camera runtime (accelerator path).
- `docker/Dockerfile.imx500-export` — reproducible amd64 export toolchain.
- `scripts/export_imx500_docker.sh` — build image + export (Mac).
- `scripts/deploy_to_pi.sh` — copy artifacts + imx500-package (Mac→Pi).
- `scripts/run_imx500_camera.sh` — run on camera (deployed to Pi).
- `docs/imx500_work_log.md` — this file.

Changed:
- `docs/imx500_deployment_notes.md` — rewritten as the working recipe.
- `README.md` — intro, IMX500 section, module table, status list.

On the Pi:
- `/home/pi/waste-sorting-project/src/imx500_camera_demo.py`
- `/home/pi/waste-sorting-project/run_imx500_camera.sh`
- `/home/pi/waste-sorting-project/models/exported/imx/{packerOut.zip,labels.txt,rpk_out/network.rpk}`

## Reproduce from scratch

```bash
# On the Mac (repo root):
scripts/export_imx500_docker.sh          # ~build + ~10 min export
scripts/deploy_to_pi.sh                   # package network.rpk on the Pi
# On the Pi desktop / VNC:
/home/pi/waste-sorting-project/run_imx500_camera.sh
```

## Limitations / next steps

- The model is the temporary public-dataset demo model (low mAP), so **live
  detections are only as good as that model**. The deployment path is
  model-agnostic: retrain on your own labeled data, rerun the three commands, get
  a new `.rpk`. No code changes.
- Firmware upload takes ~65s on each start (normal for IMX500).
- Export runs under amd64 emulation; on a native x86_64 Linux box it would be
  much faster. imgsz=640 is used; a smaller imgsz would raise on-camera FPS if
  needed (the sensor runs the NN; the Pi only post-processes, so it is already
  real-time).
- Not yet done: a live screenshot/GIF of a real detection (needs an object in
  front of the camera).
