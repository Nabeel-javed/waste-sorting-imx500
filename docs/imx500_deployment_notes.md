# IMX500 Deployment Notes

Status: **working**. The custom waste model runs on the Sony IMX500 accelerator
(not the Pi CPU). This document is the end-to-end recipe.

Pipeline:

```text
models/best.pt
      |  (1) Docker amd64 export on the Mac  -> Ultralytics format=imx + Sony converter
      v
models/best_imx_model/packerOut.zip + labels.txt
      |  (2) imx500-package on the Raspberry Pi
      v
models/exported/imx/rpk_out/network.rpk
      |  (3) src/imx500_camera_demo.py with picamera2 IMX500 device
      v
Inference runs inside the camera; the Pi only reads boxes/scores/classes.
```

## 1. Export on the Mac (Docker)

Sony's converter toolchain is x86_64-Linux only, so on Apple Silicon it runs in
an amd64 container under emulation. One command from the repo root:

```bash
scripts/export_imx500_docker.sh
# = build docker/Dockerfile.imx500-export, then:
#   yolo export model=models/best.pt format=imx data=dataset/data.yaml imgsz=640
```

INT8 quantization runs over the `val` calibration images; under emulation this
takes roughly 10-15 minutes. Output lands in `models/best_imx_model/`:

```text
packerOut.zip      <- goes to the Pi
labels.txt         <- goes to the Pi (class order = dataset/data.yaml)
model_imx.onnx     <- quantized model (reference)
dnnParams.xml
```

### Toolchain pinning (important)

Ultralytics 8.4.90's IMX exporter has a self-contradictory requirement set: it
pins `edge-mdt-cl<1.1.0` **and** `edge-mdt-tpc>=1.2.0`, but the latter pulls
`model-compression-toolkit>=2.5`, which requires `edge-mdt-cl>=1.1`. A plain
`pip install` therefore leaves a broken environment:

```text
ImportError: cannot import name 'MulticlassNMSOBB' from 'edgemdt_cl.pytorch'
```

`docker/Dockerfile.imx500-export` fixes this by installing the newest coherent
set and relaxing Ultralytics' obsolete upper bound so its runtime
`check_requirements` does not downgrade the environment:

```text
ultralytics 8.4.90
edge-mdt-cl 1.1.1
model-compression-toolkit 2.5.1
mct-quantizers 1.6.0
edge-mdt-tpc 1.3.0
imx500-converter[pt] 3.18.2
```

## 2. Package on the Raspberry Pi

`imx500-package` is provided by the Pi's `imx500-tools` package (arm64). From the
Mac, one command copies the artifacts and packages them:

```bash
scripts/deploy_to_pi.sh          # uses ssh host "pi"
```

Equivalent manual steps:

```bash
scp models/best_imx_model/packerOut.zip models/best_imx_model/labels.txt \
    pi:/home/pi/waste-sorting-project/models/exported/imx/
ssh pi 'cd /home/pi/waste-sorting-project/models/exported/imx && \
        imx500-package -i packerOut.zip -o rpk_out'
# -> rpk_out/network.rpk
```

## 3. Run on the camera

From the Raspberry Pi desktop / VNC terminal:

```bash
/home/pi/waste-sorting-project/run_imx500_camera.sh            # live preview + zones
/home/pi/waste-sorting-project/run_imx500_camera.sh --headless # prints detections
```

This calls `src/imx500_camera_demo.py`, which uses `picamera2.devices.IMX500`.
Inference runs on the sensor; the Pi reads output tensors from frame metadata and
applies the same `bin_logic` / `zone_logic` used by the desktop demo. It uses
system `python3` (picamera2 + numpy + cv2) - no torch/ultralytics on the Pi.

### Model output contract

The Ultralytics YOLOv8 IMX export produces four output tensors, verified on
device with `--debug-outputs`:

```text
output[0] boxes    (300, 4)  x0,y0,x1,y1 in input-tensor pixels (divide by 640)
output[1] scores   (300,)
output[2] class_id (300,)     values 0..5 -> the six waste classes
output[3] count    (1,)       number of valid detections
```

`convert_inference_coords` expects normalized `(y0, x0, y1, x1)`, so the decoder
normalizes by input size and reorders before mapping to display pixels.

## Notes

- Firmware upload to the IMX500 takes ~60-70s on each start ("Network Firmware
  Upload" progress bar). This is normal.
- The current `best.pt` is the temporary public-dataset demo model (low mAP), so
  live detections are only as good as that model. The deployment path is model-
  agnostic: retrain on your own data, re-run steps 1-3, and the same commands
  produce an updated `network.rpk`.
- The previous classifier project at `~/dit-imx500` on the Pi is packaging
  reference only; this project is object detection with different postprocessing.
