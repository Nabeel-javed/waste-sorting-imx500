# IMX500 Deployment Notes

Primary deployment path: Ultralytics native IMX500 export.

```bash
python src/export_imx500.py --model models/best.pt --data dataset/data.yaml --imgsz 640
```

Equivalent direct Ultralytics command:

```bash
yolo export model=models/best.pt format=imx data=dataset/data.yaml imgsz=640
```

Expected exported artifacts:

```text
models/exported/imx/
  packerOut.zip
  labels.txt
  model_quantized.onnx
```

## Raspberry Pi Packaging

Copy the exported IMX files to the Raspberry Pi:

```bash
scp models/exported/imx/packerOut.zip models/exported/imx/labels.txt pi@<pi-ip>:~/waste-sorting/
```

Package the model on the Pi:

```bash
ssh pi@<pi-ip> 'mkdir -p ~/waste-sorting/rpk_out && imx500-package -i ~/waste-sorting/packerOut.zip -o ~/waste-sorting/rpk_out'
```

The output should include a `network.rpk` file.

## Running on IMX500

Use the Raspberry Pi IMX500 object-detection demo installed by the `imx500-all` package. The exact command can differ by Raspberry Pi OS and package version, so confirm the installed example path on the Pi.

Typical steps:

```bash
ssh pi@<pi-ip>
find /usr/share -iname '*object*detection*.py' 2>/dev/null | head
python3 <imx500_object_detection_demo.py> --model ~/waste-sorting/rpk_out/network.rpk
```

## Local Reference From Prior Project

This machine has a previous IMX500 classification project at:

```text
~/dit-imx500/README.md
```

That project confirmed the Pi-side packaging pattern:

```bash
imx500-package -i packerOut.zip -o rpk_out
```

This waste project is object detection, so the model export and postprocessing path differs from the earlier classifier. Use the earlier project as packaging reference only, not as detector runtime code.

## Demo Recommendations

- Keep camera position fixed.
- Use controlled lighting.
- Test the desktop YOLO model before exporting.
- Test the exported/quantized IMX500 model before the final presentation.
- Keep the final demo objects visually distinct and large enough in frame.

