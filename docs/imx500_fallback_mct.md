# Manual IMX500 Fallback: ONNX / MCT / Converter

The primary path for this project is Ultralytics `format=imx`. Use this fallback only if the professor-provided pipeline or local tooling requires manual conversion.

## Export ONNX

```bash
yolo export model=models/best.pt format=onnx imgsz=640
```

## Other Possible Exports

```bash
yolo export model=models/best.pt format=saved_model imgsz=640
yolo export model=models/best.pt format=tflite imgsz=640 int8=True
```

## Manual Conversion Concept

The older IMX500 classifier project used this general shape:

```text
trained model
  -> quantization with Sony Model Compression Toolkit
  -> imxconv-tf conversion
  -> packerOut.zip
  -> imx500-package on Raspberry Pi
  -> network.rpk
```

For object detection, postprocessing is different from classification. Do not assume classifier code can be reused directly.

## Known Tooling Notes

- Sony converter tooling may require x86 Linux.
- `imxconv-tf` may require Java 17 and `JAVA_HOME`.
- Quantization needs representative calibration images.
- Re-test accuracy after quantization.

The previous classifier project at `~/dit-imx500/README.md` documents a working classifier conversion and Pi packaging path, including the Java 17 requirement. Treat it as a reference for environment setup, not as an exact detector recipe.

