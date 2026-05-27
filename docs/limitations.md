# Limitations

- Dataset size is small compared with production object detectors.
- Transparent plastic bottles and glass jars may be confused.
- Paper and cardboard may be confused when both are flat, brown, or white.
- Food wrappers vary heavily in texture, shape, and color.
- Lighting and reflections can reduce confidence.
- Small or far-away objects may be harder on embedded hardware.
- Crowded scenes can introduce occlusion.
- The zone demo assumes fixed left/center/right table regions and does not detect real bins.
- IMX500 deployment may require model-size, quantization, and postprocessing adjustments.

## Mitigations

- Collect more images under realistic lighting.
- Include multiple object instances and backgrounds.
- Add difficult examples intentionally: reflections, occlusions, and partial objects.
- Keep the camera position fixed for the final demo.
- Use controlled lighting for the physical presentation.
- Re-evaluate after quantized IMX500 export, not only on the desktop model.

