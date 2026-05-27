# Technical Pipeline

```text
Image capture
  -> Annotation in YOLO format
  -> Train/val/test split
  -> YOLO training
  -> Validation and metrics
  -> Desktop image/video/webcam demo
  -> IMX500 export
  -> Raspberry Pi packaging
  -> IMX500 demo
```

## Runtime Flow

1. Camera frame is read by OpenCV.
2. YOLO predicts bounding boxes, classes, and confidence scores.
3. The bin logic maps the detected class to a recycling bin.
4. The UI draws boxes, labels, confidence, and bin recommendation.
5. If zone mode is enabled, the box center determines the virtual table zone.
6. The zone logic compares the object's recommended bin with the current zone and displays correct/wrong feedback.

## Virtual Zones

The desktop demo divides the frame into three vertical zones:

| Zone | Accepted bins |
| --- | --- |
| Left | Plastic / Packaging |
| Center | Paper, Paper / Cardboard |
| Right | Glass, Metal / Recycling, General Waste / Packaging |

This is a simple demo aid. It does not detect physical bins.

