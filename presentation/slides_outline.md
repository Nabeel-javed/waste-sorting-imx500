# Presentation Outline

## Slide 1: Title

- Real-Time Waste Sorting Assistant Using YOLO and IMX500
- Name, course, professor, date

## Slide 2: Motivation

- Waste sorting is important for recycling.
- Incorrect sorting reduces recycling efficiency.
- Edge AI enables low-latency local inference without cloud dependency.

## Slide 3: Project Concept

- Camera observes waste item.
- YOLO detects object class.
- Rule layer recommends correct bin.
- Optional zone logic checks if item is placed in correct bin area.

## Slide 4: Requirements Mapping

- Object detection network: YOLOv8n/YOLO11n
- Embedded deployment: IMX500 export/deployment files
- Own dataset: 100+ labeled images collected manually
- Standard annotation: YOLO format
- Desktop testability: webcam/OpenCV demo

## Slide 5: Dataset

- 6 classes: plastic_bottle, can, paper, cardboard, glass_jar, food_wrapper
- Number of images
- Train/val/test split
- Annotation tool used
- Example images

## Slide 6: Training Setup

- Model: YOLOv8n or YOLO11n
- Image size: 640
- Epochs: 80
- Batch size: 16
- Transfer learning from pretrained COCO weights
- Augmentations if used

## Slide 7: Technical Pipeline

Dataset collection -> Annotation -> YOLO training -> Evaluation -> Export -> Desktop demo -> IMX500 deployment

## Slide 8: Detection and Bin Logic

- YOLO outputs bounding box, class, confidence.
- Bin mapping converts object class to recommended bin.
- Zone logic checks correct/wrong placement.

## Slide 9: Evaluation Results

- mAP50
- precision
- recall
- confusion matrix
- example detections

## Slide 10: Demo

- Live webcam demo
- Single object detection
- Multiple object detection
- Optional wrong-bin detection

## Slide 11: Constraints and Limitations

- Lighting affects detection.
- Transparent objects are harder.
- Similar classes such as paper/cardboard can confuse model.
- Small objects may be harder on edge device.
- Dataset size is limited.
- IMX500 has model size and operation constraints.

## Slide 12: Improvements and Outlook

- Add more classes.
- Improve dataset diversity.
- Add real bin detection.
- Add audio feedback.
- Add multilingual labels.
- Improve embedded optimization.
- Add automatic sorting actuator in future.

