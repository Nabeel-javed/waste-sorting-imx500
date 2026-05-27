# Annotation Guide

## Format

Use YOLO bounding-box annotation format:

```text
class_id x_center y_center width height
```

All coordinates must be normalized between `0` and `1`.

Class order must match `dataset/data.yaml`:

```text
0 plastic_bottle
1 can
2 paper
3 cardboard
4 glass_jar
5 food_wrapper
```

## Preferred Tool: Roboflow

1. Create a Roboflow object-detection project.
2. Add the 6 classes exactly as listed above.
3. Upload images from `dataset/images/raw/`.
4. Draw tight boxes around visible waste objects.
5. Export as YOLOv8 format with train/val/test split.
6. Unzip the export into this repository's `dataset/` folder.
7. Keep this repository's `dataset/data.yaml` if Roboflow generates different relative paths.

## Alternatives

- CVAT: good for careful manual annotation and review.
- LabelImg: simple local annotation workflow.
- makesense.ai: browser-based annotation without an account.

## Quality Checklist

- Box tightly covers the visible object.
- Do not include large table/background regions.
- Label every visible target object in multi-object scenes.
- Keep class names consistent.
- Check transparent objects carefully; do not confuse `glass_jar` with `plastic_bottle`.
- Check paper/cardboard decisions consistently.
- Avoid duplicate or corrupted images.

