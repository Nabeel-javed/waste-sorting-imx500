# Evaluation Results

Fill this file after training the final model.

## Validation Command

```bash
python src/validate.py --model models/best.pt --data dataset/data.yaml --imgsz 640
```

## Overall Metrics

| Metric | Value |
| --- | ---: |
| Precision | TBD |
| Recall | TBD |
| mAP50 | TBD |
| mAP50-95 | TBD |

## Per-Class Metrics

| Class | Precision | Recall | mAP50 | mAP50-95 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| plastic_bottle | TBD | TBD | TBD | TBD | Transparent bottles may confuse with glass. |
| can | TBD | TBD | TBD | TBD | Reflections may affect detection. |
| paper | TBD | TBD | TBD | TBD | May confuse with cardboard. |
| cardboard | TBD | TBD | TBD | TBD | May confuse with paper. |
| glass_jar | TBD | TBD | TBD | TBD | Transparency and reflections are challenging. |
| food_wrapper | TBD | TBD | TBD | TBD | High visual variation. |

## Confusion Matrix

Add exported confusion matrix image path here:

```text
runs/detect/<run_name>/confusion_matrix.png
```

## Example Detections

Add screenshots from `examples/sample_images/out/` or `examples/screenshots/`.

