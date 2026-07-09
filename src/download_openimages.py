import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

classes = ["Plastic bag", "Tin can", "Jar"]
ds = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=classes,
    max_samples=3000,
    only_matching=True,
    shuffle=True,
    seed=42,
    dataset_name="oi_waste",
)
print("COUNTS:", ds.count_values("ground_truth.detections.label"))
view = ds.filter_labels("ground_truth", F("label").is_in(classes))
view.export(
    export_dir="data/openimages_yolo",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    classes=classes,
    split="train",
)
print("EXPORT_DONE")
