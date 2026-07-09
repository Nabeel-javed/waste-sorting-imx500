"""Merge an Open Images YOLOv5 export into dataset5 train split."""
import shutil
import sys
from pathlib import Path

src_root = Path(sys.argv[1])   # e.g. data/openimages_yolo
prefix = sys.argv[2]           # e.g. oi
REMAP = {0: 4, 1: 1, 2: 3}     # Plastic bag->food_wrapper, Tin can->can, Jar->glass_jar

img_dst = Path("dataset5/images/train")
lbl_dst = Path("dataset5/labels/train")
added = boxes = 0
for lbl in sorted((src_root / "labels" / "train").glob("*.txt")):
    img = None
    for ext in (".jpg", ".jpeg", ".png"):
        cand = src_root / "images" / "train" / (lbl.stem + ext)
        if cand.exists():
            img = cand
            break
    if img is None:
        continue
    out_lines = []
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        cid = int(p[0])
        if cid not in REMAP:
            continue
        # clamp normalized coords into [0,1] (OI has a few boundary cases)
        vals = [min(max(float(v), 0.0), 1.0) for v in p[1:]]
        out_lines.append(" ".join([str(REMAP[cid])] + [f"{v:.6f}" for v in vals]))
    if not out_lines:
        continue
    (lbl_dst / f"{prefix}_{lbl.stem}.txt").write_text("\n".join(out_lines) + "\n")
    shutil.copy2(img, img_dst / f"{prefix}_{lbl.stem}{img.suffix.lower()}")
    added += 1
    boxes += len(out_lines)
print(f"added {added} images, {boxes} boxes from {src_root}")
