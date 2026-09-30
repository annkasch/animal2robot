"""
Convert a COCO keypoint dataset to YOLO keypoint format expected by RF-DETR.

COCO annotation structure (per annotation):
  {
    "id": int,
    "image_id": int,
    "category_id": int,
    "bbox": [x, y, w, h],          # top-left x/y, width, height (absolute pixels)
    "keypoints": [x1, y1, v1, ...], # x, y, visibility per keypoint (absolute pixels)
    "num_keypoints": int
  }

YOLO keypoint format (per line in .txt):
  class cx cy w h x1 y1 v1 x2 y2 v2 ...
  All values normalised to [0, 1] by image width/height.
  Visibility: 0 = not labelled, 1 = occluded, 2 = visible (COCO convention kept).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def convert(
    coco_json: Path,
    images_dir: Path,
    output_dir: Path,
    split: str = "train",
    category_id: int | None = None,
) -> None:
    """
    Convert one COCO JSON split to YOLO keypoint .txt files.

    Parameters
    ----------
    coco_json : Path
        Path to the COCO annotations JSON file.
    images_dir : Path
        Directory containing the raw images referenced by the JSON.
    output_dir : Path
        Destination root. Will create:
            output_dir/{split}/images/
            output_dir/{split}/labels/
    split : str
        "train" or "val"
    category_id : int | None
        If set, only convert annotations for this category. Otherwise converts all.
    """
    with open(coco_json) as f:
        data = json.load(f)

    # Build lookup tables
    images_meta: dict[int, dict] = {img["id"]: img for img in data["images"]}
    # Group annotations by image
    ann_by_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        if ann.get("num_keypoints", 0) == 0:
            continue
        if category_id is not None and ann["category_id"] != category_id:
            continue
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    dst_images = output_dir / split / "images"
    dst_labels = output_dir / split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    for image_id, anns in ann_by_image.items():
        img_meta = images_meta.get(image_id)
        if img_meta is None:
            skipped += 1
            continue

        img_w = img_meta["width"]
        img_h = img_meta["height"]
        file_name = img_meta["file_name"]

        src_img = images_dir / file_name
        if not src_img.exists():
            # Try basename only (some COCO zips nest under subdirs)
            src_img = images_dir / Path(file_name).name
        if not src_img.exists():
            skipped += 1
            continue

        lines: list[str] = []
        for ann in anns:
            bx, by, bw, bh = ann["bbox"]
            cx = (bx + bw / 2) / img_w
            cy = (by + bh / 2) / img_h
            nw = bw / img_w
            nh = bh / img_h

            kps = ann["keypoints"]  # flat list [x, y, v, x, y, v, ...]
            norm_kps: list[str] = []
            for i in range(0, len(kps), 3):
                kx = kps[i] / img_w
                ky = kps[i + 1] / img_h
                vis = kps[i + 2]
                norm_kps.extend([f"{kx:.6f}", f"{ky:.6f}", str(vis)])

            # class index 0 (single-class: animal)
            line = " ".join(["0", f"{cx:.6f}", f"{cy:.6f}", f"{nw:.6f}", f"{nh:.6f}"] + norm_kps)
            lines.append(line)

        if not lines:
            skipped += 1
            continue

        stem = Path(file_name).stem
        shutil.copy2(src_img, dst_images / src_img.name)
        (dst_labels / f"{stem}.txt").write_text("\n".join(lines))
        converted += 1

    print(f"  [{split}] converted {converted} images, skipped {skipped}")


def convert_splits(
    train_json: Path,
    val_json: Path,
    images_dir: Path,
    output_dir: Path,
    category_id: int | None = None,
) -> None:
    for split, json_path in [("train", train_json), ("val", val_json)]:
        print(f"Converting {split} split from {json_path.name} ...")
        convert(json_path, images_dir, output_dir, split=split, category_id=category_id)
