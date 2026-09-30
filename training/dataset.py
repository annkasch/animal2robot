"""
Dataset preparation for RF-DETR Keypoint training.

Usage:
    python training/dataset.py --dataset dog-pose-ultralytics --root datasets/
    python training/dataset.py --dataset ap10k               --root datasets/
    python training/dataset.py --dataset animal-pose         --root datasets/

Auto-download behaviour:
    - If the dataset is already present (train/images/ exists), this is a no-op.
    - For dog-pose-ultralytics: uses Roboflow API if ROBOFLOW_API_KEY is set,
      otherwise falls back to the direct Ultralytics URL.
    - For COCO-format datasets (ap10k, animal-pose): downloads the zip and
      runs the COCO→YOLO converter automatically.
"""
from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from animal2robot.datasets import DatasetSpec, get_spec, DEFAULT_DATASET


def is_present(root: Path, spec: DatasetSpec) -> bool:
    return (root / "train" / "images").exists()


def ensure_present(root: Path, spec: DatasetSpec, roboflow_api_key: str = "") -> None:
    """Download and prepare the dataset if not already present."""
    if is_present(root, spec):
        print(f"Dataset '{spec.name}' already present at {root}")
        return

    print(f"Dataset '{spec.name}' not found. Downloading...")
    root.mkdir(parents=True, exist_ok=True)

    if spec.name == "dog-pose-ultralytics":
        _download_dog_pose(root, spec, roboflow_api_key)
    elif spec.source_format == "coco":
        _download_coco_dataset(root, spec)
    else:
        _download_yolo_zip(root, spec)


# ── Per-dataset download strategies ──────────────────────────────────────────

def _download_dog_pose(root: Path, spec: DatasetSpec, roboflow_api_key: str) -> None:
    api_key = roboflow_api_key or os.environ.get("ROBOFLOW_API_KEY", "")
    workspace = os.environ.get("ROBOFLOW_WORKSPACE", spec.roboflow_workspace)
    project = os.environ.get("ROBOFLOW_PROJECT", spec.roboflow_project)

    if api_key and workspace and project:
        print("Downloading via Roboflow API...")
        _download_from_roboflow(root, api_key, workspace, project, spec.roboflow_version)
    else:
        print("Roboflow credentials not set — falling back to direct URL download.")
        _download_yolo_zip(root, spec)


def _download_from_roboflow(
    root: Path,
    api_key: str,
    workspace: str,
    project: str,
    version: int,
) -> None:
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version).download(
        "yolov8", location=str(root)
    )
    # Roboflow downloads into root/{project}-{version}/ — restructure to rfdetr layout
    downloaded_root = next(
        (p for p in root.iterdir() if p.is_dir() and p.name != "train" and p.name != "val"),
        None,
    )
    if downloaded_root:
        _restructure_roboflow(downloaded_root, root)


def _restructure_roboflow(src: Path, dst: Path) -> None:
    """Roboflow YOLOv8 export already has train/valid/test splits — rename valid→val."""
    for split_src, split_dst in [("train", "train"), ("valid", "val"), ("test", "test")]:
        for kind in ("images", "labels"):
            s = src / split_src / kind
            d = dst / split_dst / kind
            if s.exists() and not d.exists():
                shutil.copytree(s, d)
    shutil.rmtree(src, ignore_errors=True)
    _verify(dst)


def _download_yolo_zip(root: Path, spec: DatasetSpec) -> None:
    zip_path = root / f"{spec.name}.zip"
    if not zip_path.exists():
        print(f"Downloading {spec.url} ...")
        urllib.request.urlretrieve(spec.url, zip_path)

    print("Extracting...")
    extract_dir = root / "_extract"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    _restructure_yolo(extract_dir, root)
    shutil.rmtree(extract_dir, ignore_errors=True)
    _verify(root)


def _restructure_yolo(src: Path, dst: Path) -> None:
    """Move images/labels from flat YOLO layout into rfdetr's split layout."""
    # Handle nested zip root (some zips add one extra directory level)
    subdirs = [p for p in src.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and not (src / "images").exists():
        src = subdirs[0]

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            s = src / "images" / split if kind == "images" else src / "labels" / split
            d = dst / split / kind
            if s.exists() and not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                for f in s.iterdir():
                    shutil.move(str(f), d / f.name)

    # Copy data.yaml if present
    for yaml_name in ("data.yaml", "dog-pose.yaml"):
        y = src / yaml_name
        if y.exists() and not (dst / "data.yaml").exists():
            shutil.copy(str(y), dst / "data.yaml")


def _download_coco_dataset(root: Path, spec: DatasetSpec) -> None:
    from training.converters.coco_to_yolo import convert_splits

    zip_path = root / f"{spec.name}.zip"
    if not zip_path.exists():
        print(f"Downloading {spec.url} ...")
        urllib.request.urlretrieve(spec.url, zip_path)

    print("Extracting...")
    extract_dir = root / "_extract"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    print("Converting COCO → YOLO ...")
    train_json, val_json, images_dir = _find_coco_structure(extract_dir, spec.name)
    convert_splits(train_json, val_json, images_dir, root)
    shutil.rmtree(extract_dir, ignore_errors=True)
    _write_data_yaml(root, spec)
    _verify(root)


def _find_coco_structure(extract_dir: Path, dataset_name: str) -> tuple[Path, Path, Path]:
    """Locate COCO JSON files and images directory inside an extracted zip."""
    if dataset_name == "ap10k":
        # AP-10K structure: ap-10k/annotations/ap10k-train-split1.json etc.
        ann_dir = next(extract_dir.rglob("annotations"), None) or extract_dir
        train_json = next(ann_dir.glob("*train*.json"))
        val_json = next(ann_dir.glob("*val*.json"))
        images_dir = next(extract_dir.rglob("data"), extract_dir)
    elif dataset_name == "animal-pose":
        # animal-pose-dataset: annotations/train.json, val.json
        ann_dir = next(extract_dir.rglob("annotations"), extract_dir)
        train_json = ann_dir / "train.json"
        val_json = ann_dir / "val.json"
        images_dir = next(extract_dir.rglob("images"), extract_dir)
    else:
        raise ValueError(f"Unknown COCO dataset layout for '{dataset_name}'")

    return train_json, val_json, images_dir


def _write_data_yaml(root: Path, spec: DatasetSpec) -> None:
    yaml_path = root / "data.yaml"
    if yaml_path.exists():
        return
    content = (
        f"nc: 1\n"
        f"names: ['animal']\n"
        f"kpt_shape: [{spec.num_keypoints}, 3]\n"
        f"train: train/images\n"
        f"val: val/images\n"
    )
    yaml_path.write_text(content)


def _verify(root: Path) -> None:
    for split in ("train", "val"):
        imgs = list((root / split / "images").glob("*"))
        lbls = list((root / split / "labels").glob("*.txt"))
        print(f"  {split}: {len(imgs)} images, {len(lbls)} labels")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare a dataset for RF-DETR training")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=["dog-pose-ultralytics", "ap10k", "animal-pose"])
    parser.add_argument("--root", default="datasets")
    args = parser.parse_args()

    spec = get_spec(args.dataset)
    ensure_present(Path(args.root) / spec.name, spec)
    print(f"\nReady: {(Path(args.root) / spec.name).resolve()}")
