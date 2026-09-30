"""Download and restructure the Ultralytics dog-pose dataset into rfdetr layout."""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = "https://ultralytics.com/assets/dog-pose.zip"


def download(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "dog-pose.zip"
    if not zip_path.exists():
        print(f"Downloading dog-pose dataset → {zip_path}")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
    else:
        print("Zip already present, skipping download.")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(root)
    print("Done.")


def restructure(root: Path) -> None:
    """
    Move images/labels from:
        root/images/{train,val}/
        root/labels/{train,val}/
    to rfdetr's expected layout:
        root/{train,val}/images/
        root/{train,val}/labels/
    and rename dog-pose.yaml → data.yaml.
    """
    for split in ("train", "val"):
        src_images = root / "images" / split
        src_labels = root / "labels" / split
        dst_images = root / split / "images"
        dst_labels = root / split / "labels"

        if src_images.exists() and not dst_images.exists():
            print(f"Moving {src_images} → {dst_images}")
            dst_images.mkdir(parents=True, exist_ok=True)
            for f in src_images.iterdir():
                shutil.move(str(f), dst_images / f.name)

        if src_labels.exists() and not dst_labels.exists():
            print(f"Moving {src_labels} → {dst_labels}")
            dst_labels.mkdir(parents=True, exist_ok=True)
            for f in src_labels.iterdir():
                shutil.move(str(f), dst_labels / f.name)

    yaml_src = root / "dog-pose.yaml"
    yaml_dst = root / "data.yaml"
    if yaml_src.exists() and not yaml_dst.exists():
        shutil.copy(str(yaml_src), yaml_dst)
        print(f"Renamed {yaml_src.name} → data.yaml")

    for split in ("train", "val"):
        imgs = list((root / split / "images").glob("*.jpg"))
        lbls = list((root / split / "labels").glob("*.txt"))
        print(f"{split}: {len(imgs)} images, {len(lbls)} labels")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare dog-pose dataset for RF-DETR training")
    parser.add_argument("--root", default="datasets", help="Directory to download and extract into")
    args = parser.parse_args()

    root = Path(args.root)
    download(root)
    restructure(root)
    print(f"\nDataset ready at: {root.resolve()}")
