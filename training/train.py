"""
Training entrypoint for RF-DETR Keypoint.
Launched via torchrun for single- or multi-GPU training:

    torchrun --nproc_per_node=NUM_GPUS training/train.py --config training/config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    from animal2robot.datasets import get_spec
    from training.dataset import ensure_present

    spec = get_spec(cfg["dataset"])
    dataset_dir = Path(cfg["dataset_root"]) / spec.name
    species = cfg.get("species") or spec.default_species
    ensure_present(dataset_dir, spec, species=species)

    from rfdetr import RFDETRKeypointPreview

    model = RFDETRKeypointPreview()
    model.train(
        dataset_dir=str(dataset_dir),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        resolution=cfg["resolution"],
        lr=cfg["lr"],
        grad_accum_steps=cfg["grad_accum_steps"],
        output_dir=cfg["output_dir"],
    )


if __name__ == "__main__":
    main()
