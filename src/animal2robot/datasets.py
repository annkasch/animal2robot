from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DatasetSpec:
    name: str
    keypoint_names: list[str]
    paw_indices: list[int]
    skeleton_edges: list[tuple[int, int]]
    source_format: Literal["yolo", "coco"]
    # Direct download URL (zip)
    url: str = ""
    # Roboflow Universe download (preferred when API key is available)
    roboflow_workspace: str = ""
    roboflow_project: str = ""
    roboflow_version: int = 0
    # Maps species name → COCO category ID.
    # Empty for YOLO datasets (already single-species or pre-filtered).
    # Verify IDs against the actual annotation JSON before use.
    species_map: dict[str, int] = field(default_factory=dict)
    # Species included by default when no explicit filter is given.
    default_species: list[str] = field(default_factory=lambda: ["dog"])

    @property
    def paw_names(self) -> list[str]:
        return [self.keypoint_names[i] for i in self.paw_indices]

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoint_names)

    def category_ids(self, species: list[str]) -> list[int] | None:
        """
        Return COCO category IDs for the requested species list.
        Returns None for YOLO datasets (no filtering needed).
        Raises ValueError for unknown species names.
        """
        if not self.species_map:
            return None
        ids: list[int] = []
        for s in species:
            if s not in self.species_map:
                raise ValueError(
                    f"Species '{s}' not available in dataset '{self.name}'. "
                    f"Available: {list(self.species_map)}"
                )
            ids.append(self.species_map[s])
        return ids


# ── dog-pose-ultralytics ──────────────────────────────────────────────────────
# 8,476 images, 24 keypoints, dog-only. No species filtering needed.
# Roboflow Universe version is the default download; direct URL is the fallback.

_DOG_POSE_ULTRALYTICS = DatasetSpec(
    name="dog-pose-ultralytics",
    source_format="yolo",
    url="https://ultralytics.com/assets/dog-pose.zip",
    roboflow_workspace="",   # set via ROBOFLOW_WORKSPACE env var or .env
    roboflow_project="",     # set via ROBOFLOW_PROJECT env var or .env
    roboflow_version=1,
    species_map={},          # single-species dataset, no filtering
    default_species=["dog"],
    keypoint_names=[
        "nose",                  # 0
        "left_eye",              # 1
        "right_eye",             # 2
        "left_ear",              # 3
        "right_ear",             # 4
        "throat",                # 5
        "withers",               # 6
        "tail_base",             # 7
        "left_front_paw",        # 8
        "left_front_wrist",      # 9
        "left_front_elbow",      # 10
        "left_front_shoulder",   # 11
        "right_front_paw",       # 12
        "right_front_wrist",     # 13
        "right_front_elbow",     # 14
        "right_front_shoulder",  # 15
        "left_back_paw",         # 16
        "left_back_hock",        # 17
        "left_back_knee",        # 18
        "left_back_hip",         # 19
        "right_back_paw",        # 20
        "right_back_hock",       # 21
        "right_back_knee",       # 22
        "right_back_hip",        # 23
    ],
    paw_indices=[8, 12, 16, 20],
    skeleton_edges=[
        (0, 5), (5, 6), (6, 7),
        (6, 11), (11, 10), (10, 9), (9, 8),
        (6, 15), (15, 14), (14, 13), (13, 12),
        (7, 19), (19, 18), (18, 17), (17, 16),
        (7, 23), (23, 22), (22, 21), (21, 20),
    ],
)

# ── AP-10K ───────────────────────────────────────────────────────────────────
# 10,015 images, 17 keypoints, 54 animal species. COCO JSON format.
# Source: https://github.com/AlexTheBad/AP-10K
#
# Category IDs — verify against annotations/ap10k-train-split1.json before use:
#   run: python -c "import json; d=json.load(open('annotations/ap10k-train-split1.json'));
#                   print({c['name']: c['id'] for c in d['categories']})"

_AP10K = DatasetSpec(
    name="ap10k",
    source_format="coco",
    url="https://github.com/AlexTheBad/AP-10K/releases/download/v1.0/ap-10k.zip",
    species_map={
        # Domestic
        "dog":    1,   # verify
        "cat":    2,   # verify
        "horse":  3,   # verify
        "cow":    4,   # verify
        "sheep":  5,   # verify
        # Wildlife (subset — AP-10K has 54 total; add more as needed)
        "wolf":   6,   # verify
        "fox":    7,   # verify
        "lion":   8,   # verify
        "tiger":  9,   # verify
        "cheetah": 10, # verify
        "leopard": 11, # verify
        "bear":   12,  # verify
        "zebra":  13,  # verify
        "elephant": 14, # verify
        "giraffe": 15, # verify
        "deer":   16,  # verify
    },
    default_species=["dog"],
    keypoint_names=[
        "nose",                    # 0
        "left_eye",                # 1
        "right_eye",               # 2
        "left_ear",                # 3
        "right_ear",               # 4
        "left_front_shoulder",     # 5
        "right_front_shoulder",    # 6
        "left_front_elbow",        # 7
        "right_front_elbow",       # 8
        "left_front_paw",          # 9
        "right_front_paw",         # 10
        "left_back_hip",           # 11
        "right_back_hip",          # 12
        "left_back_knee",          # 13
        "right_back_knee",         # 14
        "left_back_paw",           # 15
        "right_back_paw",          # 16
    ],
    paw_indices=[9, 10, 15, 16],
    skeleton_edges=[
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (5, 11), (6, 12),
    ],
)

# ── Animal Pose Dataset (COCO-Animals) ───────────────────────────────────────
# ~5,000 images, 20 keypoints, 5 species: dog, cat, horse, sheep, cow.
# COCO JSON format.
# Source: https://github.com/noahcao/animal-pose-dataset
#
# Category IDs — verify against the annotation JSON before use:
#   run: python -c "import json; d=json.load(open('train.json'));
#                   print({c['name']: c['id'] for c in d['categories']})"

_ANIMAL_POSE = DatasetSpec(
    name="animal-pose",
    source_format="coco",
    url="https://github.com/noahcao/animal-pose-dataset/archive/refs/heads/master.zip",
    species_map={
        "dog":   1,  # verify
        "cat":   2,  # verify
        "horse": 3,  # verify
        "sheep": 4,  # verify
        "cow":   5,  # verify
    },
    default_species=["dog"],
    keypoint_names=[
        "left_eye",              # 0
        "right_eye",             # 1
        "left_ear",              # 2
        "right_ear",             # 3
        "nose",                  # 4
        "throat",                # 5
        "withers",               # 6
        "tail_base",             # 7
        "left_front_paw",        # 8
        "left_front_wrist",      # 9
        "left_front_elbow",      # 10
        "left_front_shoulder",   # 11
        "right_front_paw",       # 12
        "right_front_wrist",     # 13
        "right_front_elbow",     # 14
        "right_front_shoulder",  # 15
        "left_back_paw",         # 16
        "left_back_knee",        # 17
        "left_back_hip",         # 18
        "right_back_paw",        # 19
    ],
    paw_indices=[8, 12, 16, 19],
    skeleton_edges=[
        (4, 5), (5, 6), (6, 7),
        (6, 11), (11, 10), (10, 9), (9, 8),
        (6, 15), (15, 14), (14, 13), (13, 12),
        (7, 18), (18, 17), (17, 16),
        (7, 18), (18, 17), (17, 19),
    ],
)

# ── Registry ─────────────────────────────────────────────────────────────────

REGISTRY: dict[str, DatasetSpec] = {
    s.name: s for s in [_DOG_POSE_ULTRALYTICS, _AP10K, _ANIMAL_POSE]
}

DEFAULT_DATASET = "dog-pose-ultralytics"
DEFAULT_SPECIES = ["dog"]


def get_spec(name: str) -> DatasetSpec:
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]
