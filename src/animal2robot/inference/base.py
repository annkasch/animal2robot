from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class Detection:
    keypoints: list[Keypoint] = field(default_factory=list)
    confidence: float = 0.0


class InferenceBackend(ABC):
    @abstractmethod
    def predict(self, image_path: str, threshold: float = 0.3) -> list[Detection]:
        """Return detections for a single image."""
