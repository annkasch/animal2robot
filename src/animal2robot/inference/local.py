from __future__ import annotations

from animal2robot.inference.base import Detection, InferenceBackend, Keypoint


class LocalBackend(InferenceBackend):
    def __init__(self, checkpoint_path: str) -> None:
        if not checkpoint_path:
            raise ValueError("LOCAL_CHECKPOINT must be set for the local backend")
        self._checkpoint_path = checkpoint_path
        self._model = None

    def _load(self):
        if self._model is None:
            from rfdetr import RFDETRKeypointPreview
            self._model = RFDETRKeypointPreview(pretrain_weights=self._checkpoint_path)

    def predict(self, image_path: str, threshold: float = 0.3) -> list[Detection]:
        self._load()
        raw = self._model.predict(image_path, threshold=threshold)
        return [
            Detection(
                keypoints=[
                    Keypoint(x=kp.x, y=kp.y, confidence=kp.confidence)
                    for kp in (det.keypoints if hasattr(det, "keypoints") else [])
                ],
                confidence=getattr(det, "confidence", 0.0),
            )
            for det in raw
        ]
