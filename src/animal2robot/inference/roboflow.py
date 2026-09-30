from __future__ import annotations

from animal2robot.inference.base import Detection, InferenceBackend, Keypoint


class RoboflowBackend(InferenceBackend):
    def __init__(self, api_key: str, model_id: str) -> None:
        if not api_key or not model_id:
            raise ValueError("ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID must be set")
        self._api_key = api_key
        self._model_id = model_id
        self._client = None

    def _load(self):
        if self._client is None:
            from inference_sdk import InferenceHTTPClient
            self._client = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=self._api_key,
            )

    def predict(self, image_path: str, threshold: float = 0.3) -> list[Detection]:
        self._load()
        response = self._client.infer(image_path, model_id=self._model_id)
        detections = []
        for pred in response.get("predictions", []):
            if pred.get("confidence", 0) < threshold:
                continue
            keypoints = [
                Keypoint(
                    x=kp["x"],
                    y=kp["y"],
                    confidence=kp.get("confidence", 1.0),
                )
                for kp in pred.get("keypoints", [])
            ]
            detections.append(Detection(keypoints=keypoints, confidence=pred["confidence"]))
        return detections
