from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pandas as pd

from animal2robot.inference.base import InferenceBackend
from animal2robot.keypoints import KEYPOINT_NAMES


def video_to_keypoints(
    video_path: str | Path,
    backend: InferenceBackend,
    conf_threshold: float = 0.3,
) -> tuple[pd.DataFrame, dict]:
    """
    Run per-frame pose inference on a video.

    Returns
    -------
    df : DataFrame
        Columns: frame, timestamp, keypoint, x, y, confidence
    meta : dict
        fps, width, height, total_frames
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    records: list[dict] = []
    frame_id = 0

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imwrite(tmp_path, frame)
            detections = backend.predict(tmp_path, threshold=conf_threshold)

            if detections:
                det = max(detections, key=lambda d: d.confidence)
                for i, kp in enumerate(det.keypoints):
                    records.append(
                        {
                            "frame": frame_id,
                            "timestamp": round(frame_id / fps, 4),
                            "keypoint": KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else f"kp_{i}",
                            "x": kp.x,
                            "y": kp.y,
                            "confidence": round(kp.confidence, 4),
                        }
                    )

            frame_id += 1
    finally:
        cap.release()
        Path(tmp_path).unlink(missing_ok=True)

    df = pd.DataFrame(records)
    meta = {"fps": fps, "width": width, "height": height, "total_frames": total_frames}
    return df, meta
