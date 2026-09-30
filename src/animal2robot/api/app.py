from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from animal2robot.api.schemas import GaitResponse
from animal2robot.config import get_backend, get_settings
from animal2robot.inference.base import InferenceBackend
from animal2robot.pipeline.gait import keypoints_to_gait
from animal2robot.pipeline.video import video_to_keypoints

_ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

_backend: InferenceBackend | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _backend
    settings = get_settings()
    _backend = get_backend(settings)
    yield
    _backend = None


app = FastAPI(
    title="animal2robot",
    description="Dog pose estimation → structured gait representation",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/video", response_model=GaitResponse)
async def predict_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {_ALLOWED_EXTENSIONS}",
        )

    settings = get_settings()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await file.read())

    try:
        df, meta = video_to_keypoints(tmp_path, _backend, conf_threshold=settings.CONF_THRESHOLD)
        if df.empty:
            raise HTTPException(status_code=422, detail="No keypoints detected in the video.")
        gait = keypoints_to_gait(df, meta, conf_threshold=settings.CONF_THRESHOLD)
    finally:
        tmp_path.unlink(missing_ok=True)

    return GaitResponse(**gait, video_meta=meta)
