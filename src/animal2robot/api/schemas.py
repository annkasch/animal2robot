from __future__ import annotations

from pydantic import BaseModel


class PawPoint(BaseModel):
    t: float
    x: float
    y: float


class BodyCenterPoint(BaseModel):
    t: float
    x: float
    y: float


class GaitResponse(BaseModel):
    stride_frequency_hz: float
    inter_limb_phase: dict[str, float]
    paw_trajectories: dict[str, list[PawPoint]]
    body_center_trajectory: list[BodyCenterPoint]
    video_meta: dict
