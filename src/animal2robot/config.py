from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from animal2robot.datasets import DEFAULT_DATASET


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    INFERENCE_BACKEND: Literal["local", "roboflow"] = "roboflow"
    DATASET: str = DEFAULT_DATASET

    # Local backend
    LOCAL_CHECKPOINT: str = ""

    # Roboflow inference backend
    ROBOFLOW_API_KEY: str = ""
    ROBOFLOW_MODEL_ID: str = ""

    # Roboflow dataset download (used by training/dataset.py)
    ROBOFLOW_WORKSPACE: str = ""
    ROBOFLOW_PROJECT: str = ""

    CONF_THRESHOLD: float = 0.3


def get_settings() -> Settings:
    return Settings()


def get_backend(settings: Settings | None = None):
    from animal2robot.inference.local import LocalBackend
    from animal2robot.inference.roboflow import RoboflowBackend

    s = settings or get_settings()
    if s.INFERENCE_BACKEND == "local":
        return LocalBackend(checkpoint_path=s.LOCAL_CHECKPOINT)
    return RoboflowBackend(api_key=s.ROBOFLOW_API_KEY, model_id=s.ROBOFLOW_MODEL_ID)
