"""Unit tests for gait extraction — no GPU or inference backend required."""
import numpy as np
import pandas as pd
import pytest

from animal2robot.datasets import get_spec, DEFAULT_DATASET
from animal2robot.pipeline.gait import keypoints_to_gait

_spec = get_spec(DEFAULT_DATASET)
KEYPOINT_NAMES = _spec.keypoint_names
PAW_NAMES = _spec.paw_names


def _synthetic_df(fps: float = 30.0, duration: float = 5.0, stride_hz: float = 2.0) -> pd.DataFrame:
    """Build a synthetic keypoints DataFrame with a clean sinusoidal gait signal."""
    n_frames = int(fps * duration)
    frames = np.arange(n_frames)
    t = frames / fps

    records = []
    for frame_id, ts in zip(frames, t):
        for i, name in enumerate(KEYPOINT_NAMES):
            # Paws oscillate at stride_hz; other keypoints stay fixed
            if name in PAW_NAMES:
                paw_idx = PAW_NAMES.index(name)
                phase_offset = paw_idx * np.pi / 2  # 90° phase shift between paws
                y = 0.5 + 0.1 * np.sin(2 * np.pi * stride_hz * ts + phase_offset)
            else:
                y = 0.3
            records.append(
                {
                    "frame": frame_id,
                    "timestamp": round(float(ts), 4),
                    "keypoint": name,
                    "x": 0.5,
                    "y": float(y),
                    "confidence": 1.0,
                }
            )

    return pd.DataFrame(records)


@pytest.fixture
def synthetic_result():
    fps = 30.0
    df = _synthetic_df(fps=fps, stride_hz=2.0)
    meta = {"fps": fps, "width": 640, "height": 480, "total_frames": len(df["frame"].unique())}
    return keypoints_to_gait(df, meta)


def test_keys_present(synthetic_result):
    assert set(synthetic_result.keys()) == {
        "stride_frequency_hz",
        "inter_limb_phase",
        "paw_trajectories",
        "body_center_trajectory",
    }


def test_paw_trajectories_all_present(synthetic_result):
    assert set(synthetic_result["paw_trajectories"].keys()) == set(PAW_NAMES)


def test_stride_frequency_close_to_ground_truth(synthetic_result):
    freq = synthetic_result["stride_frequency_hz"]
    assert 1.5 <= freq <= 2.5, f"Expected ~2.0 Hz, got {freq}"


def test_inter_limb_phase_reference_is_zero(synthetic_result):
    phase = synthetic_result["inter_limb_phase"]
    assert phase["left_front_paw"] == 0.0


def test_trajectory_length_matches_frames(synthetic_result):
    fps = 30.0
    duration = 5.0
    expected = int(fps * duration)
    for paw, traj in synthetic_result["paw_trajectories"].items():
        assert len(traj) == expected, f"{paw}: expected {expected} points, got {len(traj)}"


def test_no_keypoints_raises():
    empty_df = pd.DataFrame(columns=["frame", "timestamp", "keypoint", "x", "y", "confidence"])
    meta = {"fps": 30.0, "width": 640, "height": 480, "total_frames": 10}
    result = keypoints_to_gait(empty_df, meta)
    # Should return 0.0 stride frequency gracefully rather than crash
    assert result["stride_frequency_hz"] == 0.0
