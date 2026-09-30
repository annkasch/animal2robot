from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.signal import correlate

from animal2robot.datasets import DatasetSpec, get_spec, DEFAULT_DATASET


def keypoints_to_gait(
    df: pd.DataFrame,
    video_meta: dict,
    conf_threshold: float = 0.3,
    spec: DatasetSpec | None = None,
) -> dict:
    """
    Convert a per-frame keypoints DataFrame to a structured gait representation.

    Parameters
    ----------
    df : DataFrame
        Output of video_to_keypoints — columns: frame, timestamp, keypoint, x, y, confidence
    video_meta : dict
        fps, width, height, total_frames
    conf_threshold : float
        Keypoints below this confidence are treated as missing.

    Returns
    -------
    dict with keys:
        paw_trajectories, stride_frequency_hz, inter_limb_phase, body_center_trajectory
    """
    if spec is None:
        spec = get_spec(DEFAULT_DATASET)

    paw_names = spec.paw_names

    fps = video_meta["fps"]
    total_frames = video_meta["total_frames"]
    frame_index = np.arange(total_frames)
    timestamps = frame_index / fps

    df = df.copy()
    df.loc[df["confidence"] < conf_threshold, ["x", "y"]] = np.nan

    def _time_series(keypoint_name: str, coord: str) -> np.ndarray:
        kp_df = df[df["keypoint"] == keypoint_name].set_index("frame")[coord]
        series = kp_df.reindex(frame_index).values.astype(float)
        # Linear interpolation across missing frames
        nans = np.isnan(series)
        if nans.all():
            return series
        x_idx = np.flatnonzero(~nans)
        series[nans] = np.interp(np.flatnonzero(nans), x_idx, series[x_idx])
        return series

    # Normalize y relative to body center (withers–tail_base midpoint)
    withers_y = _time_series("withers", "y")
    tail_y = _time_series("tail_base", "y")
    body_center_y = (withers_y + tail_y) / 2.0

    withers_x = _time_series("withers", "x")
    tail_x = _time_series("tail_base", "x")
    body_center_x = (withers_x + tail_x) / 2.0

    body_length = np.nanmean(np.abs(withers_x - tail_x)) or 1.0

    paw_trajectories: dict[str, list] = {}
    paw_y_signals: dict[str, np.ndarray] = {}

    for paw in paw_names:
        raw_y = _time_series(paw, "y")
        raw_x = _time_series(paw, "x")

        # Normalize: subtract body center, scale by body length
        norm_y = (raw_y - body_center_y) / body_length
        norm_x = (raw_x - body_center_x) / body_length

        # Smooth with Savitzky-Golay (window ~0.2 s, must be odd)
        win = max(5, int(fps * 0.2) | 1)
        if np.isnan(norm_y).all():
            smooth_y = norm_y
        else:
            smooth_y = savgol_filter(norm_y, window_length=win, polyorder=3)

        paw_y_signals[paw] = smooth_y
        paw_trajectories[paw] = [
            {"t": round(float(t), 4), "x": round(float(x), 4), "y": round(float(y), 4)}
            for t, x, y in zip(timestamps, norm_x, smooth_y)
        ]

    reference_paw = paw_names[0]
    stride_frequency_hz = _stride_frequency(paw_y_signals[reference_paw], fps)
    inter_limb_phase = _inter_limb_phase(paw_y_signals, stride_frequency_hz, fps, reference_paw)

    return {
        "stride_frequency_hz": round(stride_frequency_hz, 3),
        "inter_limb_phase": inter_limb_phase,
        "paw_trajectories": paw_trajectories,
        "body_center_trajectory": [
            {"t": round(float(t), 4), "x": round(float(x), 4), "y": round(float(y), 4)}
            for t, x, y in zip(timestamps, body_center_x, body_center_y)
        ],
    }


def _stride_frequency(signal: np.ndarray, fps: float) -> float:
    n = len(signal)
    if n < 4 or np.isnan(signal).all():
        return 0.0
    signal = np.nan_to_num(signal - np.nanmean(signal))

    # FFT
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power = np.abs(np.fft.rfft(signal)) ** 2
    # Only consider plausible stride frequencies (0.5–5 Hz)
    mask = (freqs >= 0.5) & (freqs <= 5.0)
    if not mask.any():
        return 0.0
    fft_freq = freqs[mask][np.argmax(power[mask])]

    # Autocorrelation sanity check
    ac = correlate(signal, signal, mode="full")
    ac = ac[n - 1 :]
    ac /= ac[0] + 1e-9
    min_lag = max(1, int(fps / 5.0))
    max_lag = int(fps / 0.5)
    search = ac[min_lag : max_lag + 1]
    if search.size == 0:
        return fft_freq
    ac_lag = np.argmax(search) + min_lag
    ac_freq = fps / ac_lag

    # Average FFT and autocorrelation estimates
    return float((fft_freq + ac_freq) / 2.0)


def _inter_limb_phase(
    paw_signals: dict[str, np.ndarray],
    stride_freq: float,
    fps: float,
    reference_paw: str = "left_front_paw",
) -> dict[str, float]:
    reference = paw_signals[reference_paw]
    stride_samples = (fps / stride_freq) if stride_freq > 0 else len(reference)
    result: dict[str, float] = {}

    for paw, signal in paw_signals.items():
        if paw == reference_paw:
            result[paw] = 0.0
            continue
        corr = correlate(reference - reference.mean(), signal - signal.mean(), mode="full")
        n = len(reference)
        lags = np.arange(-(n - 1), n)
        lag = lags[np.argmax(corr)]
        phase = (lag % stride_samples) / stride_samples
        result[paw] = round(float(phase), 3)

    return result
