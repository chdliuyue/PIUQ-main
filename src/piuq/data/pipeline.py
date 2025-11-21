from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .configs import DatasetProcessConfig


def smooth_positions(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Apply centered rolling averages to stabilize kinematic signals.
    应用中心滚动平均来稳定运动学信号。

    """
    if window <= 1:
        return df
    smoothed = df.copy()
    for col in ["x", "y", "vx", "vy", "ax", "ay"]:
        if col in smoothed.columns:
            smoothed[col] = (
                smoothed.groupby("track_id")[col]
                .transform(lambda s: s.rolling(window, min_periods=1, center=True).mean())
            )
    return smoothed


def downsample_tracks(df: pd.DataFrame, target_hz: float) -> Tuple[pd.DataFrame, float]:
    """Downsample trajectories to the target frequency while preserving alignment.
    在保持轨迹对齐的情况下将采样率降至目标频率。

    """

    if "t" not in df.columns or target_hz <= 0:
        return df, target_hz

    tracks = []
    for _, track_df in df.groupby("track_id"):
        track_df = track_df.sort_values("t")
        if len(track_df) < 2:
            tracks.append(track_df)
            continue
        dt = float(np.median(np.diff(track_df["t"].to_numpy())))
        if dt <= 0:
            tracks.append(track_df)
            continue
        desired_dt = 1.0 / target_hz
        step = max(1, int(round(desired_dt / dt)))
        tracks.append(track_df.iloc[::step])
    downsampled = pd.concat(tracks, ignore_index=True)
    downsampled = downsampled.sort_values(["recording_id", "track_id", "t"])
    return downsampled, target_hz


def harmonize_features(df: pd.DataFrame, cfg: DatasetProcessConfig) -> pd.DataFrame:
    """Reorder and back-fill columns to a consistent schema.
    重新排序并补全列以保持一致的数据模式。

    """

    df = df.copy()
    df = cfg.fill_missing_columns(df)
    return df


__all__ = ["smooth_positions", "downsample_tracks", "harmonize_features"]
