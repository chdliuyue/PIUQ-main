from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_scene(
    df: pd.DataFrame,
    recording_id: Optional[int] = None,
    tracks: Optional[Iterable[int]] = None,
    show: bool = False,
    ax: Optional[plt.Axes] = None,
):
    """Plot trajectories in map coordinates.
    在地图坐标系中绘制轨迹。

    """

    if recording_id is not None:
        df = df[df["recording_id"] == recording_id]
    if tracks is not None:
        df = df[df["track_id"].isin(list(tracks))]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    for track_id, track_df in df.groupby("track_id"):
        ax.plot(track_df["x"], track_df["y"], label=f"track {track_id}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Vehicle trajectories")
    ax.legend(loc="best", fontsize="small")
    ax.axis("equal")
    if show:
        plt.show()
    return ax


def plot_prediction(
    history: pd.DataFrame,
    future_true: Optional[pd.DataFrame] = None,
    future_pred: Optional[pd.DataFrame] = None,
    show: bool = False,
    ax: Optional[plt.Axes] = None,
):
    """Visualize one rollout in Frenet coordinates (s, n).
    以 Frenet 坐标系 (s, n) 可视化单次轨迹展开。

    """

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.plot(history["s"], history["n"], label="history", color="tab:blue")
    if future_true is not None:
        ax.plot(
            future_true["s"],
            future_true["n"],
            label="ground truth",
            color="tab:green",
            linestyle="--",
        )
    if future_pred is not None:
        ax.plot(
            future_pred["s"],
            future_pred["n"],
            label="prediction",
            color="tab:orange",
            linestyle=":",
        )

    ax.set_xlabel("s [m]")
    ax.set_ylabel("n [m]")
    ax.set_title("Trajectory prediction")
    ax.legend(loc="best")
    if show:
        plt.show()
    return ax


__all__ = ["plot_scene", "plot_prediction"]
