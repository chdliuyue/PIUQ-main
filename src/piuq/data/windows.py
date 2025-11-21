from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


class WindowBuilder:
    def __init__(
        self,
        history_sec: float,
        future_sec: float,
        step_sec: float,
        neighbor_radius_s: float,
        max_neighbors: int,
        allow_gaps: bool = False,
    ) -> None:
        self.history_sec = history_sec
        self.future_sec = future_sec
        self.step_sec = step_sec
        self.neighbor_radius_s = neighbor_radius_s
        self.max_neighbors = max_neighbors
        self.allow_gaps = allow_gaps

    def build(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        required = {"dataset", "recording_id", "track_id", "frame", "t", "s", "n"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        windows: List[Dict[str, Any]] = []
        for (dataset, rec_id), scene_df in df.groupby(["dataset", "recording_id"]):
            scene_df = scene_df.sort_values(["track_id", "frame"])
            dt_series = scene_df.groupby("track_id")["t"].diff().dropna()
            if dt_series.empty:
                continue
            dt = float(dt_series.median())
            history_frames = int(round(self.history_sec / dt))
            future_frames = int(round(self.future_sec / dt))
            step_frames = max(1, int(round(self.step_sec / dt)))

            frame_lookup = dict(tuple(scene_df.groupby("frame")))
            for track_id, track_df in scene_df.groupby("track_id"):
                track_df = track_df.sort_values("frame")
                frames = track_df["frame"].to_numpy()
                if len(frames) < history_frames + future_frames + 1:
                    continue

                for idx in range(history_frames, len(frames) - future_frames, step_frames):
                    hist_frames = frames[idx - history_frames : idx + 1]
                    fut_frames = frames[idx + 1 : idx + 1 + future_frames]
                    if not self.allow_gaps:
                        if not (
                            np.all(np.diff(hist_frames) == 1)
                            and np.all(np.diff(fut_frames) == 1)
                        ):
                            continue

                    hist_df = track_df[track_df["frame"].isin(hist_frames)].copy()
                    fut_df = track_df[track_df["frame"].isin(fut_frames)].copy()
                    if hist_df.shape[0] != len(hist_frames) or fut_df.shape[0] != len(fut_frames):
                        continue

                    center_frame = int(hist_df["frame"].iloc[-1])
                    center_time = float(hist_df["t"].iloc[-1])

                    neighbors_df_all = frame_lookup.get(center_frame)
                    if neighbors_df_all is None:
                        continue

                    ego_state = hist_df.iloc[-1]
                    neighbors = neighbors_df_all[neighbors_df_all["track_id"] != track_id]
                    if not neighbors.empty:
                        ds = np.abs(neighbors["s"].to_numpy() - ego_state["s"])
                        neighbors = neighbors.assign(ds=ds)
                        neighbors = neighbors[neighbors["ds"] <= self.neighbor_radius_s]
                        neighbors = neighbors.sort_values("ds").head(self.max_neighbors)

                    window = {
                        "dataset": dataset,
                        "recording_id": rec_id,
                        "ego_track_id": track_id,
                        "center_frame": center_frame,
                        "center_time": center_time,
                        "history": hist_df.reset_index(drop=True),
                        "future": fut_df.reset_index(drop=True),
                        "neighbors": neighbors.reset_index(drop=True),
                    }
                    windows.append(window)
        return windows
