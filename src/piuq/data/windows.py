from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


class WindowBuilder:
    """Build sliding windows for ego trajectories and neighbors.
    为自车轨迹与邻车构建滑动窗口。

    """
    def __init__(
        self,
        history_sec: float,
        future_sec: float,
        step_sec: float,
        neighbor_radius_s: float,
        max_neighbors: int,
        allow_gaps: bool = False,
        risk_ttc_thresholds: Iterable[float] = (5.0, 3.0, 1.5),
        physics_residual_aggregation: str = "mean_abs",
    ) -> None:
        self.history_sec = history_sec
        self.future_sec = future_sec
        self.step_sec = step_sec
        self.neighbor_radius_s = neighbor_radius_s
        self.max_neighbors = max_neighbors
        self.allow_gaps = allow_gaps
        self.risk_ttc_thresholds = tuple(sorted(risk_ttc_thresholds, reverse=True))
        self.physics_residual_aggregation = physics_residual_aggregation

    @staticmethod
    def _kinematics(values: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute velocity/acceleration/jerk from a sequence of 2D positions."""

        velocity = np.gradient(values, dt, axis=0)
        acceleration = np.gradient(velocity, dt, axis=0)
        jerk = np.gradient(acceleration, dt, axis=0)
        return velocity, acceleration, jerk

    @staticmethod
    def _neighbor_speed(track_df: pd.DataFrame, frame: int, dt: float) -> float:
        """Approximate longitudinal speed of a neighbor around the target frame."""

        frames = track_df["frame"].to_numpy()
        s_values = track_df["s"].to_numpy()
        if frame not in frames:
            return 0.0
        idx = int(np.where(frames == frame)[0][0])
        if idx == 0:
            return 0.0
        return float((s_values[idx] - s_values[idx - 1]) / dt)

    def _physics_features(
        self,
        hist_df: pd.DataFrame,
        neighbors: pd.DataFrame,
        track_lookup: Dict[Any, pd.DataFrame],
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Derive physics and uncertainty cues for the center frame."""

        positions = hist_df[["s", "n"]].to_numpy()
        velocity, acceleration, jerk = self._kinematics(positions, dt)
        jerk_mag = float(np.linalg.norm(jerk[-1])) if len(jerk) > 0 else 0.0

        ego_speed_s = float(velocity[-1, 0]) if len(velocity) > 0 else 0.0
        var_vel = np.var(velocity, axis=0) if len(velocity) > 0 else np.zeros(2)
        var_acc = np.var(acceleration, axis=0) if len(acceleration) > 0 else np.zeros(2)
        uncertainty = np.concatenate([var_vel, var_acc]).astype(np.float32)

        ttc = np.inf
        thw = np.inf
        drac = 0.0
        if not neighbors.empty:
            ahead = neighbors[neighbors["s"] > hist_df["s"].iloc[-1]]
            if not ahead.empty:
                leader = ahead.iloc[0]
                distance = float(leader["s"] - hist_df["s"].iloc[-1])
                leader_track = track_lookup.get(leader["track_id"])
                leader_speed = (
                    self._neighbor_speed(leader_track, int(leader["frame"]), dt)
                    if leader_track is not None
                    else 0.0
                )
                rel_speed = ego_speed_s - leader_speed
                if rel_speed > 1e-3:
                    ttc = max(distance / rel_speed, 0.0)
                if ego_speed_s > 1e-3:
                    thw = max(distance / ego_speed_s, 0.0)
                if distance > 1e-3:
                    drac = max((rel_speed**2) / (2 * distance), 0.0)

        lwr_residual = 0.0
        if len(velocity) > 2:
            l_residual = np.gradient(velocity[:, 0], dt)
            lwr_residual = float(self._aggregate_residual(l_residual))

        physics = np.array([ttc, thw, drac, jerk_mag, lwr_residual], dtype=np.float32)
        return physics, uncertainty, ttc, drac

    def _aggregate_residual(self, values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0

        mode = self.physics_residual_aggregation.lower()
        abs_values = np.abs(values)

        if mode in {"mean", "mean_abs"}:
            data = abs_values if mode.endswith("abs") else values
            return float(np.mean(data))
        if mode == "median_abs":
            return float(np.median(abs_values))
        if mode == "max_abs":
            return float(np.max(abs_values))
        raise ValueError(f"Unknown physics residual aggregation: {self.physics_residual_aggregation}")

    @staticmethod
    def _filter_neighbors_by_direction(neighbors: pd.DataFrame, ego_state: pd.Series) -> pd.DataFrame:
        """Keep neighbors that share the ego's driving direction or lane group."""

        for column in ("lane_group", "driving_direction"):
            if column in neighbors.columns and column in ego_state.index:
                ego_value = ego_state.get(column, np.nan)
                if pd.notna(ego_value):
                    return neighbors[neighbors[column] == ego_value]
        return neighbors

    @staticmethod
    def _ttc_min_future(fut_df: pd.DataFrame) -> float:
        """Minimum finite TTC over the ego future; infinity if none exist."""

        if "ttc" not in fut_df.columns:
            return np.inf
        ttc_values = fut_df["ttc"].to_numpy(dtype=float)
        mask = np.isfinite(ttc_values) & (ttc_values > 0.0)
        if not np.any(mask):
            return np.inf
        return float(np.nanmin(ttc_values[mask]))

    def _risk_label_from_ttc(self, ttc_min_future: float) -> int:
        """Map TTC minimum to a discrete risk tier based on configured thresholds."""

        if not self.risk_ttc_thresholds:
            return 0
        for idx, threshold in enumerate(self.risk_ttc_thresholds):
            if ttc_min_future > threshold:
                return idx
        return len(self.risk_ttc_thresholds)

    def build(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate window dictionaries grouped by dataset and recording.
        按数据集与录像编号生成窗口字典。

        """
        required = {"dataset", "recording_id", "track_id", "frame", "t", "s", "n"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        windows: List[Dict[str, Any]] = []
        for (dataset, rec_id), scene_df in df.groupby(["dataset", "recording_id"]):
            scene_df = scene_df.sort_values(["frame", "track_id"])
            dt_series = scene_df.groupby("track_id")["t"].diff().dropna()
            if dt_series.empty:
                continue
            dt = float(dt_series.median())
            history_frames = int(round(self.history_sec / dt))
            future_frames = int(round(self.future_sec / dt))
            step_frames = max(1, int(round(self.step_sec / dt)))

            track_lookup = {tid: g.sort_values("frame") for tid, g in scene_df.groupby("track_id")}

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
                    neighbors = self._filter_neighbors_by_direction(neighbors, ego_state)
                    ds = np.array([], dtype=float)
                    if not neighbors.empty:
                        ds = np.abs(neighbors["s"].to_numpy() - ego_state["s"])
                    neighbors = neighbors.assign(ds=ds)
                    neighbors = neighbors[neighbors["ds"] <= self.neighbor_radius_s]
                    neighbors = neighbors.sort_values("ds").head(self.max_neighbors)

                    physics, uncertainty, ttc, drac = self._physics_features(
                        hist_df, neighbors, track_lookup, dt
                    )
                    ttc_min_future = self._ttc_min_future(fut_df)
                    risk_label = self._risk_label_from_ttc(ttc_min_future)
                    scene_label = 2 if len(neighbors) >= 5 else 1 if len(neighbors) >= 2 else 0

                    window = {
                        "dataset": dataset,
                        "recording_id": rec_id,
                        "ego_track_id": track_id,
                        "center_frame": center_frame,
                        "center_time": center_time,
                        "history": hist_df.reset_index(drop=True),
                        "future": fut_df.reset_index(drop=True),
                        "neighbors": neighbors.reset_index(drop=True),
                        "physics_features": physics,
                        "uncertainty_features": uncertainty,
                        "risk_label": risk_label,
                        "scene_label": scene_label,
                        "ttc_min_future": ttc_min_future,
                    }
                    windows.append(window)
        return windows
