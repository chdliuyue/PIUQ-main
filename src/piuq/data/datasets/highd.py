from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .base import BaseDataset


class HighDDataset(BaseDataset):
    """Adapter for the highD dataset.

    The expected layout matches the official release, where CSVs live in a
    ``data/`` subfolder such as ``<raw_root>/highD/data/01_tracks.csv``.
    """

    name = "highD"

    def __init__(self, centerline_points: int = 200) -> None:
        super().__init__()
        self.centerline_points = centerline_points
        self.recording_context: Dict[int, Dict[str, float | List[np.ndarray]]] = {}

    @staticmethod
    def _require_columns(df: pd.DataFrame, required: List[str], context: str) -> None:
        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise ValueError(
                f"HighD {context} missing required columns: {', '.join(missing)}"
            )

    def load_raw(self, root: Path) -> pd.DataFrame:
        self.recording_context = {}
        root = Path(root)
        search_root = root / "data" if (root / "data").exists() else root
        track_files = sorted(search_root.glob("*_tracks.csv"))
        if not track_files:
            raise FileNotFoundError(
                "No *_tracks.csv files found under "
                f"{search_root}. Ensure highD is extracted with the data/ folder intact."
            )

        records: List[pd.DataFrame] = []
        for tracks_path in track_files:
            rec_id = tracks_path.stem.split("_")[0]
            rec_meta_path = tracks_path.with_name(f"{rec_id}_recordingMeta.csv")
            tracks_meta_path = tracks_path.with_name(f"{rec_id}_tracksMeta.csv")
            if not rec_meta_path.exists():
                raise FileNotFoundError(f"Recording meta not found: {rec_meta_path}")
            if not tracks_meta_path.exists():
                raise FileNotFoundError(f"Tracks meta not found: {tracks_meta_path}")
            rec_meta = pd.read_csv(rec_meta_path)
            context = self._extract_recording_context(rec_meta)
            self.recording_context[int(rec_meta["id"].iloc[0])] = context

            df = self._standardize_tracks(pd.read_csv(tracks_path), context["frame_rate"])
            df["recording_id"] = int(rec_meta["id"].iloc[0])
            df["frame_rate"] = context["frame_rate"]
            df["speed_limit"] = context["speed_limit"]
            df["recording_location"] = context["recording_location"]
            df["dataset"] = self.name

            track_meta = self._standardize_tracks_meta(pd.read_csv(tracks_meta_path))
            df = df.merge(track_meta, on="track_id", how="left")

            df = self._compute_kinematics(df, context["frame_rate"])
            df = self._compute_gaps(df)
            df = self._compute_lane_offsets(df)
            df = self._compute_uncertainty(df)
            df = self._compute_normalized(df)

            df = df.sort_values(["frame", "track_id"]).reset_index(drop=True)

            records.append(df)

        out = pd.concat(records, ignore_index=True)
        out = self._annotate_traffic_flow(out)
        out = self.to_frenet(out)

        keep_cols = [
            "dataset",
            "recording_id",
            "track_id",
            "frame",
            "t",
            "x",
            "y",
            "vx",
            "vy",
            "ax",
            "ay",
            "speed",
            "accel_mag",
            "jerk",
            "jerk_x",
            "jerk_y",
            "lane_id",
            "lane_offset",
            "width",
            "height",
            "vehicle_type",
            "num_lane_changes",
            "vx_mean",
            "vy_mean",
            "ax_mean",
            "ay_mean",
            "vx_var",
            "vy_var",
            "ax_var",
            "ay_var",
            "speed_var",
            "missing_rate",
            "has_missing",
            "speed_limit",
            "frame_rate",
            "recording_location",
            "dhw",
            "thw",
            "ttc",
            "density",
            "frame_mean_speed",
            "frame_headway_mean",
            "frame_headway_median",
            "vx_norm",
            "vy_norm",
            "speed_norm",
            "ax_norm",
            "ay_norm",
            "sx",
            "sy",
            "svx",
            "svy",
            "sax",
            "say",
        ]

        keep_cols = [c for c in keep_cols if c in out.columns]
        return out[keep_cols]

    def _extract_recording_context(self, rec_meta: pd.DataFrame) -> Dict[str, float]:
        frame_rate = float(rec_meta.get("frameRate", pd.Series([np.nan])).iloc[0])
        speed_limit_raw = rec_meta.get("speedLimit", pd.Series([np.nan])).iloc[0]
        speed_limit = float(speed_limit_raw) if not pd.isna(speed_limit_raw) else np.nan
        if not pd.isna(speed_limit):
            speed_limit = speed_limit / 3.6  # km/h -> m/s
        location_key = "locationId" if "locationId" in rec_meta.columns else "location"
        location = rec_meta.get(location_key, pd.Series([np.nan])).iloc[0]
        direction = rec_meta.get("drivingDirection", pd.Series([np.nan])).iloc[0]
        upper_markings = self._parse_lane_markings(rec_meta.get("upperLaneMarkings"))
        lower_markings = self._parse_lane_markings(rec_meta.get("lowerLaneMarkings"))
        lane_markings = upper_markings + lower_markings

        return {
            "frame_rate": frame_rate,
            "speed_limit": speed_limit,
            "recording_location": location,
            "driving_direction": float(direction) if not pd.isna(direction) else np.nan,
            "lane_markings": lane_markings,
            "upper_lane_markings": upper_markings,
            "lower_lane_markings": lower_markings,
        }

    def _standardize_tracks(self, df: pd.DataFrame, frame_rate: float) -> pd.DataFrame:
        required_cols = {
            "id",
            "frame",
            "x",
            "y",
            "width",
            "height",
            "xVelocity",
            "yVelocity",
            "xAcceleration",
            "yAcceleration",
            "laneId",
            "class",
        }
        self._require_columns(df, list(required_cols), "tracks")

        df = df.rename(
            columns={
                "id": "track_id",
                "frame": "frame",
                "xVelocity": "vx",
                "yVelocity": "vy",
                "xAcceleration": "ax",
                "yAcceleration": "ay",
                "laneId": "lane_id",
                "class": "vehicle_type",
                "precedingId": "preceding_id",
                "followingId": "following_id",
            }
        )
        df["track_id"] = df["track_id"].astype(int)
        df["frame"] = df["frame"].astype(int)

        df["x_center"] = df["x"] + df["width"] / 2.0
        df["y_center"] = df["y"] + df["height"] / 2.0
        df["x"] = df["x_center"]
        df["y"] = df["y_center"]
        if frame_rate and not np.isnan(frame_rate) and frame_rate > 0:
            df["t"] = (df["frame"].astype(float) - 1.0) / frame_rate
        else:
            df["t"] = np.nan
        return df

    def _standardize_tracks_meta(self, meta: pd.DataFrame) -> pd.DataFrame:
        required = ["id", "numFrames", "width", "height", "class"]
        self._require_columns(meta, required, "tracksMeta")

        if not {"startFrame", "initialFrame"} & set(meta.columns):
            raise ValueError(
                "HighD tracksMeta missing required columns: startFrame or initialFrame"
            )
        if not {"endFrame", "finalFrame"} & set(meta.columns):
            raise ValueError(
                "HighD tracksMeta missing required columns: endFrame or finalFrame"
            )

        rename_map = {
            "id": "track_id",
            "numLaneChanges": "num_lane_changes",
            "meanXVelocity": "vx_mean",
            "meanYVelocity": "vy_mean",
            "meanXAcceleration": "ax_mean",
            "meanYAcceleration": "ay_mean",
            "stdXVelocity": "vx_std",
            "stdYVelocity": "vy_std",
            "stdXAcceleration": "ax_std",
            "stdYAcceleration": "ay_std",
            "medianXVelocity": "vx_median",
            "medianYVelocity": "vy_median",
            "medianXAcceleration": "ax_median",
            "medianYAcceleration": "ay_median",
            "minXVelocity": "vx_min",
            "minYVelocity": "vy_min",
            "maxXVelocity": "vx_max",
            "maxYVelocity": "vy_max",
            "minXAcceleration": "ax_min",
            "minYAcceleration": "ay_min",
            "maxXAcceleration": "ax_max",
            "maxYAcceleration": "ay_max",
            "numFrames": "track_frames",
            "startFrame": "start_frame",
            "endFrame": "end_frame",
            "class": "vehicle_type",
            "initialFrame": "start_frame",
            "finalFrame": "end_frame",
            "traveledDistance": "traveled_distance",
            "minDHW": "min_dhw",
            "minTHW": "min_thw",
            "minTTC": "min_ttc",
        }

        meta = meta.rename(columns=rename_map)
        meta["track_id"] = meta["track_id"].astype(int)

        if {"track_frames", "start_frame", "end_frame"}.issubset(meta.columns):
            meta["start_frame"] = meta["start_frame"].astype(int)
            meta["end_frame"] = meta["end_frame"].astype(int)
            meta["track_frames"] = meta["track_frames"].astype(int)

            expected = (meta["end_frame"] - meta["start_frame"] + 1).astype(float)
            expected = expected.where(expected > 0)
            meta["missing_rate"] = 1.0 - meta["track_frames"] / expected
            meta["missing_rate"] = meta["missing_rate"].clip(lower=0)
        else:
            meta["missing_rate"] = np.nan
        meta["has_missing"] = meta["missing_rate"].fillna(0) > 0

        stats_cols = [
            "vx_mean",
            "vy_mean",
            "ax_mean",
            "ay_mean",
            "num_lane_changes",
            "missing_rate",
            "has_missing",
        ]
        stats_cols += [
            c
            for c in meta.columns
            if c.startswith("vx_") or c.startswith("vy_") or c.startswith("ax_") or c.startswith("ay_")
        ]
        keep = [
            "track_id",
            "vehicle_type",
            "width",
            "height",
            "track_frames",
            "start_frame",
            "end_frame",
            "traveled_distance",
            "min_dhw",
            "min_thw",
            "min_ttc",
        ] + list(dict.fromkeys(stats_cols))
        return meta[[c for c in keep if c in meta.columns]]

    def _compute_kinematics(self, df: pd.DataFrame, frame_rate: float) -> pd.DataFrame:
        df["speed"] = np.hypot(df.get("vx", 0.0), df.get("vy", 0.0))
        df["accel_mag"] = np.hypot(df.get("ax", 0.0), df.get("ay", 0.0))
        if frame_rate and not np.isnan(frame_rate) and frame_rate > 0:
            dt = 1.0 / frame_rate
            df["jerk_x"] = df.groupby("track_id")["ax"].diff() / dt
            df["jerk_y"] = df.groupby("track_id")["ay"].diff() / dt
            df["jerk"] = np.hypot(df["jerk_x"], df["jerk_y"])
        else:
            df[["jerk_x", "jerk_y", "jerk"]] = np.nan
            df["t"] = df["t"].astype(float)
        return df

    def _compute_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        if "preceding_id" not in df.columns:
            df[["dhw", "thw", "ttc"]] = np.nan
            return df

        prec_cols = [
            "recording_id",
            "frame",
            "track_id",
            "x",
            "y",
            "vx",
            "vy",
            "width",
            "height",
        ]
        prec_df = df[prec_cols].rename(columns={
            "track_id": "preceding_id",
            "x": "preceding_x",
            "y": "preceding_y",
            "vx": "preceding_vx",
            "vy": "preceding_vy",
            "width": "preceding_width",
            "height": "preceding_height",
        })
        merged = df.merge(prec_df, on=["recording_id", "frame", "preceding_id"], how="left")

        ego_front = merged["x"] + merged.get("width", 0.0) / 2.0
        preceding_back = merged["preceding_x"] - merged.get("preceding_width", 0.0) / 2.0
        merged["dhw"] = preceding_back - ego_front

        merged["thw"] = merged["dhw"] / merged["speed"]
        closing_speed = merged["vx"] - merged["preceding_vx"]
        merged["ttc"] = np.where(closing_speed > 1e-3, merged["dhw"] / closing_speed, np.nan)
        return merged

    def _compute_lane_offsets(self, df: pd.DataFrame) -> pd.DataFrame:
        if "lane_id" not in df.columns:
            df["lane_offset"] = np.nan
            return df
        lane_centers = (
            df.groupby(["recording_id", "lane_id"])['y']
            .mean()
            .rename("lane_center_y")
            .reset_index()
        )
        df = df.merge(lane_centers, on=["recording_id", "lane_id"], how="left")
        df["lane_offset"] = df["y"] - df["lane_center_y"]
        return df.drop(columns=["lane_center_y"])

    def _compute_normalized(self, df: pd.DataFrame) -> pd.DataFrame:
        speed_limit = df.get("speed_limit", np.nan)
        df["speed_norm"] = df["speed"] / speed_limit
        df["vx_norm"] = df.get("vx", np.nan) / speed_limit
        df["vy_norm"] = df.get("vy", np.nan) / speed_limit
        df["ax_norm"] = df.get("ax", np.nan) / speed_limit
        df["ay_norm"] = df.get("ay", np.nan) / speed_limit
        return df

    def _compute_uncertainty(self, df: pd.DataFrame) -> pd.DataFrame:
        var_stats = df.groupby("track_id")[['vx', 'vy', 'ax', 'ay', 'speed']].var()
        var_stats = var_stats.rename(columns={
            "vx": "vx_var",
            "vy": "vy_var",
            "ax": "ax_var",
            "ay": "ay_var",
            "speed": "speed_var",
        })
        var_stats = var_stats.reset_index()
        df = df.merge(var_stats, on="track_id", how="left")
        return df

    def _annotate_traffic_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        records: List[pd.DataFrame] = []
        for rec_id, rec_df in df.groupby("recording_id"):
            road_length = float(rec_df["x"].max() - rec_df["x"].min())
            frame_group = rec_df.groupby("frame")
            density = frame_group["track_id"].nunique() / road_length if road_length > 0 else np.nan
            speed_mean = frame_group["speed"].mean()
            headway_mean = frame_group["dhw"].mean()
            headway_median = frame_group["dhw"].median()
            per_frame = pd.DataFrame({
                "frame": density.index,
                "density": density.values,
                "frame_mean_speed": speed_mean.values,
                "frame_headway_mean": headway_mean.values,
                "frame_headway_median": headway_median.values,
            })
            rec_df = rec_df.merge(per_frame, on="frame", how="left")
            records.append(rec_df)
        return pd.concat(records, ignore_index=True)

    def build_centerline(self, df: pd.DataFrame) -> np.ndarray:
        centerlines, _ = self.build_centerlines(df)

        if not centerlines:
            return self._centerline_from_tracks(df)

        rec_id = int(df["recording_id"].iloc[0]) if "recording_id" in df.columns else None
        context = self.recording_context.get(rec_id, {}) if rec_id is not None else {}
        driving_direction = context.get("driving_direction", np.nan) if context else np.nan

        if not np.isnan(driving_direction) and driving_direction in centerlines:
            return centerlines[int(driving_direction)]

        return next(iter(centerlines.values()))

    def build_centerlines(
        self, df: pd.DataFrame
    ) -> tuple[Dict[int, np.ndarray], Dict[int, int]]:
        rec_id = int(df["recording_id"].iloc[0]) if "recording_id" in df.columns else None
        context = self.recording_context.get(rec_id, {}) if rec_id is not None else {}

        lane_direction_map = self._infer_lane_directions(df)

        def _lane_subset(direction: int) -> pd.DataFrame:
            lane_ids = [lane_id for lane_id, d in lane_direction_map.items() if d == direction]
            if "lane_id" not in df.columns or not lane_ids:
                return df[df.get("drivingDirection", pd.Series(dtype=float)) == direction]
            return df[df["lane_id"].isin(lane_ids)]

        centerlines: Dict[int, np.ndarray] = {}
        direction_markings = {
            1: context.get("upper_lane_markings", []),
            2: context.get("lower_lane_markings", []),
        }

        for direction in (1, 2):
            markings = direction_markings.get(direction, [])
            centerline = None
            if markings:
                centerline = self._centerline_from_lane_markings(markings)

            lane_df = _lane_subset(direction)
            if centerline is None and not lane_df.empty:
                centerline = self._centerline_from_tracks(lane_df)

            if centerline is not None:
                centerlines[direction] = self._orient_centerline(centerline, float(direction))

        if not centerlines:
            lane_markings: List[np.ndarray] = context.get("lane_markings", []) if context else []
            driving_direction = context.get("driving_direction", np.nan) if context else np.nan
            centerline = None
            if lane_markings:
                centerline = self._centerline_from_lane_markings(lane_markings)
            if centerline is None:
                centerline = self._centerline_from_tracks(df)
            direction = 1 if np.isnan(driving_direction) else int(driving_direction)
            centerlines[direction] = self._orient_centerline(centerline, float(direction))

        return centerlines, lane_direction_map

    def _infer_lane_directions(self, df: pd.DataFrame) -> Dict[int, int]:
        lane_direction: Dict[int, int] = {}

        if "lane_id" not in df.columns:
            return lane_direction

        if "drivingDirection" in df.columns:
            for lane_id, lane_df in df.groupby("lane_id"):
                dir_vals = lane_df["drivingDirection"].dropna()
                if len(dir_vals):
                    lane_direction[int(lane_id)] = int(dir_vals.mode().iloc[0])

        if lane_direction and len(set(lane_direction.values())) >= 2:
            return lane_direction

        medians = df.groupby("lane_id")["y"].median().sort_values()
        if len(medians) == 1:
            lane_direction[int(medians.index[0])] = lane_direction.get(
                int(medians.index[0]), 1
            )
            return lane_direction

        split = len(medians) // 2
        upper_lanes = medians.index[:split]
        lower_lanes = medians.index[split:]

        for lane_id in upper_lanes:
            lane_direction.setdefault(int(lane_id), 1)
        for lane_id in lower_lanes:
            lane_direction.setdefault(int(lane_id), 2)

        return lane_direction

    # --- lane geometry helpers -------------------------------------------------

    def _parse_lane_markings(self, series: pd.Series | None) -> List[np.ndarray]:
        if series is None or len(series) == 0:
            return []
        raw = series.iloc[0]
        if pd.isna(raw):
            return []

        if isinstance(raw, str):
            cleaned = raw.replace(";", ",")
            try:
                data = ast.literal_eval(cleaned)
            except (SyntaxError, ValueError):
                floats = [
                    float(v)
                    for v in cleaned.replace("[", "").replace("]", "").split(",")
                    if v.strip()
                ]
                data = floats
        else:
            data = raw

        return self._normalize_marking_data(data)

    def _normalize_marking_data(self, data: object) -> List[np.ndarray]:
        if isinstance(data, (list, tuple, np.ndarray)):
            arr = np.asarray(data, dtype=float)
        else:
            return []

        if arr.ndim == 1:
            return [self._expand_marking_1d(arr)] if arr.size else []

        if arr.ndim == 2:
            if arr.shape[1] != 2:
                return [self._expand_marking_1d(arr.ravel())]
            return [arr]

        markings: List[np.ndarray] = []
        for sub in arr:
            sub_arr = np.asarray(sub, dtype=float)
            if sub_arr.ndim == 1:
                markings.append(self._expand_marking_1d(sub_arr))
            else:
                markings.append(sub_arr.reshape(-1, 2))
        return [m for m in markings if m.size]

    def _expand_marking_1d(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float).reshape(-1)
        x_axis = np.linspace(0.0, 1.0, num=arr.size)
        return np.column_stack([x_axis, arr])

    def _resample_polyline(self, polyline: np.ndarray) -> np.ndarray:
        diffs = np.diff(polyline, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = cumulative[-1]
        if total == 0:
            return np.vstack([polyline[0], polyline[-1]])

        target = np.linspace(0.0, total, num=self.centerline_points)
        resampled = np.empty((self.centerline_points, 2), dtype=float)

        for i, s in enumerate(target):
            idx = np.searchsorted(cumulative, s, side="right") - 1
            idx = min(idx, len(seg_len) - 1)
            s0, s1 = cumulative[idx], cumulative[idx + 1]
            t = 0.0 if s1 == s0 else (s - s0) / (s1 - s0)
            resampled[i] = polyline[idx] + t * diffs[idx]
        return resampled

    def _centerline_from_lane_markings(self, lane_markings: Sequence[np.ndarray]) -> np.ndarray | None:
        polylines = [self._resample_polyline(m) for m in lane_markings if len(m) >= 2]
        if not polylines:
            return None
        if len(polylines) == 1:
            return polylines[0]

        stack = np.stack(polylines, axis=0)
        lane_centers = 0.5 * (stack[:-1] + stack[1:])
        centerline = lane_centers.mean(axis=0)
        return centerline

    def _centerline_from_tracks(self, df: pd.DataFrame) -> np.ndarray:
        x_min = float(df["x"].min())
        x_max = float(df["x"].max())
        sample_x = np.linspace(x_min, x_max, num=self.centerline_points)

        def _interp_lane(lane_df: pd.DataFrame) -> np.ndarray:
            lane_sorted = lane_df.sort_values("x")
            x_vals = lane_sorted["x"].to_numpy()
            y_vals = lane_sorted["y"].to_numpy()
            unique_x, idx = np.unique(x_vals, return_index=True)
            y_unique = y_vals[idx]
            y_interp = np.interp(sample_x, unique_x, y_unique)
            return y_interp

        lane_centers: List[np.ndarray] = []
        if "lane_id" in df.columns:
            for _, lane_df in df.groupby("lane_id"):
                lane_centers.append(_interp_lane(lane_df))

        if not lane_centers:
            y_interp = _interp_lane(df)
            lane_centers.append(y_interp)

        y_center = np.mean(lane_centers, axis=0)
        return np.column_stack([sample_x, y_center])

    def _orient_centerline(self, centerline: np.ndarray, driving_direction: float) -> np.ndarray:
        if np.isnan(driving_direction):
            return centerline

        increasing = centerline[-1, 0] >= centerline[0, 0]
        if (driving_direction == 1 and increasing) or (driving_direction == 2 and not increasing):
            return centerline[::-1]
        return centerline

    def _to_frenet_single(self, df: pd.DataFrame) -> pd.DataFrame:
        centerlines, lane_direction_map = self.build_centerlines(df)

        frames = {direction: FrenetFrame(cl) for direction, cl in centerlines.items()}

        out = df.copy()
        N = len(out)
        res_arrays: Dict[str, np.ndarray] = {
            "s": np.full(N, np.nan),
            "n": np.full(N, np.nan),
            "seg_idx": np.full(N, -1, dtype=int),
        }

        have_velocity = {"vx", "vy"}.issubset(out.columns)
        have_accel = {"ax", "ay"}.issubset(out.columns)
        if have_velocity:
            res_arrays.update({"v_s": np.full(N, np.nan), "v_n": np.full(N, np.nan)})
        if have_accel:
            res_arrays.update({"a_s": np.full(N, np.nan), "a_n": np.full(N, np.nan)})

        def _assign_direction(idx: int, row: pd.Series) -> int:
            if "drivingDirection" in row and not pd.isna(row["drivingDirection"]):
                return int(row["drivingDirection"])
            if "lane_id" in row and row["lane_id"] in lane_direction_map:
                return lane_direction_map[int(row["lane_id"])]
            if len(centerlines) == 1:
                return next(iter(centerlines))
            return sorted(centerlines.keys())[0]

        direction_indices: Dict[int, List[int]] = {}
        for idx, row in out.iterrows():
            direction = _assign_direction(idx, row)
            if direction not in centerlines:
                direction = next(iter(centerlines))
            direction_indices.setdefault(direction, []).append(idx)

        for direction, indices in direction_indices.items():
            subset = out.loc[indices]
            frenet = frames[direction]

            xy = subset[["x", "y"]].to_numpy()
            v_xy = subset[["vx", "vy"]].to_numpy() if have_velocity else None
            a_xy = subset[["ax", "ay"]].to_numpy() if have_accel else None

            res = frenet.to_frenet(xy, v_xy=v_xy, a_xy=a_xy)
            for key, values in res.items():
                res_arrays[key][subset.index] = values

        for key, values in res_arrays.items():
            out[key] = values

        return out


def dataset_factory(name: str):
    name = name.lower()
    if name == "highd":
        return HighDDataset()
    raise ValueError(f"Unsupported dataset: {name}")


__all__ = ["HighDDataset", "dataset_factory"]
