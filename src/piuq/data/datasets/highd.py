from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .base import BaseDataset


class HighDDataset(BaseDataset):
    """Adapter for the highD dataset.

    The expected layout matches the official release, where CSVs live in a
    ``data/`` subfolder such as ``<raw_root>/highD/data/01_tracks.csv``.
    """

    name = "highD"

    @staticmethod
    def _require_columns(df: pd.DataFrame, required: List[str], context: str) -> None:
        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise ValueError(
                f"HighD {context} missing required columns: {', '.join(missing)}"
            )

    def load_raw(self, root: Path) -> pd.DataFrame:
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

            df = self._standardize_tracks(pd.read_csv(tracks_path))
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
        return {
            "frame_rate": frame_rate,
            "speed_limit": speed_limit,
            "recording_location": location,
        }

    def _standardize_tracks(self, df: pd.DataFrame) -> pd.DataFrame:
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
        df["t"] = df.groupby("track_id")["frame"].transform(lambda s: s - s.min())
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
            df["t"] = df["t"] / frame_rate
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
        x_min = float(df["x"].min())
        x_max = float(df["x"].max())
        y_center = float(df["y"].mean())
        return np.array([[x_min, y_center], [x_max, y_center]], dtype=float)


def dataset_factory(name: str):
    name = name.lower()
    if name == "highd":
        return HighDDataset()
    raise ValueError(f"Unsupported dataset: {name}")


__all__ = ["HighDDataset", "dataset_factory"]
