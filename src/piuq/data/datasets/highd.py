from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseDataset


class HighDDataset(BaseDataset):
    """Adapter for the highD dataset."""

    name = "highD"

    def load_raw(self, root: Path) -> pd.DataFrame:
        root = Path(root)
        track_files = sorted(root.glob("*_tracks.csv"))
        if not track_files:
            raise FileNotFoundError(
                f"No *_tracks.csv files found under {root}. Ensure highD is extracted."
            )

        records = []
        for tracks_path in track_files:
            rec_id = tracks_path.stem.split("_")[0]
            rec_meta_path = tracks_path.with_name(f"{rec_id}_recordingMeta.csv")
            if not rec_meta_path.exists():
                raise FileNotFoundError(f"Recording meta not found: {rec_meta_path}")
            rec_meta = pd.read_csv(rec_meta_path)
            frame_rate = float(rec_meta["frameRate"].iloc[0])
            recording_id = int(rec_meta["id"].iloc[0])

            df = pd.read_csv(tracks_path)
            df["x_center"] = df["x"] + df["width"] / 2.0
            df["y_center"] = df["y"] + df["height"] / 2.0

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
                }
            )

            df["x"] = df["x_center"]
            df["y"] = df["y_center"]
            df["t"] = (df["frame"] - df["frame"].min()) / frame_rate
            df["dataset"] = self.name
            df["recording_id"] = recording_id
            records.append(df)

        out = pd.concat(records, ignore_index=True)
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
            "lane_id",
            "width",
            "height",
            "vehicle_type",
        ]
        keep_cols = [c for c in keep_cols if c in out.columns]
        return out[keep_cols]

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
