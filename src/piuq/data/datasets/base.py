from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..geometry.frenet import FrenetFrame


class BaseDataset(ABC):
    name: str = "base"

    @abstractmethod
    def load_raw(self, root: Path) -> pd.DataFrame:
        """Load raw trajectories into a unified DataFrame."""

    @abstractmethod
    def build_centerline(self, df: pd.DataFrame) -> np.ndarray:
        """Return centerline polyline for the scene."""

    def to_frenet(self, df: pd.DataFrame) -> pd.DataFrame:
        centerline = self.build_centerline(df)
        frenet = FrenetFrame(centerline)

        xy = df[["x", "y"]].to_numpy()
        v_xy = df[["vx", "vy"]].to_numpy() if {"vx", "vy"}.issubset(df.columns) else None
        a_xy = df[["ax", "ay"]].to_numpy() if {"ax", "ay"}.issubset(df.columns) else None

        res: Dict[str, np.ndarray] = frenet.to_frenet(xy, v_xy=v_xy, a_xy=a_xy)
        out = df.copy()
        for k, v in res.items():
            out[k] = v
        return out
