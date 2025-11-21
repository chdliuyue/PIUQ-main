from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel, Field, validator


class DatasetProcessConfig(BaseModel):
    """Per-dataset preprocessing options and feature layout.
    针对每个数据集的预处理选项与特征布局。

    """

    dataset: str
    raw_subdir: str
    sampling_hz: float
    target_hz: float
    smoothing_window: int = 0
    feature_groups: Dict[str, List[str]] = Field(default_factory=dict)

    @validator("dataset")
    def _lower_dataset(cls, value: str) -> str:  # pragma: no cover - trivial
        return value.strip()

    @property
    def ordered_feature_names(self) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for group in [
            "identifiers",
            "kinematics",
            "lane",
            "flow",
            "physics",
            "uncertainty",
            "frenet",
            "targets",
        ]:
            for name in self.feature_groups.get(group, []):
                if name not in seen:
                    ordered.append(name)
                    seen.add(name)
        return ordered

    def fill_missing_columns(self, df):
        """Ensure all configured columns exist so downstream code is consistent.
        确保所有配置的列都存在，以便下游代码保持一致。

        """

        import pandas as pd

        for col in self.ordered_feature_names:
            if col not in df.columns:
                df[col] = pd.NA
        return df[self.ordered_feature_names]


def load_dataset_process_config(name: str, config_dir: Path) -> DatasetProcessConfig:
    path = Path(config_dir) / f"{name.lower()}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset config not found for '{name}'. Expected at {path}."
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return DatasetProcessConfig(**data)


__all__ = ["DatasetProcessConfig", "load_dataset_process_config"]
