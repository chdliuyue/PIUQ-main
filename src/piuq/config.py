from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    raw_data: Path = Path("data/raw")
    processed_data: Path = Path("data/processed")
    logs: Path = Path("logs")


class SplitConfig(BaseModel):
    key: str = "recording_id"
    ratios: Dict[str, float] = Field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    seed: int = 42


class PreprocessConfig(BaseModel):
    sampling_hz: float = 25.0
    downsample_hz: float = 10.0
    smooth_window: int = 7
    centerline_points: int = 200
    datasets: List[str] = Field(default_factory=lambda: ["highD"])
    dataset_config_dir: Path = Path("configs/datasets")
    history_sec: float = 3.0
    future_sec: float = 3.0
    neighbor_radius_s: float = 150.0
    max_neighbors: int = 32
    allow_gaps: bool = False
    split: SplitConfig = Field(default_factory=SplitConfig)


class WindowsConfig(BaseModel):
    step_sec: float = 0.5
    pad_value: float = 0.0


class TrainingConfig(BaseModel):
    batch_size: int = 16
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    physics_loss_weight: float = 1.0
    uncertainty_loss_weight: float = 0.1


class ProjectConfig(BaseModel):
    name: str = "p-i-uncertainty-highway"
    seed: int = 42


class Config(BaseModel):
    project: ProjectConfig = ProjectConfig()
    paths: PathsConfig = PathsConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    windows: WindowsConfig = WindowsConfig()
    training: TrainingConfig = TrainingConfig()


# --- helpers ---------------------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries.
    递归地合并两个字典。

    """
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _merge_dict(dict(base[k]), v)
        else:
            base[k] = v
    return base


def _apply_env_overrides(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    for k, v in data.items():
        path = f"{prefix}.{k}" if prefix else k
        env_key = path.replace(".", "_").upper()
        if isinstance(v, dict):
            data[k] = _apply_env_overrides(v, path)
        else:
            env_val = None
            # Direct environment var overrides win.
            # 直接的环境变量覆盖优先级最高。
            import os

            if env_key in os.environ:
                env_val = os.environ[env_key]
            if env_val is not None:
                if isinstance(v, bool):
                    data[k] = env_val.lower() in {"1", "true", "yes"}
                elif isinstance(v, int):
                    data[k] = int(env_val)
                elif isinstance(v, float):
                    data[k] = float(env_val)
                else:
                    data[k] = env_val
    return data


def load_config(base: Path, overrides: Iterable[Path] | None = None) -> Config:
    """Load a YAML config with optional overrides and environment variables.
    加载 YAML 配置，可选地应用覆盖文件并支持环境变量。

    Parameters
    ----------
    base: Path
        Base YAML file path.
        基础 YAML 文件路径。
    overrides: iterable of Path, optional
        Additional YAML files applied in order.
        可按顺序应用的额外 YAML 文件。
    """

    base_dict = _load_yaml(base)
    for override in overrides or []:
        base_dict = _merge_dict(base_dict, _load_yaml(override))

    base_dict = _apply_env_overrides(base_dict)
    return Config.parse_obj(base_dict)


__all__ = [
    "Config",
    "load_config",
]
