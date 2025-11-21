from pathlib import Path

import pandas as pd

from piuq.data.configs import DatasetProcessConfig, load_dataset_process_config


def test_dataset_config_loader(tmp_path):
    cfg_path = tmp_path / "demo.yaml"
    cfg_path.write_text(
        """
dataset: demo
raw_subdir: demo
sampling_hz: 20
target_hz: 10
smoothing_window: 5
feature_groups:
  identifiers: [dataset, track_id]
  kinematics: [x, y]
  targets: [x, y]
"""
    )
    cfg = load_dataset_process_config("demo", cfg_path.parent)
    assert isinstance(cfg, DatasetProcessConfig)
    assert cfg.dataset == "demo"
    assert cfg.ordered_feature_names == ["dataset", "track_id", "x", "y"]


def test_fill_missing_columns(tmp_path):
    cfg = DatasetProcessConfig(
        dataset="demo",
        raw_subdir="demo",
        sampling_hz=20,
        target_hz=10,
        feature_groups={"identifiers": ["dataset"], "targets": ["x", "y"]},
    )
    df = pd.DataFrame({"dataset": ["demo"]})
    harmonized = cfg.fill_missing_columns(df)
    assert list(harmonized.columns) == ["dataset", "x", "y"]
    assert pd.isna(harmonized["x"]).all()
