#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from piuq.config import load_config
from piuq.data.configs import load_dataset_process_config
from piuq.data.datasets.highd import dataset_factory
from piuq.data.pipeline import downsample_tracks, harmonize_features, smooth_positions
from piuq.data.windows import WindowBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess UAV datasets to Frenet windows")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Base YAML config file",
    )
    parser.add_argument(
        "--config-overrides",
        type=Path,
        nargs="*",
        default=None,
        help="Additional YAML configs that override the base",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset name (otherwise taken from config.preprocess.datasets)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override processed output directory (default under cfg.paths.processed_data)",
    )
    return parser.parse_args()


def deterministic_split(
    df: pd.DataFrame, split_key: str, ratios: dict[str, float], seed: int
) -> dict[str, list]:
    if split_key not in df.columns:
        raise KeyError(f"Split key '{split_key}' not found in DataFrame columns")

    unique_keys = sorted(df[split_key].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_keys)

    total_ratio = float(sum(ratios.values()))
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive value")

    normalized = OrderedDict((k, v / total_ratio) for k, v in ratios.items())
    split_sizes = []
    num_keys = len(unique_keys)
    for i, (_, ratio) in enumerate(normalized.items()):
        if i < len(normalized) - 1:
            size = int(np.floor(ratio * num_keys))
        else:
            size = num_keys - sum(split_sizes)
        split_sizes.append(size)

    splits: dict[str, list] = {}
    cursor = 0
    for (split_name, _), size in zip(normalized.items(), split_sizes):
        splits[split_name] = unique_keys[cursor : cursor + size]
        cursor += size

    return splits


def build_feature_schema(df: pd.DataFrame, split_key: str) -> dict:
    vehicle_fields = [
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
        "speed_limit",
        "frame_rate",
        "recording_location",
        "s",
        "n",
        "v_s",
        "v_n",
        "a_s",
        "a_n",
    ]
    flow_fields = [
        "density",
        "frame_mean_speed",
        "frame_headway_mean",
        "frame_headway_median",
    ]
    physics_fields = ["dhw", "thw", "ttc"]
    uncertainty_fields = [
        "vx_var",
        "vy_var",
        "ax_var",
        "ay_var",
        "speed_var",
        "missing_rate",
        "has_missing",
    ]

    def present(fields: list[str]) -> list[str]:
        return sorted([f for f in fields if f in df.columns])

    return {
        "split_key": split_key,
        "vehicle": present(vehicle_fields),
        "flow": present(flow_fields),
        "physics": present(physics_fields),
        "uncertainty": present(uncertainty_fields),
        "all_columns": sorted(df.columns),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.config_overrides)

    np.random.seed(cfg.preprocess.seed)

    datasets = [args.dataset] if args.dataset else cfg.preprocess.datasets
    processed_dir = Path(cfg.paths.processed_data)

    for ds_name in datasets:
        ds_cfg = load_dataset_process_config(ds_name, cfg.preprocess.dataset_config_dir)
        ds = dataset_factory(ds_name)
        raw_path = Path(cfg.paths.raw_data) / ds_cfg.raw_subdir
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data path not found: {raw_path}")

        print(f"[INFO] Loading raw {ds_name} from {raw_path}")
        raw_df = ds.load_raw(raw_path)
        print(f"[INFO] Loaded {len(raw_df)} rows")

        raw_df = smooth_positions(raw_df, ds_cfg.smoothing_window)
        target_hz = cfg.preprocess.target_hz or ds_cfg.target_hz
        raw_df, _ = downsample_tracks(raw_df, target_hz)
        frenet_df = ds.to_frenet(raw_df)
        frenet_df = harmonize_features(frenet_df, ds_cfg)

        split_cfg = cfg.preprocess.split
        dataset_dir = args.out if args.out else (processed_dir / ds_name)
        dataset_dir = dataset_dir if dataset_dir.suffix == "" else dataset_dir.parent
        dataset_dir.mkdir(parents=True, exist_ok=True)

        splits = deterministic_split(
            frenet_df, split_cfg.key, split_cfg.ratios, split_cfg.seed
        )

        schema_path = dataset_dir / "feature_schema.json"
        schema = build_feature_schema(frenet_df, split_cfg.key)
        schema["configured_groups"] = ds_cfg.feature_groups
        schema["ordered_features"] = ds_cfg.ordered_feature_names
        schema_path.write_text(json.dumps(schema, indent=2))
        print(f"[INFO] Saved feature schema to {schema_path}")

        builder = WindowBuilder(
            history_sec=cfg.preprocess.history_sec,
            future_sec=cfg.preprocess.future_sec,
            step_sec=cfg.windows.step_sec,
            neighbor_radius_s=cfg.preprocess.neighbor_radius_s,
            max_neighbors=cfg.preprocess.max_neighbors,
            allow_gaps=cfg.preprocess.allow_gaps,
            risk_ttc_thresholds=cfg.windows.risk_ttc_thresholds,
            physics_residual_aggregation=cfg.windows.physics_residual_aggregation,
        )
        for split_name, keys in splits.items():
            split_df = frenet_df[frenet_df[split_cfg.key].isin(keys)].copy()
            if split_df.empty:
                print(f"[WARN] Split '{split_name}' is empty; skipping")
                continue

            split_path = dataset_dir / f"{split_name}.parquet"
            split_df.to_parquet(split_path, index=False)
            print(
                f"[INFO] Saved {len(split_df)} rows for split '{split_name}' to {split_path}"
            )

            windows = builder.build(split_df)
            windows_out = dataset_dir / f"{split_name}_windows.pkl"
            with open(windows_out, "wb") as f:
                pickle.dump(windows, f)
            print(
                f"[INFO] Saved {len(windows)} windows for split '{split_name}' to {windows_out}"
            )


if __name__ == "__main__":
    main()
