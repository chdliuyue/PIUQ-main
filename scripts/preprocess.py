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
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from piuq.config import load_config
from piuq.data.configs import load_dataset_process_config
from piuq.data.datasets.highd import dataset_factory
from piuq.data.pipeline import downsample_tracks, harmonize_features, smooth_positions
from piuq.data.windows import WindowBuilder

ROWS_PER_SHARD = 250_000


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
    data: pd.DataFrame | list | tuple | set | np.ndarray,
    split_key: str,
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list]:
    if isinstance(data, pd.DataFrame):
        if split_key not in data.columns:
            raise KeyError(f"Split key '{split_key}' not found in DataFrame columns")
        unique_keys = sorted(data[split_key].unique())
    else:
        unique_keys = sorted({int(k) for k in data})
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
        raw_partitions = ds.load_raw(
            raw_path, flow_frame_chunk_size=cfg.preprocess.flow_frame_chunk_size
        )

        split_cfg = cfg.preprocess.split
        dataset_dir = args.out if args.out else (processed_dir / ds_name)
        dataset_dir = dataset_dir if dataset_dir.suffix == "" else dataset_dir.parent
        dataset_dir.mkdir(parents=True, exist_ok=True)

        if split_cfg.key != "recording_id":
            raise ValueError("Sharded preprocessing currently supports recording_id splits only")

        splits = deterministic_split(
            list(ds.recording_context.keys()), split_cfg.key, split_cfg.ratios, split_cfg.seed
        )
        split_lookup = {key: split for split, keys in splits.items() for key in keys}
        for split_name in splits:
            (dataset_dir / split_name).mkdir(parents=True, exist_ok=True)

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

        shard_buffers: dict[str, list[pd.DataFrame]] = {
            split: [] for split in splits
        }
        shard_counts = {split: 0 for split in splits}
        shard_indices = {split: 0 for split in splits}
        shard_progress = {
            split: tqdm(desc=f"{split} shards", leave=False, unit="shard") for split in splits
        }
        all_columns: set[str] = set()
        rows_per_shard = ROWS_PER_SHARD
        partition_bar = tqdm(desc=f"Processing {ds_name} partitions", unit="chunk")

        def flush_shard(split_name: str, force: bool = False) -> None:
            if not shard_buffers[split_name]:
                return
            if not force and shard_counts[split_name] < rows_per_shard:
                return
            shard_dir = dataset_dir / split_name
            shard_path = shard_dir / f"part-{shard_indices[split_name]:05d}.parquet"
            shard_df = pd.concat(shard_buffers[split_name], ignore_index=True)
            shard_df.to_parquet(shard_path, index=False)
            shard_buffers[split_name].clear()
            shard_counts[split_name] = 0
            shard_indices[split_name] += 1
            shard_progress[split_name].update(1)

        for raw_df in raw_partitions:
            partition_bar.update(1)
            raw_df = smooth_positions(raw_df, ds_cfg.smoothing_window)
            target_hz = cfg.preprocess.target_hz or ds_cfg.target_hz
            raw_df, _ = downsample_tracks(raw_df, target_hz)
            frenet_df = raw_df if {"s", "n"}.issubset(raw_df.columns) else ds.to_frenet(raw_df)
            frenet_df = harmonize_features(frenet_df, ds_cfg)
            all_columns.update(frenet_df.columns)

            key_values = frenet_df[split_cfg.key].unique()
            if len(key_values) != 1:
                raise ValueError("Expected a single split key value per partition")
            rec_id = int(key_values[0])
            split_name = split_lookup.get(rec_id)
            if split_name is None:
                print(f"[WARN] Recording {rec_id} not assigned to any split; skipping")
                continue

            shard_buffers[split_name].append(frenet_df)
            shard_counts[split_name] += len(frenet_df)
            flush_shard(split_name)

        for split_name in splits:
            flush_shard(split_name, force=True)
            shard_progress[split_name].close()
        partition_bar.close()

        schema_df = pd.DataFrame(columns=sorted(all_columns))
        schema_path = dataset_dir / "feature_schema.json"
        schema = build_feature_schema(schema_df, split_cfg.key)
        schema["configured_groups"] = ds_cfg.feature_groups
        schema["ordered_features"] = ds_cfg.ordered_feature_names
        schema_path.write_text(json.dumps(schema, indent=2))
        print(f"[INFO] Saved feature schema to {schema_path}")

        for split_name, keys in splits.items():
            split_dir = dataset_dir / split_name
            shard_files = sorted(split_dir.glob("*.parquet"))
            if not shard_files:
                print(f"[WARN] Split '{split_name}' is empty; skipping")
                continue

            shard_bar = tqdm(shard_files, desc=f"Windows {split_name}", unit="shard")
            total_windows = 0
            for shard_path in shard_bar:
                shard_df = pd.read_parquet(shard_path)
                shard_windows = builder.build(shard_df)
                total_windows += len(shard_windows)

                windows_out = dataset_dir / f"{split_name}_{shard_path.stem}_windows.pkl"
                with open(windows_out, "wb") as f:
                    pickle.dump(shard_windows, f)

                shard_bar.set_postfix(
                    rows=len(shard_df),
                    shard_windows=len(shard_windows),
                    total_windows=total_windows,
                )
            shard_bar.close()

            print(
                f"[INFO] Saved {total_windows} windows for split '{split_name}' across {len(shard_files)} shards"
            )


if __name__ == "__main__":
    main()
