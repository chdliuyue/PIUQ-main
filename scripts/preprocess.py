#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from piuq.config import load_config
from piuq.data.datasets.highd import dataset_factory
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
        help="Override processed output path (Parquet)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.config_overrides)

    datasets = [args.dataset] if args.dataset else cfg.preprocess.datasets
    processed_dir = Path(cfg.paths.processed_data)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in datasets:
        ds = dataset_factory(ds_name)
        raw_path = Path(cfg.paths.raw_data) / ds_name
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data path not found: {raw_path}")

        print(f"[INFO] Loading raw {ds_name} from {raw_path}")
        raw_df = ds.load_raw(raw_path)
        print(f"[INFO] Loaded {len(raw_df)} rows")

        frenet_df = ds.to_frenet(raw_df)
        out_file = args.out or (processed_dir / f"{ds_name}_frenet.parquet")
        frenet_df.to_parquet(out_file, index=False)
        print(f"[INFO] Saved Frenet trajectories to {out_file}")

        builder = WindowBuilder(
            history_sec=cfg.preprocess.history_sec,
            future_sec=cfg.preprocess.future_sec,
            step_sec=cfg.windows.step_sec,
            neighbor_radius_s=cfg.preprocess.neighbor_radius_s,
            max_neighbors=cfg.preprocess.max_neighbors,
            allow_gaps=cfg.preprocess.allow_gaps,
        )
        windows = builder.build(frenet_df)
        windows_out = out_file.with_suffix(".windows.pkl")
        with open(windows_out, "wb") as f:
            pickle.dump(windows, f)
        print(f"[INFO] Saved {len(windows)} windows to {windows_out}")


if __name__ == "__main__":
    main()
