#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from piuq.config import load_config
from piuq.data.datasets.highd import dataset_factory
from piuq.data.geometry.frenet import FrenetFrame


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _iter_parquets(processed_dir: Path) -> Iterator[pd.DataFrame]:
    for path in sorted(processed_dir.glob("*.parquet")):
        yield pd.read_parquet(path)


def _load_windows(processed_dir: Path) -> list[dict]:
    windows: list[dict] = []
    for path in sorted(processed_dir.glob("*_windows.pkl")):
        with open(path, "rb") as f:
            windows.extend(pickle.load(f))
    return windows


def _reconstruct_xy(s: np.ndarray, n: np.ndarray, frame: FrenetFrame) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    n = np.asarray(n, dtype=float)
    idx = np.searchsorted(frame.s_seg, s, side="right") - 1
    idx = np.clip(idx, 0, len(frame.seg_len) - 1)
    along = np.clip(s - frame.s_seg[idx], 0.0, frame.seg_len[idx])
    xy = frame.centerline[idx] + along[:, None] * frame.tangents[idx]
    xy = xy + n[:, None] * frame.normals[idx]
    return xy


def _check_sampling_interval(df: pd.DataFrame, tolerance: float) -> CheckResult:
    if not {"track_id", "t"}.issubset(df.columns):
        return CheckResult(
            "Sampling interval",
            False,
            "Missing required columns track_id/t",
        )

    dt_series = df.groupby("track_id")["t"].diff().dropna()
    if dt_series.empty:
        return CheckResult("Sampling interval", False, "No deltas to check")

    median_dt = float(dt_series.median())
    non_positive = int((dt_series <= 0).sum())
    rel_dev = np.abs(dt_series - median_dt) / median_dt if median_dt else np.inf
    max_dev = float(rel_dev.max()) if len(rel_dev) else 0.0
    passed = non_positive == 0 and max_dev <= tolerance
    detail = (
        f"median Δt={median_dt:.4f}, max deviation={max_dev:.3%}, "
        f"non-positive steps={non_positive}"
    )
    return CheckResult("Sampling interval", passed, detail)


def _check_monotonic_frames(df: pd.DataFrame) -> CheckResult:
    if not {"track_id", "frame"}.issubset(df.columns):
        return CheckResult("Monotonic frames", False, "Missing track_id/frame columns")

    broken = 0
    for _, g in df.groupby("track_id"):
        frames = g["frame"].to_numpy()
        if np.any(np.diff(frames) <= 0):
            broken += 1
    passed = broken == 0
    detail = f"tracks with non-increasing frames: {broken}"
    return CheckResult("Monotonic frames", passed, detail)


def _check_frenet_reconstruction(df: pd.DataFrame, threshold: float) -> CheckResult:
    required = {"dataset", "recording_id", "x", "y", "s", "n"}
    if not required.issubset(df.columns):
        return CheckResult(
            "Frenet reconstruction",
            False,
            f"Missing columns: {sorted(required - set(df.columns))}",
        )

    errors: list[float] = []
    for (dataset_name, rec_id), g in df.groupby(["dataset", "recording_id"]):
        try:
            dataset = dataset_factory(str(dataset_name))
        except Exception as e:  # pragma: no cover - defensive
            return CheckResult("Frenet reconstruction", False, f"Dataset error: {e}")

        centerline = dataset.build_centerline(g)
        frenet = FrenetFrame(centerline)
        recon = _reconstruct_xy(g["s"].to_numpy(), g["n"].to_numpy(), frenet)
        xy = g[["x", "y"]].to_numpy()
        err = np.sqrt(np.sum((recon - xy) ** 2, axis=1))
        errors.extend(err.tolist())

    rms = float(np.sqrt(np.mean(np.square(errors)))) if errors else np.inf
    passed = bool(errors) and rms <= threshold
    detail = f"RMS Cartesian error={rms:.3f} m (threshold {threshold:.3f})"
    return CheckResult("Frenet reconstruction", passed, detail)


def _check_physics_bounds(windows: Iterable[dict], drac_max: float) -> CheckResult:
    ttcs: list[float] = []
    dracs: list[float] = []
    for w in windows:
        physics = w.get("physics_features")
        if physics is None or len(physics) < 3:
            continue
        ttcs.append(float(physics[0]))
        dracs.append(float(physics[2]))

    if not ttcs:
        return CheckResult("Physics bounds", False, "No physics_features found")

    ttcs = np.asarray(ttcs, dtype=float)
    dracs = np.asarray(dracs, dtype=float)
    bad_ttc = int(np.sum(~np.isfinite(ttcs) & (ttcs < 0)))
    bad_drac = int(np.sum((dracs < 0) | (dracs > drac_max)))
    passed = bad_ttc == 0 and bad_drac == 0
    detail = (
        f"ttc[min={np.nanmin(ttcs):.3f}, max={np.nanmax(ttcs):.3f}], "
        f"drac[min={np.nanmin(dracs):.3f}, max={np.nanmax(dracs):.3f}], "
        f"out-of-bounds ttc={bad_ttc}, drac={bad_drac}"
    )
    return CheckResult("Physics bounds", passed, detail)


def _check_neighbors(
    windows: Iterable[dict],
    df: pd.DataFrame,
    neighbor_radius: float,
    max_neighbors: int,
) -> CheckResult:
    required_cols = {"dataset", "recording_id", "frame", "track_id", "s"}
    if not required_cols.issubset(df.columns):
        return CheckResult(
            "Neighbor completeness",
            False,
            f"Missing columns: {sorted(required_cols - set(df.columns))}",
        )

    lookup = {}
    for (ds, rec, frame), g in df.groupby(["dataset", "recording_id", "frame"]):
        lookup[(ds, rec, int(frame))] = g

    missing = 0
    checked = 0
    for w in windows:
        ds = w.get("dataset")
        rec = w.get("recording_id")
        center_frame = int(w.get("center_frame"))
        hist = w.get("history")
        if hist is None or len(hist) == 0:
            continue
        ego_s = float(hist["s"].iloc[-1]) if isinstance(hist, pd.DataFrame) else None
        if ego_s is None:
            continue

        frame_df = lookup.get((ds, rec, center_frame))
        if frame_df is None:
            continue

        peers = frame_df[frame_df["track_id"] != w.get("ego_track_id")].copy()
        peers["ds"] = np.abs(peers["s"] - ego_s)
        peers = peers[peers["ds"] <= neighbor_radius]
        expected_ids = (
            peers.sort_values("ds")["track_id"].astype(int).tolist()[:max_neighbors]
        )

        neighbors_df = w.get("neighbors")
        neighbor_ids = []
        if isinstance(neighbors_df, pd.DataFrame):
            neighbor_ids = neighbors_df["track_id"].astype(int).tolist()

        if expected_ids != neighbor_ids:
            missing += 1
        checked += 1

    if checked == 0:
        return CheckResult("Neighbor completeness", False, "No windows to validate")

    passed = missing == 0
    detail = f"neighbor mismatches: {missing} / {checked}"
    return CheckResult("Neighbor completeness", passed, detail)


def run_checks(
    processed_dir: Path,
    dt_tolerance: float,
    recon_threshold: float,
    drac_max: float,
    config_path: Path,
    overrides: list[Path] | None,
) -> list[CheckResult]:
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    cfg = load_config(config_path, overrides)
    neighbor_radius = float(cfg.preprocess.neighbor_radius_s)
    max_neighbors = int(cfg.preprocess.max_neighbors)

    frames = list(_iter_parquets(processed_dir))
    if not frames:
        raise FileNotFoundError(f"No parquet files found under {processed_dir}")
    df = pd.concat(frames, ignore_index=True)

    windows = _load_windows(processed_dir)
    if not windows:
        raise FileNotFoundError(f"No *_windows.pkl files found under {processed_dir}")

    return [
        _check_sampling_interval(df, dt_tolerance),
        _check_monotonic_frames(df),
        _check_frenet_reconstruction(df, recon_threshold),
        _check_physics_bounds(windows, drac_max),
        _check_neighbors(windows, df, neighbor_radius, max_neighbors),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate processed trajectory outputs")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="Directory containing split parquet files and *_windows.pkl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Config used for preprocessing (for neighbor thresholds)",
    )
    parser.add_argument(
        "--config-overrides",
        nargs="*",
        type=Path,
        default=None,
        help="Optional override YAMLs applied on top of --config",
    )
    parser.add_argument(
        "--dt-tolerance",
        type=float,
        default=0.05,
        help="Allowed relative deviation of sampling interval (e.g., 0.05 = 5%)",
    )
    parser.add_argument(
        "--recon-threshold",
        type=float,
        default=0.5,
        help="Maximum RMS Frenet->Cartesian reconstruction error in meters",
    )
    parser.add_argument(
        "--drac-max",
        type=float,
        default=30.0,
        help="Maximum allowed DRAC value (m/s^2) in physics_features",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_checks(
        processed_dir=args.processed_dir,
        dt_tolerance=args.dt_tolerance,
        recon_threshold=args.recon_threshold,
        drac_max=args.drac_max,
        config_path=args.config,
        overrides=args.config_overrides,
    )

    print("================ Self-check report ================")
    failures = 0
    for res in results:
        status = "PASS" if res.passed else "FAIL"
        print(f"[{status}] {res.name}: {res.detail}")
        if not res.passed:
            failures += 1

    if failures:
        raise SystemExit(f"Self-check failed with {failures} issue(s)")


if __name__ == "__main__":
    main()
