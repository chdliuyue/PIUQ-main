"""Plot raw trajectories for quick visual validation.
快速可视化原始轨迹以便阶段性验证。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from piuq.visualization.trajectories import plot_scene


def _load_dataframe(path: Path) -> pd.DataFrame:
    """Load a dataframe from CSV or Parquet.
    从 CSV 或 Parquet 中加载 DataFrame。

    """

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _parse_tracks(raw_tracks: Optional[Iterable[str]]) -> Optional[Iterable[int]]:
    """Convert optional track ID strings into integers.
    将可选的轨迹 ID 字符串转换为整数。

    """

    if raw_tracks is None:
        return None
    return [int(t) for t in raw_tracks]


def main() -> None:
    """CLI entry point for rendering raw trajectory scenes.
    用于渲染原始轨迹场景的命令行入口。

    """

    parser = argparse.ArgumentParser(description="Visualize raw trajectories with optional filters")
    parser.add_argument("--input", type=Path, required=True, help="Path to CSV or Parquet containing trajectories")
    parser.add_argument("--recording-id", type=int, help="Optional recording_id filter", default=None)
    parser.add_argument(
        "--tracks",
        nargs="*",
        help="Optional list of track_ids to plot",
        default=None,
    )
    parser.add_argument("--output", type=Path, help="Optional path to save the figure", default=None)
    parser.add_argument("--show", action="store_true", help="Show the plot interactively instead of saving")
    args = parser.parse_args()

    df = _load_dataframe(args.input)
    track_filter = _parse_tracks(args.tracks)

    ax = plot_scene(df, recording_id=args.recording_id, tracks=track_filter, show=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(args.output, bbox_inches="tight")
        print(f"Saved trajectory visualization to {args.output}")
    if args.show or not args.output:
        ax.figure.show()


if __name__ == "__main__":
    main()
