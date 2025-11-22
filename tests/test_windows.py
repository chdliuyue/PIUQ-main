import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import pytest

from piuq.data.windows import WindowBuilder


def test_window_builder_minimal():
    data = {
        "dataset": ["d"] * 5,
        "recording_id": [0] * 5,
        "track_id": [1, 1, 1, 1, 1],
        "frame": [0, 1, 2, 3, 4],
        "t": [0.0, 0.1, 0.2, 0.3, 0.4],
        "s": [0, 1, 2, 3, 4],
        "n": [0, 0, 0, 0, 0],
    }
    df = pd.DataFrame(data)
    builder = WindowBuilder(history_sec=0.2, future_sec=0.1, step_sec=0.1, neighbor_radius_s=10, max_neighbors=4)
    windows = builder.build(df)
    assert len(windows) == 2
    assert all("history" in w and "future" in w for w in windows)


def test_neighbor_filter_and_ttc_risk_label():
    records = []
    for frame in range(5):
        records.append(
            {
                "dataset": "d",
                "recording_id": 0,
                "track_id": 1,
                "frame": frame,
                "t": frame * 0.1,
                "s": float(frame),
                "n": 0.0,
                "driving_direction": 1,
                "ttc": 10.0 if frame < 3 else 6.0,
            }
        )
        records.append(
            {
                "dataset": "d",
                "recording_id": 0,
                "track_id": 2,
                "frame": frame,
                "t": frame * 0.1,
                "s": float(frame) + 0.5,
                "n": 0.0,
                "driving_direction": 2,
                "ttc": np.nan,
            }
        )
    df = pd.DataFrame.from_records(records)
    builder = WindowBuilder(
        history_sec=0.2,
        future_sec=0.2,
        step_sec=0.1,
        neighbor_radius_s=5,
        max_neighbors=4,
    )
    windows = builder.build(df)

    ego_windows = [w for w in windows if w["ego_track_id"] == 1]
    assert len(ego_windows) == 1
    window = ego_windows[0]
    assert window["neighbors"].empty
    assert window["risk_label"] == 0
    assert window["ttc_min_future"] == pytest.approx(6.0)


def test_ttc_min_future_risk_tiers():
    data = {
        "dataset": ["d"] * 5,
        "recording_id": [0] * 5,
        "track_id": [1] * 5,
        "frame": [0, 1, 2, 3, 4],
        "t": [0.0, 0.1, 0.2, 0.3, 0.4],
        "s": [0, 1, 2, 3, 4],
        "n": [0, 0, 0, 0, 0],
        "driving_direction": [1] * 5,
        "ttc": [np.nan, np.inf, 4.0, 2.0, 0.5],
    }
    df = pd.DataFrame(data)
    builder = WindowBuilder(
        history_sec=0.2,
        future_sec=0.2,
        step_sec=0.1,
        neighbor_radius_s=10,
        max_neighbors=4,
    )
    windows = builder.build(df)

    assert len(windows) == 1
    window = windows[0]
    assert window["ttc_min_future"] == pytest.approx(0.5)
    assert window["risk_label"] == 3


def test_physics_residual_aggregation():
    df = pd.DataFrame(
        {
            "dataset": ["d"] * 5,
            "recording_id": [0] * 5,
            "track_id": [1] * 5,
            "frame": [0, 1, 2, 3, 4],
            "t": [0.0, 1.0, 2.0, 3.0, 4.0],
            "s": [0.0, 1.0, 3.0, 6.5, 12.5],
            "n": [0.0] * 5,
        }
    )

    mean_builder = WindowBuilder(
        history_sec=2.0,
        future_sec=1.0,
        step_sec=1.0,
        neighbor_radius_s=5,
        max_neighbors=2,
        physics_residual_aggregation="mean_abs",
    )
    max_builder = WindowBuilder(
        history_sec=2.0,
        future_sec=1.0,
        step_sec=1.0,
        neighbor_radius_s=5,
        max_neighbors=2,
        physics_residual_aggregation="max_abs",
    )

    mean_window = mean_builder.build(df)[0]
    max_window = max_builder.build(df)[0]

    hist_df = mean_window["history"]
    dt = float(hist_df["t"].diff().dropna().median())
    positions = hist_df[["s", "n"]].to_numpy()
    velocity = np.gradient(positions, dt, axis=0)
    longitudinal_residual = np.gradient(velocity[:, 0], dt)
    expected_mean = float(np.mean(np.abs(longitudinal_residual)))
    expected_max = float(np.max(np.abs(longitudinal_residual)))

    mean_residual = mean_window["physics_features"][4]
    max_residual = max_window["physics_features"][4]

    assert mean_residual == pytest.approx(expected_mean)
    assert max_residual == pytest.approx(expected_max)
