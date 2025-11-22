import pandas as pd
import pytest
import numpy as np

from piuq.data.datasets.highd import HighDDataset


def test_standardize_tracks_meta_maps_and_missing_rate():
    dataset = HighDDataset()
    meta_raw = pd.DataFrame(
        {
            "id": [1, 2],
            "initialFrame": [10, 20],
            "finalFrame": [14, 24],
            "numFrames": [5, 3],
            "width": [2.0, 2.5],
            "height": [1.5, 1.6],
            "class": ["Car", "Truck"],
            "traveledDistance": [100.0, 80.0],
            "minDHW": [2.0, 1.5],
            "minTHW": [1.2, 1.0],
            "minTTC": [5.0, 3.5],
        }
    )

    meta = dataset._standardize_tracks_meta(meta_raw)

    expected_cols = {
        "track_id",
        "start_frame",
        "end_frame",
        "track_frames",
        "traveled_distance",
        "min_dhw",
        "min_thw",
        "min_ttc",
        "missing_rate",
        "has_missing",
    }

    assert expected_cols <= set(meta.columns)
    assert meta.loc[0, "missing_rate"] == 0
    assert not bool(meta.loc[0, "has_missing"])
    assert meta.loc[1, "missing_rate"] == pytest.approx(0.4)
    assert bool(meta.loc[1, "has_missing"])


def test_tracks_meta_validation_errors():
    dataset = HighDDataset()
    incomplete_meta = pd.DataFrame(
        {
            "id": [1],
            "width": [2.0],
            "height": [1.5],
            "class": ["Car"],
            "startFrame": [0],
            "endFrame": [10],
        }
    )

    with pytest.raises(ValueError, match="numFrames"):
        dataset._standardize_tracks_meta(incomplete_meta)


def test_lane_center_frenet_offsets_small():
    dataset = HighDDataset(centerline_points=20)
    lane_markings = [
        np.array([[0.0, 0.0], [50.0, 1.0], [100.0, 2.0]]),
        np.array([[0.0, 4.0], [50.0, 5.0], [100.0, 6.0]]),
    ]

    dataset.recording_context[1] = {
        "lane_markings": lane_markings,
        "driving_direction": 1.0,
    }

    sample_x = np.linspace(0.0, 100.0, num=30)
    lower_y = np.interp(sample_x, lane_markings[0][:, 0], lane_markings[0][:, 1])
    upper_y = np.interp(sample_x, lane_markings[1][:, 0], lane_markings[1][:, 1])
    center_y = 0.5 * (lower_y + upper_y)

    df = pd.DataFrame(
        {
            "recording_id": 1,
            "track_id": 1,
            "frame": np.arange(len(sample_x)),
            "x": sample_x,
            "y": center_y,
        }
    )

    frenet_df = dataset.to_frenet(df)
    assert np.max(np.abs(frenet_df["n"])) < 1e-6


def test_tracks_vehicle_type_from_meta_only():
    dataset = HighDDataset()

    tracks_raw = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "frame": [1, 2, 1],
            "x": [0.0, 1.0, 0.5],
            "y": [0.0, 0.2, 0.1],
            "width": [2.0, 2.0, 2.2],
            "height": [1.5, 1.5, 1.6],
            "xVelocity": [1.0, 1.1, 0.9],
            "yVelocity": [0.0, 0.0, 0.0],
            "xAcceleration": [0.0, 0.0, 0.0],
            "yAcceleration": [0.0, 0.0, 0.0],
            "laneId": [1, 1, 1],
        }
    )

    tracks = dataset._standardize_tracks(tracks_raw, frame_rate=25.0)
    assert "vehicle_type" not in tracks.columns

    meta_raw = pd.DataFrame(
        {
            "id": [1, 2],
            "numFrames": [2, 1],
            "width": [2.0, 2.2],
            "height": [1.5, 1.6],
            "class": ["Car", "Truck"],
            "initialFrame": [1, 1],
            "finalFrame": [2, 1],
        }
    )

    meta = dataset._standardize_tracks_meta(meta_raw)
    merged = tracks.merge(meta, on="track_id", how="left")

    assert set(meta["vehicle_type"]) == {"Car", "Truck"}
    assert set(merged["vehicle_type"].dropna()) == {"Car", "Truck"}
