import pandas as pd
import pytest

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
