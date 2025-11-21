import pandas as pd

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
