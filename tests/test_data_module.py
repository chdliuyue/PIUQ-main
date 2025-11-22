import numpy as np
import pandas as pd

from piuq.training.data_module import collate_windows


def test_collate_respects_feature_dimensions():
    windows = [
        {
            "history": pd.DataFrame({"s": [0.0, 1.0], "n": [0.0, 0.5], "speed": [1.0, 2.0]}),
            "future": pd.DataFrame({"s": [1.5, 2.5], "n": [0.6, 0.7], "speed": [2.5, 3.5]}),
            "neighbors": pd.DataFrame(),
            "physics_features": [0.1, 0.2, 0.3],
            "uncertainty_features": [0.4, 0.5],
            "risk_label": 1,
            "scene_label": 2,
        }
    ]

    tensor = collate_windows(
        windows,
        pad_value=-1.0,
        history_features=("s", "speed"),
        future_features=("n",),
        physics_dim=5,
        uncertainty_dim=3,
    )

    assert tensor.history.shape == (1, 2, 2)
    assert np.allclose(tensor.history[0, 0], [0.0, 1.0])

    assert tensor.future.shape == (1, 2, 1)
    assert np.allclose(tensor.future[0, :, 0], [0.6, 0.7])

    assert tensor.physics.shape == (1, 5)
    assert np.allclose(tensor.physics[0, :3], [0.1, 0.2, 0.3])
    assert np.all(tensor.physics[0, 3:] == -1.0)

    assert tensor.uncertainty.shape == (1, 3)
    assert np.allclose(tensor.uncertainty[0, :2], [0.4, 0.5])
    assert tensor.uncertainty[0, 2] == -1.0

    assert tensor.neighbor_mask.size == 0
    assert np.allclose(tensor.risk, [[1.0]])
    assert np.allclose(tensor.scene, [[2.0]])
