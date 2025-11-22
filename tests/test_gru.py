import pytest

torch = pytest.importorskip("torch")
from piuq.models.gru import TrajectoryGRU


def test_gru_forward_shapes():
    model = TrajectoryGRU(input_size=2, hidden_size=4, num_layers=1, future_steps=5)
    history = torch.randn(3, 7, 2)
    out = model(history)
    assert isinstance(out, dict)
    assert out["trajectory_mean"].shape == (3, 5, 2)
    assert out["trajectory_logvar"].shape == (3, 5, 2)
    assert out["risk_logit"].shape == (3, 1)
    assert out["risk_logvar"].shape == (3, 1)
    assert out["intent_logit"].shape == (3, 1)
    assert out["intent_logvar"].shape == (3, 1)
    assert out["epistemic_uncertainty"].shape == (3, 3)
