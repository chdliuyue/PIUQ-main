from .encoders import IdentityEncoder

__all__ = ["IdentityEncoder", "TrajectoryGRU"]


def __getattr__(name):  # pragma: no cover - lazy import
    if name == "TrajectoryGRU":
        from .gru import TrajectoryGRU

        return TrajectoryGRU
    raise AttributeError(name)
