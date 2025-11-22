"""Encoder stubs for future trajectory/backbone models.
为未来的轨迹与骨干网络模型预留的编码器桩件。
"""

from __future__ import annotations

from typing import Any


class IdentityEncoder:
    """Placeholder encoder that returns input unchanged.
    返回输入本身的占位编码器。

    This avoids hard dependencies on deep learning frameworks until
    training code is wired up.
    在训练代码接入前，避免引入对深度学习框架的强耦合依赖。
    """

    def __call__(self, x: Any) -> Any:  # pragma: no cover - trivial
        return x


__all__ = ["IdentityEncoder"]
