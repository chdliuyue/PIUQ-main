"""Encoder stubs for future trajectory/backbone models."""

from __future__ import annotations

from typing import Any


class IdentityEncoder:
    """Placeholder encoder that returns input unchanged.

    This avoids hard dependencies on deep learning frameworks until
    training code is wired up.
    """

    def __call__(self, x: Any) -> Any:  # pragma: no cover - trivial
        return x


__all__ = ["IdentityEncoder"]
