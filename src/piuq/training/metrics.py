from __future__ import annotations

import torch


def ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average displacement error along last dimension.
    沿最后一个维度计算的平均位移误差。

    """

    return torch.norm(pred - target, dim=-1).mean()


def fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Final displacement error.
    最终时刻的位移误差。

    """

    return torch.norm(pred[:, -1] - target[:, -1], dim=-1).mean()


__all__ = ["ade", "fde"]
