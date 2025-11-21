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


def bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy with logits."""

    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target)


def classification_accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute accuracy from logits and binary targets."""

    preds = (logits.sigmoid() > 0.5).float()
    return (preds == target).float().mean()


__all__ = ["ade", "fde", "bce_loss", "classification_accuracy"]
