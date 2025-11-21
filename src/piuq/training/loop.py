from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import ade, fde


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module | None = None,
) -> Tuple[float, float, float]:
    """Run one training epoch and accumulate loss/metrics.
    运行一个训练周期并累计损失与评估指标。

    """
    model.train()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    loss_fn = loss_fn or nn.L1Loss()

    for batch in loader:
        optimizer.zero_grad()
        history, future = batch
        history = history.to(device)
        future = future.to(device)
        pred = model(history)
        loss = loss_fn(pred, future)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * len(history)
        total_ade += float(ade(pred, future)) * len(history)
        total_fde += float(fde(pred, future)) * len(history)

    denom = max(1, len(loader.dataset))
    return total_loss / denom, total_ade / denom, total_fde / denom


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module | None = None,
) -> Tuple[float, float, float]:
    """Evaluate a model on a validation/test split without gradients.
    在验证或测试集上评估模型且不计算梯度。

    """
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    loss_fn = loss_fn or nn.L1Loss()
    with torch.no_grad():
        for batch in loader:
            history, future = batch
            history = history.to(device)
            future = future.to(device)
            pred = model(history)
            loss = loss_fn(pred, future)
            total_loss += float(loss.item()) * len(history)
            total_ade += float(ade(pred, future)) * len(history)
            total_fde += float(fde(pred, future)) * len(history)

    denom = max(1, len(loader.dataset))
    return total_loss / denom, total_ade / denom, total_fde / denom


def window_tensor_loader(windows: Iterable, batch_size: int = 8, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader from window dictionaries.
    基于窗口字典创建 DataLoader。

    Expects each window to contain "history" and "future" DataFrames with columns (s, n).
    期望每个窗口包含带有 (s, n) 列的 "history" 与 "future" DataFrame。
    """

    def to_tensor(window):
        hist = torch.tensor(window["history"]["s"].to_numpy(), dtype=torch.float32)
        hist_n = torch.tensor(window["history"]["n"].to_numpy(), dtype=torch.float32)
        fut = torch.tensor(window["future"]["s"].to_numpy(), dtype=torch.float32)
        fut_n = torch.tensor(window["future"]["n"].to_numpy(), dtype=torch.float32)
        history = torch.stack([hist, hist_n], dim=-1)
        future = torch.stack([fut, fut_n], dim=-1)
        return history, future

    tensorized = [to_tensor(w) for w in windows]
    histories, futures = zip(*tensorized)
    dataset = torch.utils.data.TensorDataset(
        torch.nn.utils.rnn.pad_sequence(histories, batch_first=True),
        torch.nn.utils.rnn.pad_sequence(futures, batch_first=True),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


__all__ = ["train_epoch", "evaluate", "window_tensor_loader"]
