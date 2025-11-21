from __future__ import annotations

import torch
from torch import nn


class TrajectoryGRU(nn.Module):
    """Minimal GRU-based predictor for future Frenet states.
    基于 GRU 的简易预测器，用于未来 Frenet 状态预测。

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        future_steps: int = 30,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Predict future states.
        预测未来的 Frenet 状态序列。

        Parameters
        ----------
        history: torch.Tensor
            Tensor shaped (batch, seq_len, input_size) containing ordered history
            frames.
            张量形状为 (batch, seq_len, input_size)，包含按时间排序的历史帧。
        Returns
        -------
        torch.Tensor
            Predicted future sequence with shape (batch, future_steps, input_size).
            返回形状为 (batch, future_steps, input_size) 的未来序列预测。
        """

        _, h_n = self.encoder(history)
        last_hidden = h_n[-1]
        preds = []
        hidden = last_hidden.unsqueeze(0)
        step_input = history[:, -1]
        for _ in range(self.future_steps):
            _, hidden = self.encoder(step_input.unsqueeze(1), hidden)
            step_input = self.head(hidden[-1])
            preds.append(step_input)
        return torch.stack(preds, dim=1)


__all__ = ["TrajectoryGRU"]
