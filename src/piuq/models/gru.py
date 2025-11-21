from __future__ import annotations

import torch
from torch import nn


class TrajectoryGRU(nn.Module):
    """GRU-based predictor with multi-head outputs and uncertainty estimates.
    基于 GRU 的多头预测器，支持轨迹与风险/意图预测并输出不确定性。

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        future_steps: int = 30,
        dropout: float = 0.0,
        risk_dims: int = 1,
        intent_dims: int = 1,
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
        self.traj_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size * 2),
        )
        self.risk_head = nn.Linear(hidden_size, risk_dims)
        self.risk_logvar = nn.Linear(hidden_size, risk_dims)
        self.intent_head = nn.Linear(hidden_size, intent_dims)
        self.intent_logvar = nn.Linear(hidden_size, intent_dims)
        self.epistemic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, history: torch.Tensor) -> dict:  # type: ignore[override]
        """Predict future states and auxiliary targets with uncertainty.
        预测未来 Frenet 状态序列以及风险/意图并输出不确定性。

        Parameters
        ----------
        history: torch.Tensor
            Tensor shaped (batch, seq_len, input_size) containing ordered history
            frames.
            张量形状为 (batch, seq_len, input_size)，包含按时间排序的历史帧。
        Returns
        -------
        dict
            Dictionary with trajectory means/logvars, risk/intent logits &
            log-variances, and epistemic uncertainty estimates.
            包含轨迹均值/对数方差、风险/意图 logits 与对数方差及认知不确定性的字典。
        """

        _, h_n = self.encoder(history)
        last_hidden = h_n[-1]
        traj_means = []
        traj_logvars = []
        hidden = last_hidden.unsqueeze(0)
        step_input = history[:, -1]
        for _ in range(self.future_steps):
            _, hidden = self.encoder(step_input.unsqueeze(1), hidden)
            traj_params = self.traj_head(hidden[-1])
            mean, logvar = traj_params.chunk(2, dim=-1)
            step_input = mean
            traj_means.append(mean)
            traj_logvars.append(logvar)

        traj_mean = torch.stack(traj_means, dim=1)
        traj_logvar = torch.stack(traj_logvars, dim=1)
        risk_logits = self.risk_head(last_hidden)
        risk_logvar = self.risk_logvar(last_hidden)
        intent_logits = self.intent_head(last_hidden)
        intent_logvar = self.intent_logvar(last_hidden)
        epistemic = torch.nn.functional.softplus(self.epistemic_head(last_hidden))

        return {
            "trajectory_mean": traj_mean,
            "trajectory_logvar": traj_logvar,
            "risk_logit": risk_logits,
            "risk_logvar": risk_logvar,
            "intent_logit": intent_logits,
            "intent_logvar": intent_logvar,
            "epistemic_uncertainty": epistemic,
        }


__all__ = ["TrajectoryGRU"]
