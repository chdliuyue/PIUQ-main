from __future__ import annotations

import torch


def compute_jerk(traj: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """Finite-difference jerk magnitude for a trajectory.
    对轨迹使用有限差分计算加加速度（jerk）。
    """

    velocity = (traj[:, 1:] - traj[:, :-1]) / dt
    acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt
    jerk = (acceleration[:, 1:] - acceleration[:, :-1]) / dt
    return jerk.norm(dim=-1)


def compute_ttc_residual(traj: torch.Tensor, dt: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """Encourage time-to-collision (TTC) to decrease smoothly toward the horizon.
    约束时间到碰撞（TTC）朝着时间地平线平滑递减。
    """

    displacements = traj[:, 1:] - traj[:, :-1]
    speed = displacements.norm(dim=-1) / dt + eps
    remaining = (traj[:, -1].unsqueeze(1) - traj[:, :-1]).norm(dim=-1)
    ttc = remaining / speed
    # Ideal TTC drops roughly by dt each step; penalize deviations.
    target_drop = torch.full_like(ttc[:, 1:], dt)
    return torch.abs((ttc[:, :-1] - ttc[:, 1:]) - target_drop)


def physics_residuals(traj: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """Aggregate physics residual penalties (jerk and TTC conservation).
    汇总物理残差惩罚（jerk 与 TTC 约束）。
    """

    jerk_penalty = compute_jerk(traj, dt).mean()
    ttc_penalty = compute_ttc_residual(traj, dt).mean()
    return jerk_penalty + ttc_penalty


__all__ = ["compute_jerk", "compute_ttc_residual", "physics_residuals"]
