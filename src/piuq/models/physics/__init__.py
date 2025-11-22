"""Physics-inspired penalties for trajectory regularization.
用于轨迹正则化的物理启发式惩罚项。
"""

from .residuals import compute_jerk, compute_ttc_residual, physics_residuals

__all__ = ["compute_jerk", "compute_ttc_residual", "physics_residuals"]
