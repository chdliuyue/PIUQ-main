"""Physics-inspired penalties for trajectory regularization."""

from .residuals import compute_jerk, compute_ttc_residual, physics_residuals

__all__ = ["compute_jerk", "compute_ttc_residual", "physics_residuals"]