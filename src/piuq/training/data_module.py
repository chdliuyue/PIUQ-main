"""Dataset tensorization placeholder.
数据集张量化的占位模块。

This module will convert window dictionaries into tensors suitable for
future models. For now it only defines the interface.
该模块将把窗口字典转换为适用于未来模型的张量，目前仅定义接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np


@dataclass
class WindowTensor:
    history: np.ndarray
    future: np.ndarray
    neighbor_mask: np.ndarray
    physics: np.ndarray
    uncertainty: np.ndarray
    risk: np.ndarray
    scene: np.ndarray


def collate_windows(windows: Iterable[Dict[str, Any]], pad_value: float = 0.0) -> WindowTensor:
    """Pad variable-length windows into aligned numpy arrays for batching.
    将不同长度的窗口填充为对齐的 numpy 数组以便批处理。

    """
    history_list: List[np.ndarray] = []
    future_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    physics_list: List[np.ndarray] = []
    uncertainty_list: List[np.ndarray] = []
    risk_list: List[np.ndarray] = []
    scene_list: List[np.ndarray] = []

    for w in windows:
        hist = w["history"][['s','n']].to_numpy()
        fut = w["future"][['s','n']].to_numpy()
        neighbors = w["neighbors"]
        mask = np.ones(len(neighbors), dtype=bool)
        history_list.append(hist)
        future_list.append(fut)
        mask_list.append(mask)
        physics_list.append(np.asarray(w.get("physics_features", []), dtype=float))
        uncertainty_list.append(np.asarray(w.get("uncertainty_features", []), dtype=float))
        risk_list.append(np.asarray([w.get("risk_label", 0.0)], dtype=float))
        scene_list.append(np.asarray([w.get("scene_label", 0.0)], dtype=float))

    max_hist = max(arr.shape[0] for arr in history_list)
    max_fut = max(arr.shape[0] for arr in future_list)
    max_neighbors = max(mask.shape[0] for mask in mask_list) if mask_list else 0

    def pad(seq_list: List[np.ndarray], target_len: int) -> np.ndarray:
        padded = []
        for arr in seq_list:
            pad_len = target_len - arr.shape[0]
            if pad_len > 0:
                pad_block = np.full((pad_len, arr.shape[1]), pad_value)
                arr = np.vstack([arr, pad_block])
            padded.append(arr)
        return np.stack(padded)

    history_tensor = pad(history_list, max_hist)
    future_tensor = pad(future_list, max_fut)

    mask_padded = []
    for mask in mask_list:
        pad_len = max_neighbors - mask.shape[0]
        if pad_len > 0:
            mask = np.concatenate([mask, np.zeros(pad_len, dtype=bool)])
        mask_padded.append(mask)
    neighbor_mask = np.stack(mask_padded) if mask_padded else np.empty((0, 0))

    physics_tensor = np.stack(physics_list) if physics_list else np.empty((0, 0))
    uncertainty_tensor = np.stack(uncertainty_list) if uncertainty_list else np.empty((0, 0))
    risk_tensor = np.stack(risk_list) if risk_list else np.empty((0, 1))
    scene_tensor = np.stack(scene_list) if scene_list else np.empty((0, 1))

    return WindowTensor(
        history=history_tensor,
        future=future_tensor,
        neighbor_mask=neighbor_mask,
        physics=physics_tensor,
        uncertainty=uncertainty_tensor,
        risk=risk_tensor,
        scene=scene_tensor,
    )


__all__ = ["WindowTensor", "collate_windows"]
