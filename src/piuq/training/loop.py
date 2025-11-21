from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from piuq.config import TrainingConfig
from piuq.models.physics import physics_residuals

from .metrics import ade, bce_loss, classification_accuracy, fde


def _unpack_batch(batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize different batch formats to tensors.
    兼容不同批次格式，统一转换为张量。
    """

    if isinstance(batch, dict):
        history = batch["history"]
        future = batch["future"]
        risk = batch.get("risk")
        intent = batch.get("intent")
    else:
        history, future, *rest = batch
        risk = rest[0] if len(rest) > 0 else None
        intent = rest[1] if len(rest) > 1 else None

    history = history
    future = future
    batch_size = history.shape[0]
    if risk is None:
        risk = torch.zeros(batch_size, 1, device=history.device, dtype=history.dtype)
    if intent is None:
        intent = torch.zeros(batch_size, 1, device=history.device, dtype=history.dtype)
    return history, future, risk, intent


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    training_cfg: TrainingConfig,
    loss_fn: nn.Module | None = None,
) -> Tuple[float, float, float, float, float, float, float]:
    """Run one training epoch and accumulate loss/metrics.
    运行一个训练周期并累计损失与评估指标。

    """
    model.train()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_risk_loss = 0.0
    total_intent_loss = 0.0
    total_intent_acc = 0.0
    total_risk_acc = 0.0
    loss_fn = loss_fn or nn.L1Loss()

    for batch in loader:
        optimizer.zero_grad()
        history, future, risk, intent = _unpack_batch(batch)
        history = history.to(device)
        future = future.to(device)
        risk = risk.to(device)
        intent = intent.to(device)

        outputs: Dict[str, torch.Tensor] = model(history)
        traj_mean = outputs["trajectory_mean"]
        traj_logvar = outputs["trajectory_logvar"]
        risk_logits = outputs["risk_logit"]
        risk_logvar = outputs["risk_logvar"]
        intent_logits = outputs["intent_logit"]
        intent_logvar = outputs["intent_logvar"]
        epistemic = outputs.get("epistemic_uncertainty")

        data_loss = torch.exp(-traj_logvar) * (future - traj_mean).pow(2) + traj_logvar
        data_loss = data_loss.mean()
        risk_ce = bce_loss(risk_logits, risk)
        intent_ce = bce_loss(intent_logits, intent)

        risk_weighted = torch.exp(-risk_logvar) * risk_ce + risk_logvar.mean()
        intent_weighted = torch.exp(-intent_logvar) * intent_ce + intent_logvar.mean()

        physics_loss = physics_residuals(traj_mean)
        uncertainty_reg = 0.0
        if epistemic is not None:
            uncertainty_reg = epistemic.mean()
        aleatoric_reg = traj_logvar.mean() + risk_logvar.mean() + intent_logvar.mean()
        uncertainty_reg = uncertainty_reg + aleatoric_reg

        loss = (
            data_loss
            + risk_weighted
            + intent_weighted
            + training_cfg.physics_loss_weight * physics_loss
            + training_cfg.uncertainty_loss_weight * uncertainty_reg
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * len(history)
        total_ade += float(ade(traj_mean, future)) * len(history)
        total_fde += float(fde(traj_mean, future)) * len(history)
        total_risk_loss += float(risk_ce.item()) * len(history)
        total_intent_loss += float(intent_ce.item()) * len(history)
        total_risk_acc += float(classification_accuracy(risk_logits, risk)) * len(history)
        total_intent_acc += float(classification_accuracy(intent_logits, intent)) * len(history)

    denom = max(1, len(loader.dataset))
    return (
        total_loss / denom,
        total_ade / denom,
        total_fde / denom,
        total_risk_loss / denom,
        total_intent_loss / denom,
        total_risk_acc / denom,
        total_intent_acc / denom,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    training_cfg: TrainingConfig,
    loss_fn: nn.Module | None = None,
) -> Tuple[float, float, float, float, float, float, float]:
    """Evaluate a model on a validation/test split without gradients.
    在验证或测试集上评估模型且不计算梯度。

    """
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_risk_loss = 0.0
    total_intent_loss = 0.0
    total_risk_acc = 0.0
    total_intent_acc = 0.0
    loss_fn = loss_fn or nn.L1Loss()
    with torch.no_grad():
        for batch in loader:
            history, future, risk, intent = _unpack_batch(batch)
            history = history.to(device)
            future = future.to(device)
            risk = risk.to(device)
            intent = intent.to(device)

            outputs: Dict[str, torch.Tensor] = model(history)
            traj_mean = outputs["trajectory_mean"]
            traj_logvar = outputs["trajectory_logvar"]
            risk_logits = outputs["risk_logit"]
            risk_logvar = outputs["risk_logvar"]
            intent_logits = outputs["intent_logit"]
            intent_logvar = outputs["intent_logvar"]
            epistemic = outputs.get("epistemic_uncertainty")

            data_loss = torch.exp(-traj_logvar) * (future - traj_mean).pow(2) + traj_logvar
            data_loss = data_loss.mean()
            risk_ce = bce_loss(risk_logits, risk)
            intent_ce = bce_loss(intent_logits, intent)

            risk_weighted = torch.exp(-risk_logvar) * risk_ce + risk_logvar.mean()
            intent_weighted = torch.exp(-intent_logvar) * intent_ce + intent_logvar.mean()
            physics_loss = physics_residuals(traj_mean)
            uncertainty_reg = 0.0
            if epistemic is not None:
                uncertainty_reg = epistemic.mean()
            aleatoric_reg = traj_logvar.mean() + risk_logvar.mean() + intent_logvar.mean()
            uncertainty_reg = uncertainty_reg + aleatoric_reg

            loss = (
                data_loss
                + risk_weighted
                + intent_weighted
                + training_cfg.physics_loss_weight * physics_loss
                + training_cfg.uncertainty_loss_weight * uncertainty_reg
            )
            total_loss += float(loss.item()) * len(history)
            total_ade += float(ade(traj_mean, future)) * len(history)
            total_fde += float(fde(traj_mean, future)) * len(history)
            total_risk_loss += float(risk_ce.item()) * len(history)
            total_intent_loss += float(intent_ce.item()) * len(history)
            total_risk_acc += float(classification_accuracy(risk_logits, risk)) * len(history)
            total_intent_acc += float(classification_accuracy(intent_logits, intent)) * len(history)

    denom = max(1, len(loader.dataset))
    return (
        total_loss / denom,
        total_ade / denom,
        total_fde / denom,
        total_risk_loss / denom,
        total_intent_loss / denom,
        total_risk_acc / denom,
        total_intent_acc / denom,
    )


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
        risk = torch.tensor([window.get("risk_label", 0.0)], dtype=torch.float32)
        intent = torch.tensor([window.get("intent_label", 0.0)], dtype=torch.float32)
        physics = torch.tensor(window.get("physics_features", []), dtype=torch.float32)
        if physics.numel() == 0:
            physics = torch.zeros(5, dtype=torch.float32)
        uncertainty = torch.tensor(window.get("uncertainty_features", []), dtype=torch.float32)
        if uncertainty.numel() == 0:
            uncertainty = torch.zeros(4, dtype=torch.float32)
        scene = torch.tensor([window.get("scene_label", 0.0)], dtype=torch.float32)
        neighbor_mask = torch.ones(len(window.get("neighbors", [])), dtype=torch.bool)
        return {
            "history": history,
            "future": future,
            "risk": risk,
            "intent": intent,
            "physics": physics,
            "uncertainty": uncertainty,
            "scene": scene,
            "neighbor_mask": neighbor_mask,
        }

    tensorized = [to_tensor(w) for w in windows]

    def collate_fn(batch: Iterable[dict]):
        histories = [b["history"] for b in batch]
        futures = [b["future"] for b in batch]
        risks = torch.stack([b["risk"] for b in batch])
        intents = torch.stack([b["intent"] for b in batch])
        physics = torch.stack([b["physics"] for b in batch])
        uncertainty = torch.stack([b["uncertainty"] for b in batch])
        scenes = torch.stack([b["scene"] for b in batch])
        max_neighbors = max((m.shape[0] for m in [b["neighbor_mask"] for b in batch]), default=0)
        padded_masks = []
        for b in batch:
            mask = b["neighbor_mask"]
            pad_len = max_neighbors - mask.shape[0]
            if pad_len > 0:
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.bool)])
            padded_masks.append(mask)
        neighbor_masks = torch.stack(padded_masks) if padded_masks else torch.empty((0, 0), dtype=torch.bool)

        return {
            "history": torch.nn.utils.rnn.pad_sequence(histories, batch_first=True),
            "future": torch.nn.utils.rnn.pad_sequence(futures, batch_first=True),
            "risk": risks,
            "intent": intents,
            "physics": physics,
            "uncertainty": uncertainty,
            "scene": scenes,
            "neighbor_mask": neighbor_masks,
        }

    return DataLoader(tensorized, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


__all__ = ["train_epoch", "evaluate", "window_tensor_loader"]
