# Validation guide / 自检指南

This document outlines small, repeatable checks you can run after code changes. Each section pairs an English step with a Chinese translation.
本文档列出代码更改后的可重复性验证步骤，每个步骤均提供中英文说明。

## 1. Configuration merge / 配置合并
- **English**: Ensure base and dataset configs load and environment overrides are applied.
- **中文**：确保基础配置与数据集配置能加载，并且环境变量覆盖生效。

```bash
python - <<'PY'
from pathlib import Path
from piuq.config import load_config
cfg = load_config(Path("configs/default.yaml"), [Path("configs/datasets/highd.yaml")])
print(cfg.dict())
PY
```

## 2. Raw data smoothing and sampling / 原始数据平滑与采样
- **English**: Validate `smooth_positions` and `downsample_tracks` on a small CSV to confirm kinematic preprocessing is correct.
- **中文**：在小型 CSV 上运行 `smooth_positions` 与 `downsample_tracks`，确认运动学预处理正确。

```bash
python - <<'PY'
import pandas as pd
from piuq.data.pipeline import smooth_positions, downsample_tracks
sample = pd.DataFrame({"track_id": [1]*5, "t": [0,0.04,0.08,0.12,0.16], "x": range(5), "y": range(5), "vx": range(5), "vy": range(5), "ax": range(5), "ay": range(5)})
print(smooth_positions(sample, 3).head())
print(downsample_tracks(sample, 5)[0].head())
PY
```

## 3. Visualization / 可视化
- **English**: Generate a scene plot from raw trajectories and verify the output image exists.
- **中文**：从原始轨迹生成场景图，并验证输出图片存在。

```bash
python scripts/visualize_raw.py --input data/raw/highd_sample.csv --output docs/artifacts/scene.png --recording-id 1
ls docs/artifacts/scene.png
```

## 4. Window extraction / 窗口提取
- **English**: Run `scripts/preprocess.py` with a small dataset and inspect the produced pickle/Parquet files in `data/processed/`.
- **中文**：使用小型数据运行 `scripts/preprocess.py`，检查 `data/processed/` 生成的 pickle/Parquet 文件。

## 5. Training sanity check / 训练合理性检查
- **English**: Build a `DataLoader` via `window_tensor_loader` and ensure metrics compute without errors.
- **中文**：通过 `window_tensor_loader` 构建 `DataLoader`，确保指标计算无误。

```bash
python - <<'PY'
import torch
from piuq.training.loop import window_tensor_loader, ade, fde
windows = [{"history": torch.zeros((3,2)), "future": torch.ones((2,2))} for _ in range(4)]
loader = window_tensor_loader(windows, batch_size=2, shuffle=False)
for hist, fut in loader:
    pred = torch.zeros_like(fut)
    print(float(ade(pred, fut)), float(fde(pred, fut)))
PY
```
