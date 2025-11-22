# Configuration guide
# 配置指南

All runnable scripts consume a YAML configuration. The default lives at `configs/default.yaml` and is validated by `piuq.config.Config`.
所有可执行脚本都会读取一个 YAML 配置，默认配置位于 `configs/default.yaml`，并由 `piuq.config.Config` 进行校验。

## Base concepts
## 基本概念
- **paths:** Locations for raw/processed data and logs. `data/raw` and `data/processed` are ignored by git.
- **paths：** 原始/处理后数据与日志的位置，`data/raw` 与 `data/processed` 已被 git 忽略。
- **preprocess:** Sampling/downsampling, smoothing, centerline resolution, window lengths, neighbor search radius, and dataset list.
- **preprocess：** 采样/降采样、平滑、中心线分辨率、窗口长度、邻居搜索半径及数据集列表。
- **windows:** Stride and padding value for tensorization.
- **windows：** 张量化的步长与填充值。
- **training:** Placeholders for future model/trainer parameters including physics and uncertainty loss weights.
- **training：** 未来模型/训练参数的占位字段，包括物理与不确定性损失权重。

## Overrides
## 配置覆盖
1. Supply additional YAMLs via `--config-overrides` (applied in order).
1. 通过 `--config-overrides` 传入额外 YAML（按顺序应用）。
2. Use environment variables matching the dotted path in uppercase, for example:
2. 使用与点分路径对应的全大写环境变量，例如：
   ```bash
   PREPROCESS_HISTORY_SEC=4.0 PREPROCESS_ALLOW_GAPS=true python scripts/preprocess.py
   ```
3. CLI flags `--dataset` and `--out` take precedence for the preprocess entrypoint.
3. 在预处理入口中，命令行参数 `--dataset` 与 `--out` 拥有最高优先级。

## Example
## 示例
```bash
python scripts/preprocess.py \
  --config configs/default.yaml \
  --config-overrides configs/experiments/highd_downsample.yaml \
  --dataset highD
```
