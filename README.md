# PIUQ: Physics-informed + Uncertainty highway trajectory toolkit
# PIUQ：基于物理约束与不确定性的高速公路轨迹工具包

This repository provides data processing scaffolding and future modeling helpers for UAV highway datasets such as highD, exiD, and A43.
本仓库为高速公路无人机数据集（如 highD、exiD、A43 等）的数据处理与后续建模提供脚手架。

## Layout
## 项目结构
- `configs/`: YAML configuration files; see `configs/default.yaml` and dataset-specific files under `configs/datasets/` (for example, `configs/datasets/highd.yaml`).
- `configs/`：YAML 配置文件；参考 `configs/default.yaml` 以及 `configs/datasets/` 下的数据集专用配置（如 `configs/datasets/highd.yaml`）。
- `scripts/`: Entrypoints such as `preprocess.py` and helpers for visualization.
- `scripts/`：脚本入口（如 `preprocess.py`）以及可视化辅助脚本。
- `src/piuq/`: Library code for configuration, dataset adapters, Frenet conversion, window building, visualization utilities, and baseline GRU training helpers.
- `src/piuq/`：库代码，涵盖配置、数据集适配器、Frenet 转换、窗口构建、可视化工具以及基线 GRU 训练辅助。
- `data/raw/`: Place raw dataset archives or extracted CSV files (gitignored).
- `data/raw/`：存放原始数据压缩包或解压后的 CSV（已被 git 忽略）。
- `data/processed/`: Outputs such as Parquet files and window pickles (gitignored).
- `data/processed/`：存放处理后的 Parquet 与窗口 pickle 等输出（已被 git 忽略）。
- `docs/`: Configuration and architecture notes.
- `docs/`：配置与架构说明文档。
- `notebooks/`: Exploratory analysis notebooks (currently empty scaffolding).
- `notebooks/`：探索性分析的 Notebook（目前为空白脚手架）。
- `tests/`: Unit tests.
- `tests/`：单元测试。

## Quickstart
## 快速开始
```bash
pip install -r requirements.txt
pip install -e .
python scripts/preprocess.py --config configs/default.yaml --dataset highD
```

The editable install makes the `piuq` package available when running scripts directly. Processed Frenet trajectories and window pickles will be written to `data/processed/`.
处理后的 Frenet 轨迹与窗口 pickle 将写入 `data/processed/`。

## Detailed workflow
## 详细流程
The project is organized around three stages so you can quickly self-validate changes.
项目围绕三个阶段组织，便于快速自检改动。

1. **Configuration merge**: Load `configs/default.yaml` plus a dataset override from `configs/datasets/`. Environment variables such as `PREPROCESS_SAMPLING_HZ` can override any field (see `src/piuq/config.py`).
1. **配置合并**：加载 `configs/default.yaml` 与 `configs/datasets/` 中的数据集覆盖配置。环境变量（如 `PREPROCESS_SAMPLING_HZ`）可覆盖任意字段，详见 `src/piuq/config.py`。
2. **Preprocessing and windowing**: Run `python scripts/preprocess.py --config configs/default.yaml --dataset highD` to convert raw CSV/Parquet files into harmonized Frenet trajectories and sliding windows stored in `data/processed/`.
2. **预处理与窗口生成**：运行 `python scripts/preprocess.py --config configs/default.yaml --dataset highD` 将原始 CSV/Parquet 转换为统一的 Frenet 轨迹与滑动窗口，存放于 `data/processed/`。
3. **Visualization and modeling checks**: Use `scripts/visualize_raw.py` to render trajectories before training, then feed the generated windows into baseline GRU training utilities in `src/piuq/training/`.
3. **可视化与建模检查**：使用 `scripts/visualize_raw.py` 先行渲染轨迹，再将生成的窗口输入 `src/piuq/training/` 中的基线 GRU 训练工具。

## Raw data visualization
## 原始数据可视化
Use the CLI below to confirm raw trajectories render correctly (ideal for stage-by-stage validation).
使用以下命令确认原始轨迹渲染正常（便于逐阶段验证）。

```bash
python scripts/visualize_raw.py --input data/raw/highd_sample.csv --recording-id 1 --tracks 15 42 --output docs/artifacts/highd_scene.png
```

- The command loads CSV/Parquet data, filters by recording ID and track IDs, and generates a map-view plot using `piuq.visualization.plot_scene`.
- 该命令加载 CSV/Parquet 数据，按录制 ID 与轨迹 ID 过滤，并使用 `piuq.visualization.plot_scene` 生成地图视角的图像。
- `--output` saves the figure instead of opening an interactive window, which is convenient for CI or remote environments.
- `--output` 会保存图片而非打开交互窗口，适合 CI 或远程环境。

## Validation checklist
## 验证清单
- ✅ Configuration files load with environment overrides: `python -c "from pathlib import Path; from piuq.config import load_config; print(load_config(Path('configs/default.yaml')))"`
- ✅ 配置文件可加载并支持环境变量覆盖：`python -c "from pathlib import Path; from piuq.config import load_config; print(load_config(Path('configs/default.yaml')))"`
- ✅ Preprocessing stage completes: `python scripts/preprocess.py --config configs/default.yaml --dataset highD`
- ✅ 预处理阶段可正常完成：`python scripts/preprocess.py --config configs/default.yaml --dataset highD`
- ✅ Visualization stage produces a plot: run the `visualize_raw.py` command above and confirm an image is written.
- ✅ 可视化阶段能生成图像：运行上述 `visualize_raw.py` 命令并确认图片已写出。
- ✅ Processed outputs pass automated QA: `python scripts/data_selfcheck.py --processed-dir data/processed/highd --config configs/default.yaml`
- ✅ 处理结果通过自动质检：`python scripts/data_selfcheck.py --processed-dir data/processed/highd --config configs/default.yaml`

## Documentation
## 文档
- `docs/configuration.md`: Reference for YAML fields.
- `docs/configuration.md`：YAML 字段参考。
- `docs/validation.md`: Step-by-step validation guide for configuration, preprocessing, and visualization.
- `docs/validation.md`：配置、预处理与可视化的分步验证指南。
