# PIUQ: Physics-informed + Uncertainty highway trajectory toolkit

This repository scaffolds data processing and future modeling for UAV highway datasets (highD/exiD/A43, etc.).

## Layout
- `configs/`: YAML configuration files; see `configs/default.yaml` and dataset specific
  files under `configs/datasets/` (e.g., `configs/datasets/highd.yaml`).
- `scripts/`: entrypoints (e.g., `preprocess.py`).
- `src/piuq/`: library code for configuration, dataset adapters, Frenet conversion,
  window building, visualisation utilities, and baseline GRU model/training helpers.
- `data/raw/`: place raw dataset archives or extracted CSVs (gitignored).
- `data/processed/`: outputs such as Parquet and window pickles (gitignored).
- `docs/`: configuration and architecture notes.
- `notebooks/`: exploratory analysis (empty scaffold).
- `tests/`: unit tests.

## Quickstart
```bash
pip install -r requirements.txt
python scripts/preprocess.py --config configs/default.yaml --dataset highD
```

Processed Frenet trajectories and window pickles will be written to `data/processed/`.

## Detailed workflow

The project is organized around three stages so you can quickly self-validate changes:

1. **Configuration merge**: load `configs/default.yaml` plus a dataset override from `configs/datasets/`. Environment variables such as `PREPROCESS_SAMPLING_HZ` can override any field (see `src/piuq/config.py`).
2. **Preprocessing and windowing**: `python scripts/preprocess.py --config configs/default.yaml --dataset highD` converts raw CSV/Parquet files into harmonized Frenet trajectories and sliding windows stored in `data/processed/`.
3. **Visualization and modeling checks**: use the new `scripts/visualize_raw.py` helper to render trajectories before training, then feed the generated windows into baseline GRU training utilities in `src/piuq/training/`.

## Raw data visualisation

Use the CLI below to confirm raw trajectories render correctly (ideal for stage-by-stage validation):

```bash
python scripts/visualize_raw.py --input data/raw/highd_sample.csv --recording-id 1 --tracks 15 42 --output docs/artifacts/highd_scene.png
```

- The command loads CSV/Parquet data, filters by recording ID and track IDs, and generates a map-view plot using `piuq.visualization.plot_scene`.
- `--output` saves the figure instead of opening an interactive window, which is convenient for CI or remote environments.

## Validation checklist

- ✅ Configuration files load with environment overrides: `python -c "from pathlib import Path; from piuq.config import load_config; print(load_config(Path('configs/default.yaml')))"`
- ✅ Preprocessing stage completes: `python scripts/preprocess.py --config configs/default.yaml --dataset highD`
- ✅ Visualization stage produces a plot: run the `visualize_raw.py` command above and confirm an image is written.

## Documentation

- `docs/configuration.md`: reference for YAML fields.
- `docs/validation.md`: step-by-step validation guide for configuration, preprocessing, and visualization.
