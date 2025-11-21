# PIUQ: Physics-informed + Uncertainty highway trajectory toolkit

This repository scaffolds data processing and future modeling for UAV highway datasets (highD/exiD/A43, etc.).

## Layout
- `configs/`: YAML configuration files; see `configs/default.yaml`.
- `scripts/`: entrypoints (e.g., `preprocess.py`).
- `src/piuq/`: library code for configuration, datasets, Frenet conversion, window building, and model/training stubs.
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
