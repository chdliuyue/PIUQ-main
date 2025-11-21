# Configuration guide

All runnable scripts consume a YAML configuration. The default lives at `configs/default.yaml` and is validated by `piuq.config.Config`.

## Base concepts
- **paths:** locations for raw/processed data and logs. `data/raw` and `data/processed` are ignored by git.
- **preprocess:** sampling/downsampling, smoothing, centerline resolution, window lengths, neighbor search radius, and dataset list.
- **windows:** stride and padding value for tensorization.
- **training:** placeholders for future model/trainer parameters including physics and uncertainty loss weights.

## Overrides
1. Supply additional YAMLs via `--config-overrides` (applied in order).
2. Use environment variables matching the dotted path in uppercase, e.g.:
   ```bash
   PREPROCESS_HISTORY_SEC=4.0 PREPROCESS_ALLOW_GAPS=true python scripts/preprocess.py
   ```
3. CLI flags `--dataset` and `--out` take precedence for the preprocess entrypoint.

## Example
```bash
python scripts/preprocess.py \
  --config configs/default.yaml \
  --config-overrides configs/experiments/highd_downsample.yaml \
  --dataset highD
```
