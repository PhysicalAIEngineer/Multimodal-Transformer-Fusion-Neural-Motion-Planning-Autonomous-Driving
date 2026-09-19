# CARLA Automated Evaluation

## Goal

Run fixed CARLA routes, collect synchronized sensor frames and driving events, measure runtime, and produce machine-readable and presentation-ready results.

## Architecture

```
CARLA
  |
  +-- fixed route / seed
  |
  +-- RGB + LiDAR
  |
  +-- synchronized world.tick()
  |
  +-- model adapter (optional until a real checkpoint is supplied)
  |
  +-- prediction + timing
  |
  +-- collision / lane / traffic events
  |
  +-- JSONL episode records
  |
  +-- evaluation_results.json
  |
  +-- evaluation_report.md
  +-- evaluation_metrics.png
```

## Run

Start CARLA separately, then run:

```bash
python scripts/carla_evaluate.py \
  --host 127.0.0.1 \
  --port 2000 \
  --seed 1 \
  --routes route_00:0:1,route_01:2:3 \
  --output-dir artifacts/evaluation
```

Aggregate the resulting records:

```bash
python scripts/evaluate_jsonl.py \
  --input artifacts/evaluation/evaluation_records.jsonl \
  --output artifacts/evaluation/evaluation_results.json
```

Create plots and a report:

```bash
python scripts/plot_evaluation.py \
  --input artifacts/evaluation/evaluation_results.json \
  --output-dir artifacts/evaluation/plots

python scripts/generate_evaluation_report.py \
  --input artifacts/evaluation/evaluation_results.json \
  --output artifacts/evaluation/evaluation_report.md
```

## Fixed-route protocol

For every comparison:

1. Use the same CARLA map.
2. Use the same route list.
3. Use the same seed.
4. Use the same sensor configuration.
5. Use the same maximum episode length.
6. Change only the experimental variable.
7. Save the repository commit SHA and checkpoint SHA-256.
8. Run enough episodes to report dispersion rather than a single cherry-picked episode.

## Model evaluation boundary

The current repository checkpoint is intentionally rejected by the production runtime because it is not a usable trained artifact. Therefore the runner currently provides a simulator/evaluation harness and a controller baseline. Neural-model metrics become populated only after a real compatible checkpoint and model adapter are supplied.

Do not present the baseline controller as TransFuser performance and do not present empty neural metrics as measured results.
