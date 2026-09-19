# Production Deployment Guide

This branch adds a deployment layer around the existing research implementation without rewriting the research model.

## Architecture

CARLA / Sensor Preprocessing
          |
          v
RGB + LiDAR + target + speed tensors
          |
          v
FastAPI /predict ---> Prometheus /metrics
          |
          v
LidarCenterNet
  |       |       |
  v       v       v
Waypoints Detection Semantic + Depth + BEV

## Local validation

python -m pip install -r requirements-api.txt
python -m ruff check production tests
python -m compileall -q production "Helper Scripts"
pytest -q

## Model serving

Install model dependencies separately because PyTorch is platform-specific:

pip install -r requirements-model.txt
export MODEL_PATH=models/transfuser_regnet032_seed1_39.pth
python -m production

Endpoints: /health, /ready, /model, /predict, /metrics.

The /ready endpoint deliberately fails until a real checkpoint is present. This prevents a container from reporting readiness while serving random or untrained weights.

## Production controls

- immutable artifact path and SHA-256 model version
- lazy, thread-safe model loading
- CPU/GPU selection through DEVICE
- liveness/readiness separation
- Prometheus request, error, latency and model-health metrics
- read-only model volume in Compose
- container capability dropping and restart policy
- CI lint, compilation and unit tests
- explicit checkpoint validation

## Safety boundary

This is a research/engineering inference service, not a safety-certified autonomous-driving controller. Real-vehicle deployment requires independent safety monitors, redundancy, fail-safe behavior, simulation, road testing, verification/validation, and hard vehicle-level safety constraints.
