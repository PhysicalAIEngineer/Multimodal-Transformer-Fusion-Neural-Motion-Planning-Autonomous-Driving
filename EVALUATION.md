# Evaluation and Benchmarking

This repository now treats evaluation as a first-class research artifact. The framework separates perception, planning, system-runtime, and driving metrics and records the information required to reproduce an experiment.

## 1. Perception metrics

| Metric | Definition | Primary source |
| --- | --- | --- |
| 3D detection AP | Average precision using the detector's documented matching and IoU configuration | Detector evaluator |
| BEV detection AP | Average precision in bird's-eye-view box space | BEV detector evaluator |
| Segmentation mIoU | Mean intersection-over-union across evaluated semantic classes | Ground-truth semantic maps |
| Depth MAE | Mean absolute depth error | Ground-truth depth |
| Depth RMSE | Root mean squared depth error | Ground-truth depth |

The AP fields are deliberately exposed as integration points because the exact AP/NDS definition depends on the box representation, class set, and evaluation convention. Do not mix metrics from incompatible evaluators.

## 2. Planning metrics

| Metric | Definition |
| --- | --- |
| Waypoint ADE | Mean Euclidean distance between predicted and target waypoints |
| Waypoint FDE | Euclidean distance between the final predicted and target waypoint |
| Trajectory collision rate | Fraction of evaluated trajectories with at least one collision |
| Route completion | Completed route distance divided by total route distance |
| Off-road rate | Fraction of evaluated frames or episodes marked off-road |

## 3. System metrics

| Metric | Definition |
| --- | --- |
| FPS | 1000 divided by mean end-to-end latency in milliseconds |
| End-to-end latency | p50 and p95 wall-clock latency |
| GPU memory | Mean and peak VRAM recorded during evaluation |
| GPU utilization | Mean utilization when platform telemetry is available |
| Preprocessing latency | p50 and p95 preprocessing time |
| Model inference latency | p50 and p95 model execution time |
| Rendering latency | p50 and p95 visualization time |

Use synchronized CUDA measurements for GPU inference timing when the model is executed on CUDA. Wall-clock timing without synchronization can under-report GPU work.

## 4. Driving metrics

| Metric | Definition |
| --- | --- |
| Collision count | Number of collision events |
| Red-light violations | Number of traffic-light violations |
| Lane departures | Number of lane-boundary departures |
| Route completion | Fraction of route completed |
| Intervention / recovery count | Number of planner/controller interventions or recoveries |

