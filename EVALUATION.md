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

## 5. Fusion ablation

Run the same route set and evaluation protocol for:

| Configuration | 3D AP | BEV AP | mIoU | Depth RMSE | Waypoint ADE | Collision Rate | Route Completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RGB-only | — | — | — | — | — | — | — |
| LiDAR-only | — | — | — | — | — | — | — |
| RGB + LiDAR | — | — | — | — | — | — | — |

This isolates the effect of multimodal fusion without hard-coding a preferred outcome.

## 6. Required experiment manifest

Each run should record:

- repository commit SHA
- model checkpoint SHA-256
- CARLA version
- map and route identifiers
- sensor resolution and rate
- random seed
- episode count
- weather configuration
- GPU and memory
- dependency versions

A template is provided in configs/evaluation.example.json.

## 7. Evaluation record format

The JSONL aggregator accepts one record per frame, sample, or episode. Example:

{
  "3D_detection_APs": [],
  "BEV_detection_APs": [],
  "segmentation_mIoU_values": [0.72],
  "depth_MAE_values": [1.8],
  "depth_RMSE_values": [3.1],
  "waypoint_ADE_values": [0.45],
  "waypoint_FDE_values": [0.82],
  "trajectory_collision_flags": [false],
  "route_completion_values": [0.94],
  "offroad_flags": [false],
  "collision_flags": [false],
  "red_light_flags": [false],
  "lane_departure_flags": [false],
  "intervention_flags": [false],
  "runtime_samples": [
    {
      "preprocessing_ms": 4.2,
      "inference_ms": 18.7,
      "rendering_ms": 3.1
    }
  ],
  "gpu_memory_mb": [5320],
  "gpu_utilization_pct": [87]
}

These are example field names only. The numeric values above are illustrative and must not be reported as project results.

## 8. Running the aggregator

python scripts/evaluate_jsonl.py --input evaluation_records.jsonl --output evaluation_results.json

The framework creates a machine-readable report that can be consumed by CI, notebooks, dashboards, or later publication tooling.

## Research reporting standard

For a paper, arXiv submission, or technical portfolio entry, report the exact evaluation split, route count, seed, checkpoint hash, simulator version, sensor configuration, and confidence/dispersion where appropriate. Never report illustrative numbers as measurements.
