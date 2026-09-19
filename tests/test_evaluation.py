from __future__ import annotations

import numpy as np

from production.evaluation import aggregate_rows, compare_fusion_modes, evaluate_sample
from production.metrics import EvaluationAccumulator, mean_iou, route_completion, waypoint_ade, waypoint_fde


def test_segmentation_miou() -> None:
    prediction = np.array([[0, 1], [1, 1]])
    target = np.array([[0, 1], [0, 1]])
    assert mean_iou(prediction, target) == 2.0 / 3.0


def test_waypoint_metrics() -> None:
    prediction = np.array([[0.0, 0.0], [2.0, 0.0]])
    target = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert waypoint_ade(prediction, target) == 1.0
    assert waypoint_fde(prediction, target) == 1.0


def test_route_completion() -> None:
    assert route_completion(120.0, 100.0) == 1.0
    assert route_completion(50.0, 100.0) == 0.5


def test_evaluation_sample() -> None:
    metrics = evaluate_sample(
        pred_semantic=np.array([[0, 1], [1, 1]]),
        target_semantic=np.array([[0, 1], [0, 1]]),
        pred_waypoints=np.array([[0.0, 0.0], [2.0, 0.0]]),
        target_waypoints=np.array([[0.0, 1.0], [1.0, 0.0]]),
    )
    assert "segmentation_mIoU" in metrics
    assert metrics["waypoint_ADE"] == 1.0


def test_accumulator_summary() -> None:
    accumulator = EvaluationAccumulator()
    accumulator.waypoint_ade_values.append(0.2)
    accumulator.waypoint_fde_values.append(0.4)
    accumulator.route_completion_values.append(0.8)
    accumulator.trajectory_collision_flags.append(False)
    accumulator.add_runtime_sample(
        preprocessing_ms=2.0,
        inference_ms=8.0,
        rendering_ms=5.0,
    )
    summary = accumulator.summary()
    assert summary["planning"]["waypoint_ADE"] == 0.2
    assert summary["planning"]["trajectory_collision_rate"] == 0.0
    assert summary["system"]["FPS"] > 0.0


def test_aggregation_and_ablation() -> None:
    rows = [
        {
            "segmentation_mIoU_values": [0.5, 0.7],
            "depth_MAE_values": [1.0],
            "depth_RMSE_values": [2.0],
            "waypoint_ADE_values": [0.3],
            "waypoint_FDE_values": [0.5],
            "route_completion_values": [0.8],
        }
    ]
    summary = aggregate_rows(rows)
    assert summary["perception"]["segmentation_mIoU"] == 0.6

    ablation = compare_fusion_modes(
        {"3d_ap": 0.1},
        {"3d_ap": 0.2},
        {"3d_ap": 0.3},
    )
    assert set(ablation) == {"rgb_only", "lidar_only", "rgb_lidar"}
