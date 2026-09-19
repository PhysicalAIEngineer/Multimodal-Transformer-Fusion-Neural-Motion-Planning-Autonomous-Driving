from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics import (
    EvaluationAccumulator,
    mean_absolute_error,
    mean_iou,
    root_mean_squared_error,
    waypoint_ade,
    waypoint_fde,
)


def evaluate_sample(
    *,
    pred_semantic: Any | None = None,
    target_semantic: Any | None = None,
    pred_depth: Any | None = None,
    target_depth: Any | None = None,
    pred_waypoints: Any | None = None,
    target_waypoints: Any | None = None,
    detection_ap: float | None = None,
    bev_detection_ap: float | None = None,
) -> dict[str, float]:
    """Compute sample-level metrics from measured predictions and labels."""
    metrics: dict[str, float] = {}

    if pred_semantic is not None and target_semantic is not None:
        metrics["segmentation_mIoU"] = mean_iou(pred_semantic, target_semantic)

    if pred_depth is not None and target_depth is not None:
        metrics["depth_MAE"] = mean_absolute_error(pred_depth, target_depth)
        metrics["depth_RMSE"] = root_mean_squared_error(pred_depth, target_depth)

    if pred_waypoints is not None and target_waypoints is not None:
        metrics["waypoint_ADE"] = waypoint_ade(pred_waypoints, target_waypoints)
        metrics["waypoint_FDE"] = waypoint_fde(pred_waypoints, target_waypoints)

    if detection_ap is not None:
        metrics["3D_detection_AP"] = float(detection_ap)

    if bev_detection_ap is not None:
        metrics["BEV_detection_AP"] = float(bev_detection_ap)

    return metrics


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate JSON-compatible episode/sample records into one report."""
    accumulator = EvaluationAccumulator()

    for row in rows:
        accumulator.detection_ap.extend(row.get("3D_detection_APs", []))
        accumulator.bev_detection_ap.extend(row.get("BEV_detection_APs", []))
        accumulator.segmentation_miou.extend(row.get("segmentation_mIoU_values", []))
        accumulator.depth_mae.extend(row.get("depth_MAE_values", []))
        accumulator.depth_rmse.extend(row.get("depth_RMSE_values", []))
        accumulator.waypoint_ade_values.extend(row.get("waypoint_ADE_values", []))
        accumulator.waypoint_fde_values.extend(row.get("waypoint_FDE_values", []))
        accumulator.trajectory_collision_flags.extend(
            row.get("trajectory_collision_flags", [])
        )
        accumulator.route_completion_values.extend(
            row.get("route_completion_values", [])
        )
        accumulator.offroad_flags.extend(row.get("offroad_flags", []))
        accumulator.collision_flags.extend(row.get("collision_flags", []))
        accumulator.red_light_flags.extend(row.get("red_light_flags", []))
        accumulator.lane_departure_flags.extend(row.get("lane_departure_flags", []))
        accumulator.intervention_flags.extend(row.get("intervention_flags", []))
        accumulator.gpu_memory_mb.extend(row.get("gpu_memory_mb", []))
        accumulator.gpu_utilization_pct.extend(row.get("gpu_utilization_pct", []))
        for sample in row.get("runtime_samples", []):
            accumulator.add_runtime_sample(**sample)

    return accumulator.summary()


def aggregate_jsonl(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Aggregate one JSON object per line into a deterministic JSON report."""
    rows = [
        json.loads(line)
        for line in Path(input_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = aggregate_rows(rows)
    Path(output_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def compare_fusion_modes(
    rgb_only: dict[str, float],
    lidar_only: dict[str, float],
    rgb_lidar: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Return a three-way ablation table without assuming a winner."""
    return {
        "rgb_only": dict(rgb_only),
        "lidar_only": dict(lidar_only),
        "rgb_lidar": dict(rgb_lidar),
    }
