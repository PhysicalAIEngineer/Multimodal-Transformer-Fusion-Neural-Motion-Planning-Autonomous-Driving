from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("autodrive_inference_requests_total", "Inference requests", ["endpoint"])
ERRORS = Counter("autodrive_inference_errors_total", "Inference errors", ["endpoint", "type"])
LATENCY = Histogram("autodrive_inference_latency_seconds", "Inference latency", ["endpoint"])
MODEL_LOADED = Gauge("autodrive_model_loaded", "Whether the model is loaded", ["model_version"])


def _xy(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Expected points with shape [N,2] or [B,N,2].")
    return arr[:, :2]


def mean_absolute_error(prediction: Any, target: Any) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {truth.shape}")
    return float(np.mean(np.abs(pred - truth)))


def root_mean_squared_error(prediction: Any, target: Any) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {truth.shape}")
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def mean_iou(prediction: Any, target: Any, num_classes: int | None = None) -> float:
    pred = np.asarray(prediction)
    truth = np.asarray(target)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {truth.shape}")
    classes = range(num_classes) if num_classes is not None else np.unique(
        np.concatenate([pred.ravel(), truth.ravel()])
    )
    values = []
    for cls in classes:
        p = pred == cls
        t = truth == cls
        union = np.logical_or(p, t).sum()
        if union:
            values.append(np.logical_and(p, t).sum() / union)
    return 0.0 if not values else float(np.mean(values))


def waypoint_ade(prediction: Any, target: Any) -> float:
    pred = _xy(prediction)
    truth = _xy(target)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {truth.shape}")
    return float(np.linalg.norm(pred - truth, axis=1).mean())


def waypoint_fde(prediction: Any, target: Any) -> float:
    pred = _xy(prediction)
    truth = _xy(target)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {truth.shape}")
    return float(np.linalg.norm(pred[-1] - truth[-1]))


def event_rate(flags: Iterable[bool]) -> float:
    values = [bool(x) for x in flags]
    return 0.0 if not values else float(np.mean(values))


def event_count(flags: Iterable[bool]) -> int:
    return sum(bool(x) for x in flags)


def route_completion(completed_distance: float, route_distance: float) -> float:
    if route_distance <= 0:
        return 0.0
    return float(np.clip(completed_distance / route_distance, 0.0, 1.0))


def percentile(values: Sequence[float], p: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    return 0.0 if arr.size == 0 else float(np.percentile(arr, p))


@dataclass
class RuntimeSample:
    preprocessing_ms: float
    inference_ms: float
    rendering_ms: float = 0.0
    end_to_end_ms: float | None = None

    def __post_init__(self) -> None:
        if self.end_to_end_ms is None:
            self.end_to_end_ms = self.preprocessing_ms + self.inference_ms + self.rendering_ms


@dataclass
class EvaluationAccumulator:
    detection_ap: list[float] = field(default_factory=list)
    bev_detection_ap: list[float] = field(default_factory=list)
    segmentation_miou: list[float] = field(default_factory=list)
    depth_mae: list[float] = field(default_factory=list)
    depth_rmse: list[float] = field(default_factory=list)
    waypoint_ade_values: list[float] = field(default_factory=list)
    waypoint_fde_values: list[float] = field(default_factory=list)
    trajectory_collision_flags: list[bool] = field(default_factory=list)
    route_completion_values: list[float] = field(default_factory=list)
    offroad_flags: list[bool] = field(default_factory=list)
    collision_flags: list[bool] = field(default_factory=list)
    red_light_flags: list[bool] = field(default_factory=list)
    lane_departure_flags: list[bool] = field(default_factory=list)
    intervention_flags: list[bool] = field(default_factory=list)
    runtime_samples: list[RuntimeSample] = field(default_factory=list)
    gpu_memory_mb: list[float] = field(default_factory=list)
    gpu_utilization_pct: list[float] = field(default_factory=list)

    def add_runtime_sample(self, **kwargs: float) -> None:
        self.runtime_samples.append(RuntimeSample(**kwargs))

    def summary(self) -> dict[str, Any]:
        samples = self.runtime_samples
        e2e = [float(x.end_to_end_ms or 0.0) for x in samples]
        return {
            "perception": {
                "3d_detection_ap": mean(self.detection_ap) if self.detection_ap else None,
                "bev_detection_ap": mean(self.bev_detection_ap) if self.bev_detection_ap else None,
                "segmentation_mIoU": mean(self.segmentation_miou) if self.segmentation_miou else None,
                "depth_MAE": mean(self.depth_mae) if self.depth_mae else None,
                "depth_RMSE": mean(self.depth_rmse) if self.depth_rmse else None,
            },
            "planning": {
                "waypoint_ADE": mean(self.waypoint_ade_values) if self.waypoint_ade_values else None,
                "waypoint_FDE": mean(self.waypoint_fde_values) if self.waypoint_fde_values else None,
                "trajectory_collision_rate": event_rate(self.trajectory_collision_flags),
                "route_completion": mean(self.route_completion_values)
                if self.route_completion_values
                else None,
                "offroad_rate": event_rate(self.offroad_flags),
            },
            "system": {
                "FPS": 0.0 if not e2e or mean(e2e) <= 0 else 1000.0 / mean(e2e),
                "end_to_end_latency_ms_p50": percentile(e2e, 50),
                "end_to_end_latency_ms_p95": percentile(e2e, 95),
                "preprocessing_latency_ms_p50": percentile([x.preprocessing_ms for x in samples], 50),
                "preprocessing_latency_ms_p95": percentile([x.preprocessing_ms for x in samples], 95),
                "model_inference_latency_ms_p50": percentile([x.inference_ms for x in samples], 50),
                "model_inference_latency_ms_p95": percentile([x.inference_ms for x in samples], 95),
                "rendering_latency_ms_p50": percentile([x.rendering_ms for x in samples], 50),
                "rendering_latency_ms_p95": percentile([x.rendering_ms for x in samples], 95),
                "gpu_memory_mb": mean(self.gpu_memory_mb) if self.gpu_memory_mb else None,
                "gpu_memory_peak_mb": max(self.gpu_memory_mb) if self.gpu_memory_mb else None,
                "gpu_utilization_pct": mean(self.gpu_utilization_pct)
                if self.gpu_utilization_pct
                else None,
            },
            "driving": {
                "collision_count": event_count(self.collision_flags),
                "red_light_violations": event_count(self.red_light_flags),
                "lane_departures": event_count(self.lane_departure_flags),
                "route_completion": mean(self.route_completion_values)
                if self.route_completion_values
                else None,
                "intervention_recovery_count": event_count(self.intervention_flags),
            },
        }


def timed_call(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000.0
