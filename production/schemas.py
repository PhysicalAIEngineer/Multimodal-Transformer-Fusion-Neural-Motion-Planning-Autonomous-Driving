from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field

class TensorPayload(BaseModel):
    data: list[Any]
    shape: list[int] = Field(min_length=1, max_length=5)

class PredictionRequest(BaseModel):
    rgb: TensorPayload
    lidar: TensorPayload
    target_point_image: TensorPayload
    speed: TensorPayload
    target_point: TensorPayload

class PredictionResponse(BaseModel):
    model_version: str
    device: str
    waypoints: list[Any]
    detections: list[Any]
    semantic_shape: list[int]
    depth_shape: list[int]
    bev_shape: list[int]
