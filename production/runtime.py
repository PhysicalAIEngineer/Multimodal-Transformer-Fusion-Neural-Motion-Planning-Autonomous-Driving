from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from threading import Lock
from typing import Any

from .settings import settings


class ModelRuntime:
    """Thread-safe lazy loader around the repository's existing LidarCenterNet."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._torch: Any | None = None
        self._lock = Lock()
        self.model_version = "unloaded"
        self.last_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _artifact_version(self, path: Path) -> str:
        return f"transfuser-{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            path = settings.model_path
            if not path.exists():
                self.last_error = f"Model artifact not found: {path}"
                raise FileNotFoundError(self.last_error)
            if path.stat().st_size < 1024:
                self.last_error = (
                    f"Model artifact is too small ({path.stat().st_size} bytes); provide the real checkpoint."
                )
                raise RuntimeError(self.last_error)
            try:
                import torch
            except ImportError as exc:
                self.last_error = "PyTorch is not installed"
                raise RuntimeError(self.last_error) from exc

            helper_dir = str(Path(__file__).resolve().parents[1] / "Helper Scripts")
            if helper_dir not in sys.path:
                sys.path.insert(0, helper_dir)
            config_module = importlib.import_module("config")
            model_module = importlib.import_module("model")
            config = config_module.GlobalConfig()
            config.debug = True
            device = torch.device(settings.resolved_device)
            model = model_module.LidarCenterNet(
                config,
                device,
                config.backbone,
                image_architecture="regnety_032",
                lidar_architecture="regnety_032",
                estimate_loss=False,
            )
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if isinstance(state, dict):
                state = {k.removeprefix("module."): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            model.eval()
            self._torch = torch
            self._model = model
            self.model_version = self._artifact_version(path)
            self.last_error = None

    def predict_tensors(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run inference and return raw model outputs for local evaluation."""
        self.load()
        assert self._model is not None and self._torch is not None
        torch = self._torch
        tensors: dict[str, Any] = {}
        for key, item in payload.items():
            tensors[key] = (
                torch.tensor(item["data"], dtype=torch.float32)
                .reshape(item["shape"])
                .to(settings.resolved_device)
            )
        if settings.resolved_device.startswith("cuda"):
            torch.cuda.synchronize()
        with torch.inference_mode():
            _, output = self._model(tensors)
        if settings.resolved_device.startswith("cuda"):
            torch.cuda.synchronize()
        return output

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        output = self.predict_tensors(payload)
        pred_semantic = output.get("pred_semantic")
        pred_depth = output.get("pred_depth")
        pred_bev = output.get("pred_bev")
        pred_wp = output.get("pred_wp")
        detections = output.get("detections", [])
        return {
            "model_version": self.model_version,
            "device": settings.resolved_device,
            "waypoints": pred_wp.tolist() if hasattr(pred_wp, "tolist") else [],
            "detections": [x.tolist() for x in detections],
            "semantic_shape": list(pred_semantic.shape) if hasattr(pred_semantic, "shape") else [],
            "depth_shape": list(pred_depth.shape) if hasattr(pred_depth, "shape") else [],
            "bev_shape": list(pred_bev.shape) if hasattr(pred_bev, "shape") else [],
        }


runtime = ModelRuntime()
