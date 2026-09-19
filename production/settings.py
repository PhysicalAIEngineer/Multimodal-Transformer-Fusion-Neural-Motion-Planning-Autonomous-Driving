from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv("MODEL_PATH", "models/transfuser_regnet032_seed1_39.pth"))
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    device: str = os.getenv("DEVICE", "auto")
    log_level: str = os.getenv("LOG_LEVEL", "info")

    @property
    def resolved_device(self) -> str:
        if self.device != "auto": return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

settings = Settings()
