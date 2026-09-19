from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .metrics import ERRORS, LATENCY, MODEL_LOADED, REQUESTS
from .runtime import runtime
from .schemas import PredictionRequest, PredictionResponse
from .settings import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(
    title="Multimodal Transformer Fusion Autonomous Driving API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "autonomous-driving-inference"}


@app.get("/ready")
def ready() -> dict[str, str]:
    try:
        runtime.load()
        MODEL_LOADED.labels(runtime.model_version).set(1)
        return {"status": "ready", "model_version": runtime.model_version}
    except Exception as exc:
        ERRORS.labels("ready", type(exc).__name__).inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/model")
def model_info() -> dict[str, str | bool]:
    return {
        "loaded": runtime.loaded,
        "model_version": runtime.model_version,
        "device": settings.resolved_device,
        "artifact": str(settings.model_path),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    start = time.perf_counter()
    REQUESTS.labels("predict").inc()
    try:
        result = runtime.predict(request.model_dump())
        MODEL_LOADED.labels(runtime.model_version).set(1)
        return PredictionResponse(**result)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        ERRORS.labels("predict", type(exc).__name__).inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        ERRORS.labels("predict", "internal").inc()
        raise HTTPException(status_code=500, detail="Inference failed") from exc
    finally:
        LATENCY.labels("predict").observe(time.perf_counter() - start)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
