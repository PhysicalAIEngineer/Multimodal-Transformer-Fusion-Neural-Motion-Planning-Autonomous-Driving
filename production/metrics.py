from prometheus_client import Counter, Gauge, Histogram
REQUESTS = Counter("autodrive_inference_requests_total", "Inference requests", ["endpoint"])
ERRORS = Counter("autodrive_inference_errors_total", "Inference errors", ["endpoint", "type"])
LATENCY = Histogram("autodrive_inference_latency_seconds", "Inference latency", ["endpoint"])
MODEL_LOADED = Gauge("autodrive_model_loaded", "Whether the model is loaded", ["model_version"])
