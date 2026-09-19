from fastapi.testclient import TestClient
from production.api import app

def test_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_model_endpoint() -> None:
    response = TestClient(app).get("/model")
    assert response.status_code == 200
    assert "model_version" in response.json()
