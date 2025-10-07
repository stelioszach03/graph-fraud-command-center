from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "timestamp_utc" in data
    assert "model_loaded" in data


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "aegis_http_requests_total" in r.text
