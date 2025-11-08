from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["service"] == "aegis-graph-fraud-gnn"
    assert "timestamp_utc" in data
    assert "model_loaded" in data
    assert "model_blend_weight" in data
    assert "alert_min_score" in data
    assert "nodes_total" in data
    assert "edges_total" in data


def test_root_endpoint():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["service"] == "aegis-graph-fraud-gnn"
    assert data["status"] == "ok"
    assert isinstance(data["endpoints"], list)
    assert "summary" in data


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "aegis_http_requests_total" in r.text
    assert "aegis_risk_score" in r.text


def test_response_headers():
    r = client.get("/health")
    assert r.headers.get("x-service") == "aegis-graph-fraud-gnn"
    assert "x-response-time-ms" in {k.lower() for k in r.headers.keys()}
