from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_score_endpoint():
    payload = {
        "sender_id": "ACC_9001",
        "receiver_id": "ACC_9002",
        "amount": 4200,
        "currency": "USD",
        "channel": "wire",
        "country_from": "US",
        "country_to": "US",
    }
    r = client.post("/api/v1/score", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_band"] in {"low", "medium", "high", "critical"}


def test_simulate_and_summary():
    r = client.post("/api/v1/simulate?events=50")
    assert r.status_code == 200

    s = client.get("/api/v1/graph/summary")
    assert s.status_code == 200
    summary = s.json()
    assert summary["events_total"] >= 50
