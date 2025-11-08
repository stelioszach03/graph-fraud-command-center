from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_reset_endpoint_clears_state():
    # Seed a few events
    client.post("/api/v1/simulate?events=30&seed=7")

    s_before = client.get("/api/v1/graph/summary").json()
    assert s_before["events_total"] >= 30

    r = client.post("/api/v1/reset")
    assert r.status_code == 200
    body = r.json()
    assert body["cleared_events"] >= 30
    assert body["cleared_nodes"] >= 1

    s_after = client.get("/api/v1/graph/summary").json()
    assert s_after["nodes_total"] == 0
    assert s_after["edges_total"] == 0
    assert s_after["events_total"] == 0
    assert s_after["alerts_total"] == 0


def test_simulate_returns_elapsed_and_critical():
    client.post("/api/v1/reset")
    r = client.post("/api/v1/simulate?events=300&seed=11")
    assert r.status_code == 200
    data = r.json()
    assert data["generated"] == 300
    assert data["alerts_created"] >= 0
    assert "critical_created" in data
    assert "elapsed_ms" in data
    assert data["elapsed_ms"] >= 0


def test_neighborhood_404_for_unknown_node():
    client.post("/api/v1/reset")
    r = client.get("/api/v1/graph/neighborhood/NOT_A_REAL_NODE")
    assert r.status_code == 404


def test_neighborhood_returns_subgraph_after_simulate():
    client.post("/api/v1/reset")
    client.post("/api/v1/simulate?events=400&seed=42")

    # Pick any node from the summary route via alerts
    alerts = client.get("/api/v1/alerts?min_score=0.0&limit=1").json()
    assert len(alerts) >= 1
    focus = alerts[0]["sender_id"]

    r = client.get(f"/api/v1/graph/neighborhood/{focus}?depth=1&limit=20")
    assert r.status_code == 200
    data = r.json()
    assert data["focus"] == focus
    assert data["depth"] == 1
    assert any(n["is_focus"] for n in data["nodes"])
    assert len(data["nodes"]) >= 1
    for node in data["nodes"]:
        assert "in_count" in node and "out_count" in node
