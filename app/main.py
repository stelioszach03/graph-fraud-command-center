from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, Query

from app.schemas import AlertItem, GraphSummary, ScoreRequest, ScoreResponse, SimulateResponse
from app.services.graph_store import GraphStore, TxEvent
from app.services.scoring import FraudScoringService
from app.services.simulator import generate_stream
from app.settings import get_settings


settings = get_settings()
app = FastAPI(title="Aegis Graph Fraud GNN", version=settings.APP_VERSION)
store = GraphStore()
scorer = FraudScoringService(store=store, model_path=settings.MODEL_PATH, alert_min_score=settings.ALERT_MIN_SCORE)


@app.get("/")
def root() -> dict:
    return {
        "service": "aegis-graph-fraud-gnn",
        "status": "ok",
        "version": settings.APP_VERSION,
        "env": settings.APP_ENV,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "aegis-graph-fraud-gnn",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


@app.post("/api/v1/score", response_model=ScoreResponse)
def score_transaction(payload: ScoreRequest) -> ScoreResponse:
    tx = TxEvent(
        tx_id=payload.tx_id or f"TX-{uuid4().hex[:14]}",
        sender_id=payload.sender_id,
        receiver_id=payload.receiver_id,
        amount=float(payload.amount),
        currency=payload.currency,
        channel=payload.channel,
        country_from=payload.country_from,
        country_to=payload.country_to,
        timestamp_utc=payload.timestamp_utc or datetime.now(timezone.utc),
    )
    result = scorer.score(tx)
    return ScoreResponse(**result)


@app.get("/api/v1/graph/summary", response_model=GraphSummary)
def graph_summary() -> GraphSummary:
    return GraphSummary(**store.summary())


@app.get("/api/v1/alerts", response_model=list[AlertItem])
def alerts(
    min_score: float = Query(default=0.82, ge=0.0, le=1.0),
    limit: int = Query(default=25, ge=1, le=200),
) -> list[AlertItem]:
    rows = store.latest_alerts(min_score=float(min_score), limit=int(limit))
    return [AlertItem(**r) for r in rows]


@app.post("/api/v1/simulate", response_model=SimulateResponse)
def simulate(events: int = Query(default=250, ge=1, le=5000)) -> SimulateResponse:
    generated = generate_stream(n=events)
    alerts_before = len(store.alerts)

    for tx in generated:
        scorer.score(tx)

    alerts_after = len(store.alerts)
    return SimulateResponse(generated=len(generated), alerts_created=max(0, alerts_after - alerts_before))
