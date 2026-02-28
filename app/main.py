from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Query, Request, Response

from app.metrics import record_http, record_score, render_metrics, set_graph_gauges
from app.schemas import AlertItem, GraphSummary, ScoreRequest, ScoreResponse, SimulateResponse
from app.services.graph_store import GraphStore, TxEvent
from app.services.scoring import FraudScoringService
from app.services.simulator import generate_stream
from app.settings import get_settings

settings = get_settings()
app = FastAPI(title="Aegis Graph Fraud GNN", version=settings.APP_VERSION)
store = GraphStore()
scorer = FraudScoringService(
    store=store,
    model_path=settings.MODEL_PATH,
    alert_min_score=settings.ALERT_MIN_SCORE,
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    t0 = perf_counter()
    response = await call_next(request)
    dt = perf_counter() - t0
    record_http(
        method=request.method,
        path=request.url.path,
        status=int(response.status_code),
        duration_s=float(dt),
    )
    return response


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
        "model_loaded": scorer.model is not None,
        "alert_min_score": settings.ALERT_MIN_SCORE,
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
    risk_score = float(result["risk_score"])
    record_score(
        score=risk_score,
        high_risk=risk_score >= settings.ALERT_MIN_SCORE,
    )
    set_graph_gauges(store.summary())
    return ScoreResponse(**result)


@app.get("/api/v1/graph/summary", response_model=GraphSummary)
def graph_summary() -> GraphSummary:
    summary = store.summary()
    set_graph_gauges(summary)
    return GraphSummary(**summary)


@app.get("/api/v1/alerts", response_model=list[AlertItem])
def alerts(
    min_score: float = Query(default=settings.ALERT_MIN_SCORE, ge=0.0, le=1.0),
    limit: int = Query(default=25, ge=1, le=200),
) -> list[AlertItem]:
    rows = store.latest_alerts(min_score=float(min_score), limit=int(limit))
    return [AlertItem(**r) for r in rows]


@app.post("/api/v1/simulate", response_model=SimulateResponse)
def simulate(
    events: int = Query(default=250, ge=1, le=5000),
    seed: int | None = Query(default=None, ge=0, le=2_147_483_647),
) -> SimulateResponse:
    generated = generate_stream(n=events, seed=seed)
    alerts_before = len(store.alerts)

    for tx in generated:
        score_payload = scorer.score(tx)
        risk_score = float(score_payload["risk_score"])
        record_score(
            score=risk_score,
            high_risk=risk_score >= settings.ALERT_MIN_SCORE,
        )

    alerts_after = len(store.alerts)
    set_graph_gauges(store.summary())
    return SimulateResponse(generated=len(generated), alerts_created=max(0, alerts_after - alerts_before))


@app.get("/metrics")
def metrics() -> Response:
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)
