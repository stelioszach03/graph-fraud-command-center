from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.metrics import (
    record_http,
    record_score,
    render_metrics,
    set_graph_gauges,
)
from app.schemas import (
    AlertItem,
    GraphNeighborhood,
    GraphSummary,
    HealthResponse,
    ResetResponse,
    RootResponse,
    ScoreRequest,
    ScoreResponse,
    SimulateResponse,
)
from app.services.graph_store import GraphStore, TxEvent
from app.services.scoring import FraudScoringService
from app.services.simulator import generate_stream
from app.settings import get_settings

settings = get_settings()

app = FastAPI(
    title="Aegis Graph Fraud GNN",
    description=(
        "Graph-native fraud detection API. Hybrid heuristic + PyTorch GNN "
        "scoring, explainable reason codes, and operator-grade observability."
    ),
    version=settings.APP_VERSION,
    contact={
        "name": "Stelios Zacharioudakis",
        "url": "https://stelioszach.com",
    },
    license_info={"name": "MIT"},
)

# Permissive CORS for public demo; tighten in production via env allowlist.
_origins = [o.strip() for o in (settings.CORS_ALLOW_ORIGINS or "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)

store = GraphStore()
scorer = FraudScoringService(
    store=store,
    model_path=settings.MODEL_PATH,
    alert_min_score=settings.ALERT_MIN_SCORE,
    model_blend_weight=settings.MODEL_BLEND_WEIGHT,
    model_uplift_only=settings.MODEL_UPLIFT_ONLY,
    amount_z_warmup_events=settings.AMOUNT_Z_WARMUP_EVENTS,
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
    # Expose latency in a standard debug header for operators.
    response.headers["X-Response-Time-Ms"] = f"{dt * 1000:.2f}"
    response.headers["X-Service"] = "aegis-graph-fraud-gnn"
    return response


@app.get("/", response_model=RootResponse, tags=["meta"])
def root() -> RootResponse:
    summary = store.summary()
    return RootResponse(
        service="aegis-graph-fraud-gnn",
        status="ok",
        version=settings.APP_VERSION,
        env=settings.APP_ENV,
        docs_url="/docs",
        endpoints=[
            "GET  /health",
            "GET  /metrics",
            "POST /api/v1/score",
            "POST /api/v1/simulate",
            "POST /api/v1/reset",
            "GET  /api/v1/graph/summary",
            "GET  /api/v1/graph/neighborhood/{node_id}",
            "GET  /api/v1/alerts",
        ],
        summary=GraphSummary(**summary),
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    summary = store.summary()
    return HealthResponse(
        status="ok",
        service="aegis-graph-fraud-gnn",
        version=settings.APP_VERSION,
        env=settings.APP_ENV,
        timestamp_utc=datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        model_loaded=scorer.model is not None,
        model_blend_weight=settings.MODEL_BLEND_WEIGHT,
        alert_min_score=settings.ALERT_MIN_SCORE,
        nodes_total=summary["nodes_total"],
        edges_total=summary["edges_total"],
        events_total=summary["events_total"],
        alerts_total=summary["alerts_total"],
    )


@app.post("/api/v1/score", response_model=ScoreResponse, tags=["scoring"])
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
        heuristic_score=float(result["heuristic_score"]),
        model_score=(float(result["model_score"]) if result["model_score"] is not None else None),
    )
    set_graph_gauges(store.summary())
    return ScoreResponse(**result)


@app.get("/api/v1/graph/summary", response_model=GraphSummary, tags=["graph"])
def graph_summary() -> GraphSummary:
    summary = store.summary()
    set_graph_gauges(summary)
    return GraphSummary(**summary)


@app.get(
    "/api/v1/graph/neighborhood/{node_id}",
    response_model=GraphNeighborhood,
    tags=["graph"],
)
def graph_neighborhood(
    node_id: str,
    depth: int = Query(default=1, ge=1, le=2),
    limit: int = Query(default=40, ge=1, le=200),
) -> GraphNeighborhood:
    """Return a small local subgraph around `node_id` for investigation UIs."""
    data = store.neighborhood(node_id=node_id, depth=int(depth), limit=int(limit))
    if data is None:
        raise HTTPException(status_code=404, detail=f"node '{node_id}' not found")
    return GraphNeighborhood(**data)


@app.get("/api/v1/alerts", response_model=list[AlertItem], tags=["alerts"])
def alerts(
    min_score: float = Query(default=settings.ALERT_MIN_SCORE, ge=0.0, le=1.0),
    limit: int = Query(default=25, ge=1, le=200),
) -> list[AlertItem]:
    rows = store.latest_alerts(min_score=float(min_score), limit=int(limit))
    return [AlertItem(**r) for r in rows]


@app.post("/api/v1/simulate", response_model=SimulateResponse, tags=["scoring"])
def simulate(
    events: int = Query(default=250, ge=1, le=5000),
    seed: int | None = Query(default=None, ge=0, le=2_147_483_647),
) -> SimulateResponse:
    t0 = perf_counter()
    generated = generate_stream(n=events, seed=seed)
    alerts_before = len(store.alerts)
    critical_before = sum(1 for a in store.alerts if a.get("risk_band") == "critical")

    for tx in generated:
        score_payload = scorer.score(tx)
        risk_score = float(score_payload["risk_score"])
        record_score(
            score=risk_score,
            high_risk=risk_score >= settings.ALERT_MIN_SCORE,
            heuristic_score=float(score_payload["heuristic_score"]),
            model_score=(
                float(score_payload["model_score"]) if score_payload["model_score"] is not None else None
            ),
        )

    alerts_after = len(store.alerts)
    critical_after = sum(1 for a in store.alerts if a.get("risk_band") == "critical")
    set_graph_gauges(store.summary())
    elapsed_ms = round((perf_counter() - t0) * 1000.0, 2)

    return SimulateResponse(
        generated=len(generated),
        alerts_created=max(0, alerts_after - alerts_before),
        critical_created=max(0, critical_after - critical_before),
        elapsed_ms=elapsed_ms,
    )


@app.post("/api/v1/reset", response_model=ResetResponse, tags=["scoring"])
def reset_state() -> ResetResponse:
    """Flush graph, events, and alerts. Useful for demos and deterministic runs."""
    cleared_nodes = store.graph.number_of_nodes()
    cleared_edges = store.graph.number_of_edges()
    cleared_events = len(store.events)
    cleared_alerts = len(store.alerts)
    store.clear()
    set_graph_gauges(store.summary())
    return ResetResponse(
        cleared_nodes=int(cleared_nodes),
        cleared_edges=int(cleared_edges),
        cleared_events=int(cleared_events),
        cleared_alerts=int(cleared_alerts),
    )


@app.get("/metrics", tags=["meta"])
def metrics() -> Response:
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)
