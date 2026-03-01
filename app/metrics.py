from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "aegis_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "aegis_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=(0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
SCORE_REQUESTS = Counter("aegis_score_requests_total", "Total score requests")
HIGH_RISK_ALERTS = Counter("aegis_high_risk_alerts_total", "High risk alerts generated")
RISK_SCORE_HIST = Histogram(
    "aegis_risk_score",
    "Distribution of risk scores",
    buckets=(0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.92, 1.0),
)
HEURISTIC_SCORE_HIST = Histogram(
    "aegis_heuristic_score",
    "Distribution of heuristic scores",
    buckets=(0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.92, 1.0),
)
MODEL_SCORE_HIST = Histogram(
    "aegis_model_score",
    "Distribution of model scores",
    buckets=(0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
)
MODEL_UPLIFT_DELTA_HIST = Histogram(
    "aegis_model_uplift_delta",
    "Final score minus heuristic score",
    buckets=(-0.3, -0.1, -0.05, 0.0, 0.01, 0.03, 0.06, 0.1, 0.2),
)
EVENTS_TOTAL = Gauge("aegis_events_total", "Total processed events")
ALERTS_TOTAL = Gauge("aegis_alerts_total", "Total alert rows")
NODES_TOTAL = Gauge("aegis_graph_nodes_total", "Total graph nodes")
EDGES_TOTAL = Gauge("aegis_graph_edges_total", "Total graph edges")


def record_http(method: str, path: str, status: int, duration_s: float) -> None:
    REQUEST_COUNT.labels(method=method, path=path, status=str(status)).inc()
    REQUEST_LATENCY.labels(method=method, path=path).observe(float(duration_s))


def record_score(
    score: float,
    high_risk: bool,
    *,
    heuristic_score: float | None = None,
    model_score: float | None = None,
) -> None:
    SCORE_REQUESTS.inc()
    RISK_SCORE_HIST.observe(float(score))
    if heuristic_score is not None:
        HEURISTIC_SCORE_HIST.observe(float(heuristic_score))
        MODEL_UPLIFT_DELTA_HIST.observe(float(score) - float(heuristic_score))
    if model_score is not None:
        MODEL_SCORE_HIST.observe(float(model_score))
    if high_risk:
        HIGH_RISK_ALERTS.inc()


def set_graph_gauges(summary: dict[str, int]) -> None:
    NODES_TOTAL.set(float(summary.get("nodes_total", 0)))
    EDGES_TOTAL.set(float(summary.get("edges_total", 0)))
    EVENTS_TOTAL.set(float(summary.get("events_total", 0)))
    ALERTS_TOTAL.set(float(summary.get("alerts_total", 0)))


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
