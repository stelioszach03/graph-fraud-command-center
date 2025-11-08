from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ─────────────────────────────────────────────────────────────
# Requests
# ─────────────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    tx_id: str | None = None
    sender_id: str = Field(min_length=2, max_length=64)
    receiver_id: str = Field(min_length=2, max_length=64)
    amount: float = Field(gt=0)
    currency: str = Field(default="USD", min_length=3, max_length=8)
    channel: Literal["wire", "ach", "card", "crypto", "cash"] = "wire"
    country_from: str = Field(default="US", min_length=2, max_length=3)
    country_to: str = Field(default="US", min_length=2, max_length=3)
    timestamp_utc: datetime | None = None

    @field_validator("timestamp_utc", mode="before")
    @classmethod
    def normalize_ts(cls, v):
        if v is None:
            return datetime.now(timezone.utc)
        return v


# ─────────────────────────────────────────────────────────────
# Responses
# ─────────────────────────────────────────────────────────────

class ScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    tx_id: str
    risk_score: float
    risk_band: Literal["low", "medium", "high", "critical"]
    model_score: float | None = None
    heuristic_score: float
    reasons: list[str]
    top_features: dict[str, float]
    processed_at_utc: str


class GraphSummary(BaseModel):
    nodes_total: int
    edges_total: int
    events_total: int
    alerts_total: int
    high_risk_last_hour: int


class AlertItem(BaseModel):
    tx_id: str
    risk_score: float
    risk_band: str
    sender_id: str
    receiver_id: str
    amount: float
    timestamp_utc: str
    reasons: list[str]


class SimulateResponse(BaseModel):
    generated: int
    alerts_created: int
    critical_created: int = 0
    elapsed_ms: float = 0.0


class ResetResponse(BaseModel):
    cleared_nodes: int
    cleared_edges: int
    cleared_events: int
    cleared_alerts: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    service: str
    version: str
    env: str
    timestamp_utc: str
    model_loaded: bool
    model_blend_weight: float
    alert_min_score: float
    nodes_total: int
    edges_total: int
    events_total: int
    alerts_total: int


class RootResponse(BaseModel):
    service: str
    status: str
    version: str
    env: str
    docs_url: str
    endpoints: list[str]
    summary: GraphSummary


# ─────────────────────────────────────────────────────────────
# Graph neighborhood
# ─────────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    id: str
    in_count: int
    out_count: int
    in_total: float
    out_total: float
    is_focus: bool = False
    first_seen_utc: str | None = None
    last_seen_utc: str | None = None


class GraphEdge(BaseModel):
    source: str
    target: str
    count: int
    total: float
    last_ts_utc: str | None = None


class GraphNeighborhood(BaseModel):
    focus: str
    depth: int
    nodes: list[GraphNode]
    edges: list[GraphEdge]
