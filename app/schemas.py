from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator


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


class ScoreResponse(BaseModel):
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
