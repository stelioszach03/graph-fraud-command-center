from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.services.graph_store import GraphStore, TxEvent
from app.services.scoring import FraudScoringService


def _tx(
    *,
    tx_id: str,
    sender_id: str,
    receiver_id: str,
    amount: float,
    channel: str,
    country_from: str,
    country_to: str,
) -> TxEvent:
    return TxEvent(
        tx_id=tx_id,
        sender_id=sender_id,
        receiver_id=receiver_id,
        amount=amount,
        currency="USD",
        channel=channel,
        country_from=country_from,
        country_to=country_to,
        timestamp_utc=datetime.now(timezone.utc),
    )


def test_cold_start_low_value_stays_below_alert_threshold():
    scorer = FraudScoringService(store=GraphStore(), model_path="/tmp/not-found.pt", alert_min_score=0.75)
    out = scorer.score(
        _tx(
            tx_id="T1",
            sender_id="ACC_A1",
            receiver_id="ACC_A2",
            amount=45.0,
            channel="ach",
            country_from="US",
            country_to="US",
        )
    )

    assert out["risk_score"] < 0.75


def test_cold_start_high_risk_pattern_reaches_alert_threshold():
    scorer = FraudScoringService(store=GraphStore(), model_path="/tmp/not-found.pt", alert_min_score=0.75)
    out = scorer.score(
        _tx(
            tx_id="T2",
            sender_id="ACC_S1",
            receiver_id="ACC_S2",
            amount=24000.0,
            channel="crypto",
            country_from="GB",
            country_to="AE",
        )
    )

    assert out["risk_score"] >= 0.75
    assert out["risk_band"] in {"high", "critical"}


def test_uplift_only_fusion_does_not_lower_heuristic(monkeypatch):
    scorer = FraudScoringService(
        store=GraphStore(),
        model_path="/tmp/not-found.pt",
        alert_min_score=0.75,
        model_blend_weight=0.5,
        model_uplift_only=True,
    )
    monkeypatch.setattr(scorer, "_model_score", lambda _: 0.01)

    out = scorer.score(
        _tx(
            tx_id="T3",
            sender_id="ACC_M1",
            receiver_id="ACC_M2",
            amount=12000.0,
            channel="wire",
            country_from="US",
            country_to="CY",
        )
    )

    assert out["risk_score"] >= out["heuristic_score"]
    assert out["risk_score"] == pytest.approx(out["heuristic_score"], abs=1e-9)
