from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _label_rule(tx) -> int:
    suspicious_ring = 1.0 if tx.sender_id.startswith("RING_") or tx.receiver_id.startswith("RING_") else 0.0
    high_amount = 1.0 if tx.amount >= 9000 else 0.0
    cross_border = 1.0 if tx.country_from != tx.country_to else 0.0
    high_risk_channel = 1.0 if tx.channel in {"crypto", "wire", "cash"} else 0.0
    score = 0.45 * suspicious_ring + 0.25 * high_amount + 0.20 * cross_border + 0.10 * high_risk_channel
    return int(score >= 0.45)


def _quality_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (scores >= threshold).astype(np.int32)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())

    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float((2.0 * precision * recall) / max(1e-9, precision + recall))

    return {
        "threshold": round(float(threshold), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "positive_rate": round(float(pred.mean()), 6),
    }


def main() -> None:
    from app.services.graph_store import GraphStore
    from app.services.scoring import FraudScoringService
    from app.services.simulator import generate_stream

    parser = argparse.ArgumentParser(description="Evaluate scorer quality on deterministic synthetic traffic")
    parser.add_argument("--events", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model-path", type=str, default="artifacts/models/edge_model.pt")
    parser.add_argument("--alert-threshold", type=float, default=0.75)
    parser.add_argument("--out", type=str, default="benchmarks/quality_latest.json")
    args = parser.parse_args()

    stream = generate_stream(n=max(500, int(args.events)), seed=int(args.seed))
    store = GraphStore(max_events=max(6000, len(stream) + 1000))
    scorer = FraudScoringService(
        store=store,
        model_path=args.model_path,
        alert_min_score=float(args.alert_threshold),
    )

    labels: list[int] = []
    risk_scores: list[float] = []
    heuristic_scores: list[float] = []
    model_scores: list[float] = []

    for tx in stream:
        labels.append(_label_rule(tx))
        out = scorer.score(tx)
        risk_scores.append(float(out["risk_score"]))
        heuristic_scores.append(float(out["heuristic_score"]))
        model_scores.append(float(out["model_score"] if out["model_score"] is not None else 0.0))

    y = np.asarray(labels, dtype=np.int32)
    risk = np.asarray(risk_scores, dtype=np.float32)
    heur = np.asarray(heuristic_scores, dtype=np.float32)
    mdl = np.asarray(model_scores, dtype=np.float32)

    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    quality = [_quality_at_threshold(y, risk, thr) for thr in thresholds]

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "events_total": int(len(stream)),
        "label_positive_ratio": round(float(y.mean()), 6),
        "risk_mean": round(float(risk.mean()), 6),
        "risk_std": round(float(risk.std()), 6),
        "heuristic_mean": round(float(heur.mean()), 6),
        "model_mean": round(float(mdl.mean()), 6),
        "quality_by_threshold": quality,
    }

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)

    print(json.dumps(out, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
