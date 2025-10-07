from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


def _payload(i: int) -> dict:
    channels = ["wire", "ach", "card", "crypto", "cash"]
    countries = ["US", "US", "US", "GB", "CY", "AE", "SG"]
    sender = f"ACC_{random.randint(1, 450):04d}"
    receiver = f"ACC_{random.randint(1, 450):04d}"
    while receiver == sender:
        receiver = f"ACC_{random.randint(1, 450):04d}"

    return {
        "tx_id": f"BCH-{i:07d}",
        "sender_id": sender,
        "receiver_id": receiver,
        "amount": round(random.lognormvariate(7.1, 0.95), 2),
        "currency": "USD",
        "channel": random.choice(channels),
        "country_from": random.choice(countries),
        "country_to": random.choice(countries),
    }


def _post_json(url: str, payload: dict, timeout: float) -> tuple[int, dict | None]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            return int(resp.status), data
    except urllib.error.HTTPError as e:
        return int(e.code), None
    except Exception:
        return 0, None


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int((len(s) - 1) * q)
    return float(s[idx])


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Aegis Graph Fraud GNN score endpoint")
    parser.add_argument("--base-url", default="http://localhost:8090")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=4.0)
    parser.add_argument("--out", default="benchmarks/latest.json")
    args = parser.parse_args()

    random.seed(42)
    n = max(10, int(args.requests))
    url = args.base_url.rstrip("/") + "/api/v1/score"

    lat_ms: list[float] = []
    ok = 0
    errors = 0
    high_risk = 0

    t0_all = time.perf_counter()
    for i in range(n):
        payload = _payload(i)
        t0 = time.perf_counter()
        status, data = _post_json(url, payload, timeout=float(args.timeout))
        dt_ms = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(float(dt_ms))

        if status == 200 and data is not None:
            ok += 1
            if float(data.get("risk_score", 0.0)) >= 0.85:
                high_risk += 1
        else:
            errors += 1

    elapsed = time.perf_counter() - t0_all
    rps = (ok / elapsed) if elapsed > 0 else 0.0

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "base_url": args.base_url,
        "requests_total": n,
        "ok": ok,
        "errors": errors,
        "success_rate": round((ok / n) if n else 0.0, 6),
        "throughput_rps": round(rps, 3),
        "latency_ms": {
            "mean": round(statistics.mean(lat_ms), 3) if lat_ms else 0.0,
            "p50": round(_pct(lat_ms, 0.50), 3),
            "p95": round(_pct(lat_ms, 0.95), 3),
            "p99": round(_pct(lat_ms, 0.99), 3),
            "max": round(max(lat_ms), 3) if lat_ms else 0.0,
        },
        "high_risk_ratio": round((high_risk / ok) if ok else 0.0, 6),
    }

    out_path = args.out
    out_dir = out_path.rsplit("/", 1)[0] if "/" in out_path else "."
    if out_dir and out_dir != ".":
        import os

        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
