#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8090}"

echo "[healthtest] base_url=${BASE_URL}"

curl -fsS "${BASE_URL}/health" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["status"]=="ok"; print("[ok] /health")'
curl -fsS "${BASE_URL}/api/v1/graph/summary" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert "events_total" in d; print("[ok] /api/v1/graph/summary")'

echo "[healthtest] PASS"
