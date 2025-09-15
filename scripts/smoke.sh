#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8090}"

echo "[smoke] base_url=${BASE_URL}"

curl -fsS "${BASE_URL}/health" >/dev/null
echo "[ok] /health"

curl -fsS -X POST "${BASE_URL}/api/v1/score" \
  -H 'Content-Type: application/json' \
  -d '{
    "sender_id":"ACC_0001",
    "receiver_id":"ACC_0002",
    "amount":1950.50,
    "currency":"USD",
    "channel":"wire",
    "country_from":"US",
    "country_to":"US"
  }' >/tmp/score_out.json

grep -q 'risk_score' /tmp/score_out.json
echo "[ok] /api/v1/score"

curl -fsS -X POST "${BASE_URL}/api/v1/simulate?events=120" >/dev/null
echo "[ok] /api/v1/simulate"

curl -fsS "${BASE_URL}/api/v1/graph/summary" >/tmp/summary_out.json
grep -q 'events_total' /tmp/summary_out.json
echo "[ok] /api/v1/graph/summary"

echo "[smoke] PASS"
