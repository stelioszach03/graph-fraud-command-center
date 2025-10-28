PY := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYBIN := $(VENV)/bin/python

.PHONY: setup run lint test train smoke benchmark eval

setup:
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(VENV)/bin/uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload

lint:
	$(VENV)/bin/ruff check .

test:
	$(PYBIN) -m pytest

train:
	$(PYBIN) scripts/train_synthetic.py --epochs 10 --samples 15000

smoke:
	bash scripts/smoke.sh

benchmark:
	$(PYBIN) scripts/benchmark.py --base-url http://localhost:8090 --requests 2000 --out benchmarks/latest.json

eval:
	$(PYBIN) scripts/evaluate_quality.py --events 3000 --seed 123 --out benchmarks/quality_latest.json
