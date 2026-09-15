.PHONY: help install dev test lint validate experiments serve clean

PY ?= python3
export PYTHONPATH := src:.

help:
	@echo "install      install the package"
	@echo "dev          install with API, test and plotting extras"
	@echo "test         run the test suite"
	@echo "lint         run ruff"
	@echo "validate     run the full validation suite -> results/validation.json"
	@echo "experiments  regenerate every figure and results file"
	@echo "serve        start the dashboard on http://127.0.0.1:8000"

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[api,dev,plots]"

test:
	$(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check src api tests benchmarks

validate:
	$(PY) -m physim.cli validate --out results/validation.json

experiments:
	$(PY) -m physim.experiments.run_all

serve:
	$(PY) -m physim.cli serve --port 8000

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ build dist *.egg-info
