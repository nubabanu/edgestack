# Convenience targets (POSIX make; on Windows run the commands directly).
PY ?= .venv/bin/python

.PHONY: install lint type test test-all demo demo-offline

install:
	$(PY) -m pip install -e .[dev,api]

lint:
	$(PY) -m ruff check src tests

type:
	$(PY) -m mypy

test:
	$(PY) -m pytest tests -q

test-all: lint type test

demo:
	$(PY) scripts/demo.py

demo-offline:
	$(PY) scripts/demo.py --offline
