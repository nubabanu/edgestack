# Convenience targets (POSIX make; on Windows run the commands directly).
PY ?= .venv/bin/python

.PHONY: install format-check lint package type test test-all demo demo-offline

install:
	$(PY) -m pip install -e .[dev,api,research]

format-check:
	$(PY) -m ruff format --check src tests scripts

lint:
	$(PY) -m ruff check src tests scripts

type:
	$(PY) -m mypy src/edgestack

package:
	$(PY) -m build
	$(PY) -m twine check dist/*

test:
	$(PY) -m pytest tests -q

test-all: format-check lint type test package

demo:
	$(PY) scripts/demo.py

demo-offline:
	$(PY) scripts/demo.py --offline
