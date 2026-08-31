PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python

.PHONY: setup test test-stdlib smoke env-doctor env-demo clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip setuptools wheel
	$(VENV_PYTHON) -m pip install -e ".[dev]"

test:
	$(VENV_PYTHON) -m pytest

test-stdlib:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

smoke:
	$(VENV)/bin/benchmark-toolbox run --config configs/examples/smoke.json

env-doctor:
	$(VENV)/bin/benchmark-toolbox env doctor

# End-to-end demo of the environment manager: creates a venv env and runs through it.
env-demo:
	$(VENV)/bin/benchmark-toolbox env prepare --env configs/environments/echo.yaml
	$(VENV)/bin/benchmark-toolbox run --config configs/examples/echo_local.yaml

clean:
	rm -rf artifacts .pytest_cache .benchmark_toolbox
