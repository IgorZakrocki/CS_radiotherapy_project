VENV := .venv
PYTHON := python3

ifeq ($(OS),Windows_NT)
	PY := $(VENV)/Scripts/python
	PIP := $(VENV)/Scripts/pip
else
	PY := $(VENV)/bin/python
	PIP := $(VENV)/bin/pip
endif

.PHONY: install

install:
	@$(PYTHON) -m venv $(VENV)
	@$(PY) -m pip install --upgrade pip setuptools wheel
	@$(PIP) install -r requirements.txt
