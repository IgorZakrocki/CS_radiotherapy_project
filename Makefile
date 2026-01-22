VENV := .venv
PYTHON := python3

ifeq ($(OS),Windows_NT)
    PY := $(VENV)/Scripts/python
    PIP := $(VENV)/Scripts/pip
    NBCONVERT := $(PY) -m nbconvert
else
    PY := $(VENV)/bin/python
    PIP := $(VENV)/bin/pip
    NBCONVERT := $(VENV)/bin/jupyter nbconvert
endif

SIM_DIR := simulations
NOTEBOOKS := $(wildcard $(SIM_DIR)/*.ipynb)

.PHONY: install run clean

install:
	@$(PYTHON) -m venv $(VENV)
	@$(PY) -m pip install --upgrade pip setuptools wheel
	@$(PIP) install -r requirements.txt

run:
	@echo "Uruchamianie symulacji z katalogu $(SIM_DIR)..."
	@$(foreach nb, $(NOTEBOOKS), \
		echo "Przetwarzanie: $(nb)"; \
		$(NBCONVERT) --to notebook --execute --inplace $(nb) || exit 1; \
	)
	@echo "Wszystkie symulacje zakończone."

clean:
	@echo "Czyszczenie wyników w notebookach..."
	@$(foreach nb, $(NOTEBOOKS), \
		$(NBCONVERT) --ClearOutputPreprocessor.enabled=True --inplace $(nb); \
	)
	@echo "Gotowe."