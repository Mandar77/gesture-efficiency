# gesture-efficiency — reproducibility entrypoints.
# On Windows, run these via Git Bash, or call the underlying python directly.
# PY is the project venv interpreter; override with `make PY=python smoke`.

PY ?= .venv/Scripts/python.exe

.PHONY: help smoke repro-main test data-jester data-nvgesture clean

help:
	@echo "Targets:"
	@echo "  smoke        - 1-2 min end-to-end pipeline check (synthetic data, tiny model)"
	@echo "  test         - run unit + smoke tests (pytest)"
	@echo "  repro-main   - regenerate headline tables + Pareto figures from committed results"
	@echo "  data-jester  - print Jester download/prepare instructions"
	@echo "  data-nvgesture - print NVGesture download/prepare instructions"

smoke:
	$(PY) scripts/train.py --config configs/smoke.yaml

test:
	$(PY) -m pytest -q tests/

repro-main:
	$(PY) scripts/make_figures.py --results experiments --out paper/figures
	$(PY) scripts/make_tables.py --results experiments --out paper/tables.tex

data-jester:
	$(PY) src/data/download_data.py --dataset jester

data-nvgesture:
	$(PY) src/data/download_data.py --dataset nvgesture

clean:
	rm -rf **/__pycache__ .pytest_cache
