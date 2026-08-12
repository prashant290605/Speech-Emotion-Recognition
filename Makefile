# Thin wrapper over the `ser` CLI, which is the canonical interface.
# Every target delegates, so there is exactly one implementation of each step.
#
# `make` is frequently absent on Windows. The equivalent there is:
#     python -m ser.cli <command>
# with PYTHONPATH=src, or `pip install -e .` to get the `ser` entrypoint.

PYTHON ?= python
SER    ?= PYTHONPATH=src $(PYTHON) -m ser.cli
CONFIG ?= configs/default.yaml

.PHONY: help install test smoke inventory schema \
        check-refs manifest splits dataset-stats extract verify-cache \
        baselines run-grid select label-shift figures tables verify clean-pyc

help:
	@$(SER) --help

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

# -- Phase 0 ---------------------------------------------------------------
test:
	$(PYTHON) -m pytest

smoke:
	$(SER) -c $(CONFIG) smoke

inventory:
	$(SER) -c $(CONFIG) inventory

schema:
	$(SER) -c $(CONFIG) schema

# -- Phase 1 ---------------------------------------------------------------
check-refs:
	$(SER) -c $(CONFIG) check-refs

# -- Phase 2 ---------------------------------------------------------------
manifest:
	$(SER) -c $(CONFIG) manifest

splits:
	$(SER) -c $(CONFIG) splits

dataset-stats:
	$(SER) -c $(CONFIG) dataset-stats

# -- Phase 3 ---------------------------------------------------------------
extract:
	$(SER) -c $(CONFIG) extract

verify-cache:
	$(SER) -c $(CONFIG) verify-cache

# -- Phase 4 ---------------------------------------------------------------
baselines:
	$(SER) -c $(CONFIG) baselines

# -- Phase 7 ---------------------------------------------------------------
run-grid:
	$(SER) -c $(CONFIG) run-grid

# -- Phase 8 ---------------------------------------------------------------
select:
	$(SER) -c $(CONFIG) select

# -- Phase 9 ---------------------------------------------------------------
label-shift:
	$(SER) -c $(CONFIG) label-shift

# -- Phase 10 --------------------------------------------------------------
figures:
	$(SER) -c $(CONFIG) figures

# -- Phase 11 --------------------------------------------------------------
tables:
	$(SER) -c $(CONFIG) tables

# Consolidated leakage + reproducibility assertions. Named in the paper's
# reproducibility statement.
verify:
	$(SER) -c $(CONFIG) verify

clean-pyc:
	$(PYTHON) -c "import pathlib,shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
