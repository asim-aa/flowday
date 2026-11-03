.PHONY: site install test coverage mutation lint typecheck check train evaluate figures app all synth-log retrain-from-log sensitivity

install:
	uv sync

test:
	uv run pytest

# line coverage of the package by the test suite
coverage:
	uv run pytest --cov=flowday --cov-report=term-missing:skip-covered

# inject known bugs into the scheduler; every non-equivalent one must be caught by the tests
mutation:
	uv run python scripts/mutation_check.py

lint:
	uv run ruff check .

typecheck:
	uv run mypy flowday tests

check: lint typecheck test

# generate synthetic data, train + tune models, write artifacts/ and reports/model_metrics.json
train:
	uv run python -m flowday.models.train

# run the scheduler end to end in the simulator with every estimator; writes reports/ and figures
evaluate:
	uv run python -m flowday.evaluation.run

# ablation, generator sweep + seeds, and the caution frontier (takes ~15 minutes; needs `make train`)
sensitivity:
	uv run python -m flowday.evaluation.sensitivity

figures:
	uv run python -m flowday.evaluation.plots

app:
	uv run streamlit run flowday/app/main.py

# write synthetic days into a local event log (so retrain-from-log can be tried without real data)
synth-log:
	uv run python -m flowday.eventlog.simulate --db data/synthetic_log.db

# retrain the estimators from an event log: make retrain-from-log LOG=data/flowday.db
LOG ?= data/synthetic_log.db
retrain-from-log:
	uv run python -m flowday.models.retrain --db $(LOG)

# refresh the figures used by the github.io page (docs/); the page's numbers are copied by hand from the README
site:
	cp reports/figures/fig3_schedule_outcomes.png reports/figures/fig4_caution_frontier.png docs/figures/

all: train evaluate
