.PHONY: install test coverage mutation lint typecheck check train evaluate figures app all

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

figures:
	uv run python -m flowday.evaluation.plots

app:
	uv run streamlit run flowday/app/main.py

all: train evaluate
