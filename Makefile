.PHONY: install test lint typecheck check train evaluate figures app all

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy flowday

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
