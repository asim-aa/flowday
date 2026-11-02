.PHONY: install test lint typecheck check data train evaluate app all

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy flowday

check: lint typecheck test
