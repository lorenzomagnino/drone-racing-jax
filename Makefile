.PHONY: install test lint format

install:
	uv sync --group dev

test:
	uv run pytest tests/

lint:
	uv run ruff check .

format:
	uv run ruff check --select I --fix .
	uv run black .
