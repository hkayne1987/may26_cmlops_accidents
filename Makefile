.PHONY: install test lint format clean train

install:
	uv sync

install-dev:
	uv sync --extra dev

test:
	pytest tests/ -v

lint:
	ruff check src/ml_project

format:
	black src/ml_project tests/

typecheck:
	mypy src/ml_project

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache

train:
	python -m ml_project train --config configs/train.yaml

mlflow:
	mlflow ui

requirements:
	uv export -o requirements.txt