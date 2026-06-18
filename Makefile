.PHONY: install test lint format clean train

install:
	uv sync

install-dev:
	uv sync

test:
	pytest tests/ -v

lint:
	ruff check src/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache

train:
	python -m training train --data configs/train.yaml

mlflow:
	mlflow ui

requirements:
	uv export -o requirements.txt

preprocess:
	python -m src.data.preprocess