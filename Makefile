.PHONY: install test lint format clean help preprocess train train-sample tune tune-sample evaluate preprocess-train-evaluate

help:
	@echo "Cibles disponibles :"
	@echo "  make preprocess    - prépare les données (train/test)"
	@echo "  make train         - entraîne le modèle"
	@echo "  make train-sample  - entraîne sur un échantillon (test rapide)"
	@echo "  make tune          - recherche d'hyperparamètres (long)"
	@echo "  make tune-sample   - recherche d'hyperparamètres sur échantillon (test rapide)"
	@echo "  make evaluate      - évalue le modèle sur le test"

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

mlflow:
	mlflow ui

requirements:
	uv export -o requirements.txt

preprocess:
	python -m src.data.preprocess

train:
	python -m src.training.train

train-sample:
	python -m src.training.train --sample

tune:
	python -m src.training.train --tune

tune-sample:
	python -m src.training.train --tune --sample

evaluate:
	python -m src.training.evaluate

preprocess-train-evaluate:
	python -m src.data.preprocess && python -m src.training.train && python -m src.training.evaluate