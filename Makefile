.PHONY: install test lint format clean help preprocess train train-sample tune tune-sample evaluate preprocess-train-evaluate mlflow-ui-local

help:
	@echo "Available targets:"
	@echo "  make preprocess        - prepare the data (train/test)"
	@echo "  make train             - train the model"
	@echo "  make train-sample      - train on a subsample (quick test)"
	@echo "  make tune              - hyperparameter search (long)"
	@echo "  make tune-sample       - hyperparameter search on a subsample (quick test)"
	@echo "  make evaluate          - evaluate the model on the test set"
	@echo "  make mlflow-ui-local   - open the local MLflow UI (./mlruns); remote tracking runs on DagsHub"

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

# Experiment tracking runs on DagsHub, which serves its own UI -- there is no
# local server to start. This target only opens a local UI on ./mlruns, which
# is useful when training offline with the file:./mlruns fallback.
mlflow-ui-local:
	mlflow ui --backend-store-uri file:./mlruns

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