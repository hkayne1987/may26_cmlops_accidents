.PHONY: install test lint format clean help preprocess train train-sample tune tune-sample evaluate preprocess-train-evaluate mlflow-ui-local users

help:
	@echo "Available targets:"
	@echo "  make preprocess        - prepare the data (train/test)"
	@echo "  make train             - train the model"
	@echo "  make train-sample      - train on a subsample (quick test)"
	@echo "  make tune              - hyperparameter search (long)"
	@echo "  make tune-sample       - hyperparameter search on a subsample (quick test)"
	@echo "  make evaluate          - evaluate the model on the test set"
	@echo "  make mlflow-ui-local   - open the local MLflow UI (./mlruns); remote tracking runs on DagsHub"
	@echo "  make users ARGS=...    - manage API accounts, e.g. ARGS=\"create alice --role admin\""

install:
	uv sync

install-dev:
	uv sync

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache

# Experiment tracking runs on DagsHub, which serves its own UI -- there is no
# local server to start. This target only opens a local UI on ./mlruns, which
# is useful when training offline with the file:./mlruns fallback.
mlflow-ui-local:
	uv run mlflow ui --backend-store-uri file:./mlruns

requirements:
	uv export -o requirements.txt

# Manage API accounts, e.g. make users ARGS="create alice --role admin"
# Needs JWT_SECRET_KEY from .env, like the API itself.
users:
	uv run python -m src.api.manage_users $(ARGS)

preprocess:
	uv run python -m src.data.preprocess

train:
	uv run python -m src.training.train

train-sample:
	uv run python -m src.training.train --sample

tune:
	uv run python -m src.training.train --tune

tune-sample:
	uv run python -m src.training.train --tune --sample

evaluate:
	uv run python -m src.training.evaluate

preprocess-train-evaluate:
	uv run python -m src.data.preprocess && \
	uv run python -m src.training.train && \
	uv run python -m src.training.evaluate