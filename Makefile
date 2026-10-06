.PHONY: install test lint format clean help preprocess train train-sample tune tune-sample evaluate preprocess-train-evaluate mlflow-ui-local users replay drift \
	setup data api health monitoring up airflow airflow-password down clean-state clean-images

help:
	@echo "Stack (see README, Quick start):"
	@echo "  make setup             - install dependencies, create .env from env.example if missing"
	@echo "  make data              - pull the raw data from DagsHub (DVC) and preprocess it (~1 min)"
	@echo "  make up                - start the API and the monitoring stack"
	@echo "  make api / monitoring  - start only one of them"
	@echo "  make health            - show the API status and the model version it serves"
	@echo "  make airflow           - start Airflow on http://localhost:8080"
	@echo "  make airflow-password  - print the Airflow admin password"
	@echo "  make down              - stop and remove every container, keep all data"
	@echo "  make clean-state       - down, then erase metrics, dashboards state, Airflow history, drift reports and logged predictions"
	@echo "  make clean-images      - down, then remove the project's Docker images"
	@echo ""
	@echo "Pipeline:"
	@echo "  make preprocess        - prepare the data (train/test)"
	@echo "  make train             - train the model"
	@echo "  make train-sample      - train on a subsample (quick test)"
	@echo "  make tune              - hyperparameter search (long)"
	@echo "  make tune-sample       - hyperparameter search on a subsample (quick test)"
	@echo "  make evaluate          - evaluate the model on the test set"
	@echo "  make mlflow-ui-local   - open the local MLflow UI (./mlruns); remote tracking runs on DagsHub"
	@echo "  make users ARGS=...    - manage API accounts, e.g. ARGS=\"create alice --role admin\""
	@echo "  make replay            - send real accidents to the API, e.g. SCENARIO=motorway ROWS=500"
	@echo "  make drift             - compare logged predictions with the training data (Evidently)"

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
# Drift detection demo. The API gets no real traffic, so replay real BAAC
# accidents through it; a scenario other than "normal" makes the inputs drift.
# Scenarios: normal, motorway, pedestrians, night, two_wheelers.
# Needs API_USERNAME / API_PASSWORD, read from .env.
SCENARIO ?= normal
ROWS ?= 500
replay:
	set -a; . ./.env; set +a; \
	uv run python -m src.monitoring.replay --scenario $(SCENARIO) --rows $(ROWS)

# Same job as the daily Airflow DAG. Needs the monitoring profile up so the
# result reaches the Pushgateway.
drift:
	docker-compose --profile drift run --rm drift

# --- Stack -------------------------------------------------------------------
# Shortcuts over docker-compose, so the whole stack runs with make alone.
PROFILES = --profile monitoring --profile airflow --profile autoupdate

setup:
	uv sync
	@test -f .env || (cp env.example .env && echo "Created .env: fill in the DagsHub credentials, JWT_SECRET_KEY and API_PASSWORD")

# Pulls only the raw CSVs, then rebuilds data/processed from them, like
# retrain.yml does. The data/processed recorded in dvc.lock predates later
# preprocessing changes, so pulling it would bring back stale parquet files.
data:
	docker-compose --profile dvc run --rm dvc pull data/raw.dvc
	docker-compose --profile training run --rm training uv run python -m src.data.preprocess

api:
	docker-compose up -d api

health:
	@curl -s http://localhost:8000/health; echo

monitoring:
	docker-compose --profile monitoring up -d

up: api monitoring

airflow:
	docker-compose --profile airflow up -d

# Standalone Airflow generates the admin password when it starts.
airflow-password:
	@docker-compose --profile airflow logs airflow | grep "Password for user"

down:
	docker-compose $(PROFILES) down

# Erases what the stack recorded at run time. Keeps the data, the models and
# the API accounts. The directories stay, only their content goes, because
# the containers expect to find them.
clean-state: down
	find prometheus_data grafana_data airflow_data/db airflow_data/logs -mindepth 1 -delete 2>/dev/null || true
	rm -f reports/drift/*.html reports/drift/*.json data/predictions.db

# Removes the images the stack uses; the next start pulls them again.
clean-images:
	docker-compose $(PROFILES) down --rmi all
