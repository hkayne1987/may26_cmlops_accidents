"""Daily drift check on the predictions served by the API.

Runs src/monitoring/detection.py in the training image: it compares the
requests logged over the last days with the training data and pushes the
result to the Prometheus Pushgateway, where Grafana shows it.

Scheduled, unlike ml_pipeline: this DAG only measures and never touches the
served model, so running it unattended is safe. A drift alert calls for a
human look, not an automatic retrain: retraining needs new labelled data,
which BAAC releases once a year.
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# See ml_pipeline.py: task containers are siblings of Airflow on the host, so
# their bind mounts use host paths.
HOST_PROJECT_DIR = os.environ["HOST_PROJECT_DIR"]
TRAINING_IMAGE = os.environ.get("TRAINING_IMAGE", "baac-training:latest")
# The compose network, named in docker-compose.yml, where the Pushgateway is.
COMPOSE_NETWORK = os.environ.get("COMPOSE_NETWORK", "baac")

DRIFT_ENV = {
    "PUSHGATEWAY_URL": os.environ.get("PUSHGATEWAY_URL", "http://pushgateway:9091"),
    # Passed through so the thresholds can be tuned from .env.
    **{
        name: os.environ[name]
        for name in (
            "DRIFT_WINDOW_DAYS",
            "DRIFT_MIN_ROWS",
            "DRIFT_REFERENCE_ROWS",
            "DRIFT_ALERT_SHARE",
        )
        if name in os.environ
    },
}

with DAG(
    dag_id="drift_monitoring",
    description="Compare served predictions with the training data (Evidently)",
    default_args={
        "owner": "mlops-team",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "monitoring"],
) as dag:
    DockerOperator(
        task_id="detect_drift",
        image=TRAINING_IMAGE,
        command="uv run python -m src.monitoring.detection",
        mounts=[
            # Reads the prediction log and the training data...
            Mount(target="/app/data", source=f"{HOST_PROJECT_DIR}/data", type="bind"),
            # ...and writes the Evidently HTML report.
            Mount(
                target="/app/reports",
                source=f"{HOST_PROJECT_DIR}/reports",
                type="bind",
            ),
        ],
        environment=DRIFT_ENV,
        docker_url="unix://var/run/docker.sock",
        network_mode=COMPOSE_NETWORK,
        auto_remove="success",
        mount_tmp_dir=False,
    )
