"""End-to-end training pipeline: data -> train -> evaluate -> release gate.

Each task runs one of the images already built for this project, so Airflow
orchestrates the exact same containers a developer runs by hand. Nothing is
reimplemented here, and the Airflow image needs none of the project
dependencies.

The pipeline never promotes a model on its own merits alone: `promote` is a
release gate that moves the production alias only when the metrics logged by
`evaluate` clear the thresholds in src/training/promote.py. A degraded model
leaves production untouched, which matters because emergency call centre
operators act on these predictions.

Triggered manually (schedule=None): a model change must be a decision, not a
side effect of a nightly run.
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.bash import BashOperator
from docker.types import Mount

# Absolute path of the repo on the Docker host. The containers started by
# Airflow are siblings, not children: their bind mounts are resolved by the
# host daemon, so a path inside the Airflow container would not exist.
# Docker Compose sets this from the host shell.
HOST_PROJECT_DIR = os.environ["HOST_PROJECT_DIR"]

# The images release.yml publishes on every merge to main, same names as in
# docker-compose.yml.
TRAINING_IMAGE = os.environ.get(
    "TRAINING_IMAGE", "ghcr.io/hkayne1987/may26_cmlops_accidents/baac-training:latest"
)
DVC_IMAGE = os.environ.get(
    "DVC_IMAGE", "ghcr.io/hkayne1987/may26_cmlops_accidents/baac-dvc:latest"
)
# Pull before each task so the pipeline always runs the latest published code.
# Set PULL_IMAGES=false to run images built locally and not yet published.
PULL_IMAGES = os.environ.get("PULL_IMAGES", "true").lower() == "true"

# Credentials reach the tasks through the Airflow container's own environment,
# which Docker Compose fills from .env. Nothing is hardcoded here.
DAGSHUB_ENV = {
    "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
    "MLFLOW_TRACKING_USERNAME": os.environ.get("MLFLOW_TRACKING_USERNAME", ""),
    "MLFLOW_TRACKING_PASSWORD": os.environ.get("MLFLOW_TRACKING_PASSWORD", ""),
    "DAGSHUB_USER": os.environ.get("DAGSHUB_USER", ""),
    "DAGSHUB_TOKEN": os.environ.get("DAGSHUB_TOKEN", ""),
}

DATA_MOUNT = Mount(
    target="/app/data", source=f"{HOST_PROJECT_DIR}/data", type="bind"
)
MODELS_MOUNT = Mount(
    target="/app/models", source=f"{HOST_PROJECT_DIR}/models", type="bind"
)
REPO_MOUNT = Mount(target="/app", source=HOST_PROJECT_DIR, type="bind")

default_args = {
    "owner": "mlops-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def docker_task(task_id: str, image: str, command: str, mounts: list, **kwargs):
    """Builds a DockerOperator with the settings shared by every task."""
    return DockerOperator(
        task_id=task_id,
        image=image,
        command=command,
        mounts=mounts,
        environment=DAGSHUB_ENV,
        # Talk to the host daemon through the socket mounted into Airflow.
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove="success",
        force_pull=PULL_IMAGES,
        # Airflow's default tmp mount collides with the images' own /tmp usage.
        mount_tmp_dir=False,
        **kwargs,
    )


with DAG(
    dag_id="ml_pipeline",
    description="Data pull, preprocessing, training, evaluation and gated release",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,  # manual trigger only
    catchup=False,
    tags=["mlops", "training"],
) as dag:
    # Pull the DVC-tracked raw data from DagsHub so the pipeline starts from a
    # known version rather than whatever happens to sit on the host. Only
    # data/raw: preprocess rebuilds data/processed, and pulling the stale copy
    # in dvc.lock fails when the local files differ from it.
    dvc_pull = docker_task(
        task_id="dvc_pull",
        image=DVC_IMAGE,
        command="pull data/raw.dvc",
        mounts=[REPO_MOUNT],
    )

    preprocess = docker_task(
        task_id="preprocess",
        image=TRAINING_IMAGE,
        command="uv run python -m src.data.preprocess",
        mounts=[DATA_MOUNT, MODELS_MOUNT],
    )

    train = docker_task(
        task_id="train",
        image=TRAINING_IMAGE,
        command="uv run python -m src.training.train",
        mounts=[DATA_MOUNT, MODELS_MOUNT],
    )

    # Logs auc_roc, recall and precision onto the training run. The gate below
    # reads them back, so this task is what makes promotion possible at all.
    evaluate = docker_task(
        task_id="evaluate",
        image=TRAINING_IMAGE,
        command="uv run python -m src.training.evaluate",
        mounts=[DATA_MOUNT, MODELS_MOUNT],
    )

    # Release gate: fails (and leaves production untouched) when the metrics
    # fall short. A red task here means the previous model is still served.
    promote = docker_task(
        task_id="promote",
        image=TRAINING_IMAGE,
        command="uv run python -m src.training.promote",
        mounts=[DATA_MOUNT, MODELS_MOUNT],
    )

    # Tell the API to pick up the newly promoted model. Without this it keeps
    # serving the previous one until someone restarts the container.
    reload_api = BashOperator(
        task_id="reload_api",
        bash_command="""
set -euo pipefail
TOKEN=$(curl -sf -X POST "$API_URL/token" \
  -d "username=$API_USERNAME&password=$API_PASSWORD" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
curl -sf -X POST "$API_URL/admin/reload-model" \
  -H "Authorization: Bearer $TOKEN"
""",
        env={
            # The API is reachable on the compose network under its service name.
            "API_URL": os.environ.get("API_URL", "http://api:8000"),
            "API_USERNAME": os.environ.get("API_USERNAME", "airflow_service"),
            "API_PASSWORD": os.environ.get("API_PASSWORD", ""),
        },
    )

    dvc_pull >> preprocess >> train >> evaluate >> promote >> reload_api
