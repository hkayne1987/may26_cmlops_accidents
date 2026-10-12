"""End-to-end training pipeline: data -> train -> evaluate -> release gate.

Runs every day, and retrains only when the data on main has changed: in
practice, when a pull request opened by .github/workflows/ingest.yml has been
merged. A manual trigger always runs the whole pipeline.

    check_new_data -> sync_main -> dvc_pull -> preprocess -> train
                   -> evaluate -> promote -> reload_api

The pipeline works on its own clone of main (airflow_data/workspace), never on
the developer's checkout: it trains on the reviewed data, whatever branch or
local changes sit on the host. The code comes from the images release.yml
publishes from main, so data and code always match.

Each task runs one of the images already built for this project, so Airflow
orchestrates the exact same containers a developer runs by hand. Nothing is
reimplemented here, and the Airflow image needs none of the project
dependencies.

The pipeline never promotes a model on its own merits alone: `promote` is a
release gate that moves the production alias only when the metrics logged by
`evaluate` clear the thresholds in src/training/promote.py. A degraded model
leaves production untouched, which matters because emergency call centre
operators act on these predictions.
"""

import base64
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import ShortCircuitOperator
from docker.types import Mount

log = logging.getLogger(__name__)

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

# Where the reviewed data lives: main on GitHub. The repository is public, so
# reading it needs no credentials.
REPO_URL = os.environ.get(
    "PIPELINE_REPO_URL", "https://github.com/hkayne1987/may26_cmlops_accidents.git"
)
REPO_BRANCH = os.environ.get("PIPELINE_REPO_BRANCH", "main")
RAW_DVC_URL = (
    REPO_URL.removesuffix(".git").replace("github.com", "raw.githubusercontent.com")
    + f"/{REPO_BRANCH}/data/raw.dvc"
)
REGISTERED_MODEL_NAME = "xgb_severity"

# Credentials reach the tasks through the Airflow container's own environment,
# which Docker Compose fills from .env. Nothing is hardcoded here.
DAGSHUB_ENV = {
    "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
    "MLFLOW_TRACKING_USERNAME": os.environ.get("MLFLOW_TRACKING_USERNAME", ""),
    "MLFLOW_TRACKING_PASSWORD": os.environ.get("MLFLOW_TRACKING_PASSWORD", ""),
    "DAGSHUB_USER": os.environ.get("DAGSHUB_USER", ""),
    "DAGSHUB_TOKEN": os.environ.get("DAGSHUB_TOKEN", ""),
}

# The pipeline's clone of main, kept between runs so each sync only fetches
# what changed. airflow_data/ is gitignored.
WORKSPACE_DIR = f"{HOST_PROJECT_DIR}/airflow_data/workspace"
WORKSPACE_MOUNT = Mount(target="/app", source=WORKSPACE_DIR, type="bind")
DATA_MOUNT = Mount(target="/app/data", source=f"{WORKSPACE_DIR}/data", type="bind")
MODELS_MOUNT = Mount(
    target="/app/models", source=f"{WORKSPACE_DIR}/models", type="bind"
)

default_args = {
    "owner": "mlops-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def docker_task(
    task_id: str, image: str, command: str | list[str], mounts: list, **kwargs
):
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


def _get(url: str, auth: bool = False) -> bytes:
    request = urllib.request.Request(url)
    if auth:
        user = DAGSHUB_ENV["MLFLOW_TRACKING_USERNAME"]
        password = DAGSHUB_ENV["MLFLOW_TRACKING_PASSWORD"]
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main_data_version() -> str:
    """The data version on main, as train.py records it: raw:<md5 of data/raw>."""
    match = re.search(r"md5:\s*([0-9a-f]{32})", _get(RAW_DVC_URL).decode())
    if not match:
        raise ValueError(f"No md5 in {RAW_DVC_URL}")
    return f"raw:{match.group(1)}"


def latest_trained_data_version() -> str | None:
    """data_version of the most recent registered model, None if there is none.

    The most recent version, not the one in production: when the release gate
    rejects a model, production keeps its older data version, and comparing
    against it would retrain on the same data every day.
    """
    api = f"{DAGSHUB_ENV['MLFLOW_TRACKING_URI']}/api/2.0/mlflow"
    query = urllib.parse.urlencode(
        {"filter": f"name='{REGISTERED_MODEL_NAME}'", "max_results": 1000}
    )
    versions = json.loads(_get(f"{api}/model-versions/search?{query}", auth=True))
    versions = versions.get("model_versions", [])
    if not versions:
        return None
    latest = max(versions, key=lambda v: int(v["version"]))
    run = json.loads(_get(f"{api}/runs/get?run_id={latest['run_id']}", auth=True))
    params = {p["key"]: p["value"] for p in run["run"]["data"].get("params", [])}
    log.info(f"Latest registered model: v{latest['version']}")
    return params.get("data_version")


def has_new_data(dag_run) -> bool:
    """True when main holds data no model was trained on yet.

    A manual trigger always runs. Any error (GitHub or DagsHub unreachable)
    fails the task rather than retraining blindly.
    """
    if dag_run.run_type == "manual":
        log.info("Manual run: training whatever the data version")
        return True
    on_main = main_data_version()
    trained = latest_trained_data_version()
    log.info(f"Data on main: {on_main}, latest model trained on: {trained}")
    return on_main != trained


with DAG(
    dag_id="ml_pipeline",
    description="Retrains when the data on main changes, then gates the release",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    # Daily, like the drift check: the data changes a few times a year, so a
    # new model the day after the merge is soon enough. Trigger the DAG by hand
    # to retrain at once. Without new data the run stops at the first task.
    schedule="@daily",
    catchup=False,
    # Never two trainings at once, e.g. a manual run during a scheduled one.
    max_active_runs=1,
    # Active as soon as Airflow starts: retraining must not wait for someone
    # to unpause the DAG in the UI.
    is_paused_upon_creation=False,
    tags=["mlops", "training"],
) as dag:
    # Skips every task below when there is nothing new to train on.
    check_new_data = ShortCircuitOperator(
        task_id="check_new_data",
        python_callable=has_new_data,
    )

    # Clones main into the workspace, or brings it up to date. The DVC image
    # has git; its entrypoint runs dvc, so it is replaced by a plain shell.
    # airflow_data is mounted rather than the workspace itself, because a bind
    # mount needs its source to exist and the first run has to create it.
    sync_main = docker_task(
        task_id="sync_main",
        image=DVC_IMAGE,
        entrypoint=["sh", "-c"],
        # A list, so the script reaches sh -c as one argument: a plain string
        # would be split on spaces and sh would only run its first word.
        command=[
            "set -e; W=/airflow_data/workspace; "
            "git config --global --add safe.directory '*'; "
            'if [ -d "$W/.git" ]; then '
            f'git -C "$W" fetch --depth 1 origin {REPO_BRANCH} && '
            'git -C "$W" reset --hard FETCH_HEAD; '
            f'else git clone --depth 1 --branch {REPO_BRANCH} {REPO_URL} "$W"; fi; '
            'mkdir -p "$W/models"; '
            'git -C "$W" log -1 --format="Workspace at %h: %s"'
        ],
        mounts=[
            Mount(
                target="/airflow_data",
                source=f"{HOST_PROJECT_DIR}/airflow_data",
                type="bind",
            )
        ],
    )

    # Pulls the raw data main points to. Only data/raw: preprocess rebuilds
    # data/processed, and the copy recorded in dvc.lock is stale.
    dvc_pull = docker_task(
        task_id="dvc_pull",
        image=DVC_IMAGE,
        command="pull data/raw.dvc",
        mounts=[WORKSPACE_MOUNT],
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

    # Tells the API to load the newly promoted model right away. It reaches one
    # replica behind the service name; the others notice the new alias on
    # their own within MODEL_POLL_SECONDS.
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

    (
        check_new_data
        >> sync_main
        >> dvc_pull
        >> preprocess
        >> train
        >> evaluate
        >> promote
        >> reload_api
    )
