
# Road Accident Severity Prediction (BAAC)

MLOps project predicting the severity of road accidents in France from the
public BAAC dataset. The goal is to expose a trained classifier through an API;
the focus of the project is on the MLOps lifecycle (reproducible pipeline,
deployment, monitoring) rather than on modeling performance.

**Target:** binary classification — `grave` (killed or hospitalized) vs
`non grave` (unharmed or slightly injured).
**Scope:** BAAC 2019–2024 (homogeneous schema after the 2019 TRAxy reform).
**Grain:** one row per road user involved in an accident.

## Requirements

- Python 3.10, pinned in `.python-version` to match the CI and the Docker
  images. uv picks it up on its own (and downloads it if needed): after a
  `git pull`, `uv sync` recreates `.venv` if it was built on another version.
- [uv](https://docs.astral.sh/uv/) for dependency management
- Docker (optional, to run the containerized services)

## Installation

```bash
git clone https://github.com/hkayne1987/may26_cmlops_accidents.git
cd may26_cmlops_accidents

## Install UV if you don't have it:
brew install uv           #bash
or 
curl -LsSf https://astral.sh/uv/install.sh | sh

## Create the virtual environment
uv venv
source .venv/bin/activate
uv sync 
```

> macOS note: XGBoost requires the OpenMP runtime. If you hit a
> `libxgboost.dylib could not be loaded` error, run `brew install libomp`.

## Data

The BAAC data is versioned with DVC on DagsHub, not stored in Git. Git only
carries the hashes (`data/raw.dvc`, `dvc.lock`); the files themselves are
pulled on demand:

```bash
docker-compose --profile dvc run --rm dvc pull
```

Run this once after cloning, and again after any `git pull` that changes
`dvc.lock` or a `.dvc` file. It is never automatic — `dvc status` tells you
whether you are out of date:

```bash
docker-compose --profile dvc run --rm dvc status
```

This populates `data/raw/` with the 4 tables per year for **2019 to 2024**
(`caracteristiques`, `lieux`, `vehicules`, `usagers` — 24 files) and
`data/processed/` with `train.parquet` / `test.parquet`.

After regenerating the data, push it back to the remote:

```bash
docker-compose --profile dvc run --rm dvc push
```

The original source is
[data.gouv.fr](https://www.data.gouv.fr/datasets/53698f4ca3a729239d2036df/);
the variable description PDF is in `docs/`. New years and corrections are
picked up from there automatically every month, and `data/sources.json`
records which published file each local one comes from, see
[Automatic updates](#automatic-updates).

## Pipeline

The pipeline runs in three steps (pull the data first, see [Data](#data)):

```bash
make preprocess   # load, merge and clean the 4 tables -> data/processed/{train,test}.parquet
make train        # train XGBoost on train.parquet -> models/ + MLflow run
make evaluate     # evaluate on test.parquet, log metrics to the MLflow run
```

`make train` registers the model in the MLflow registry and writes the run id to
`models/run_id.txt`; `make evaluate` reuses that run id so metrics land on the
same run. See [MLflow tracking](#mlflow-tracking-dagshub) below.

Equivalent direct commands:

```bash
python -m src.data.preprocess
python -m src.training.train
python -m src.training.evaluate
```

Optional flags for training:

```bash
make train-sample   # train on a small subsample (quick test)
make tune           # run hyperparameter search (long)
make tune-sample    # run hyperparameter search on a subsample (quick test)
```

## MLflow tracking (DagsHub)

Experiment tracking and the model registry are hosted on
[DagsHub](https://dagshub.com/hkayne1987/may26_cmlops_accidents). Training logs
hyperparameters, metrics and the data version there, and registers each model
as a new version of `xgb_severity`. The `production` alias only moves when the
model clears the [release gate](#the-release-gate).

The data version is the md5 DVC records for `data/raw` in `data/raw.dvc`
(`raw:8cbc0202...`), so any model can be traced back to the exact files it was
trained on.

Create a `.env` file at the repo root (it is gitignored — never commit it):

```
MLFLOW_TRACKING_URI=https://dagshub.com/hkayne1987/may26_cmlops_accidents.mlflow
MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

Each team member needs their own token (DagsHub → Settings → Tokens); tokens are
personal and must not be shared.

Without `MLFLOW_TRACKING_URI`, training falls back to a local `file:./mlruns`
backend, which is enough to run the pipeline offline.

## Running with Docker

Services are defined in `docker-compose.yml` and read credentials from `.env`.
The API needs `JWT_SECRET_KEY` there or it refuses to start, see
[Authentication](#authentication).

```bash
docker-compose up -d api             # inference API on http://localhost:8000
```

The images are the ones CI publishes on GHCR (see [CI/CD](#cicd)): `up` pulls
them if they are missing, `docker-compose pull` fetches the latest release.
To run local changes instead, build them under the same name with
`docker-compose build api` (or `up -d --build api`).

The API pulls `models:/xgb_severity@production` from the registry at startup
(~10-15s), falling back to `models/xgb_severity.joblib` if the registry is
unreachable. Check which model it loaded with:

```bash
curl http://localhost:8000/health
# {"status":"healthy","model_loaded":true,"model_version":"2"}
```

Training runs on demand rather than as part of the default stack. The image
carries no data: `data/` is bind-mounted, so pull it first (see [Data](#data)).

```bash
docker-compose --profile training run --rm training \
  uv run python -m src.training.train --sample
```

This registers a new model version. It only reaches production through the
release gate, and the API then picks it up on its own within 5 minutes, see
[Automatic updates](#automatic-updates).

Prometheus and Grafana sit behind a `monitoring` profile, see
[Monitoring](#monitoring).

## API
Run the API
Test the API Internally
Start the API locally with:
```bash
uv run uvicorn src.api.main:app --reload

## Once started, the API will be available at:
API: http://localhost:8000
Interactive documentation (Swagger UI): http://localhost:8000/docs
ReDoc documentation: http://localhost:8000/redoc
```

Endpoints:

| Endpoint | Access | Purpose |
|---|---|---|
| `GET /health` | public | Liveness, whether a model is loaded, and the served version |
| `POST /token` | public | Exchanges username and password for an access token |
| `GET /me` | any logged-in user | Returns the account behind the current token |
| `POST /predict` | operator, service, admin | Runs inference |
| `GET /model/schema` | operator, service, admin | Lists the columns `/predict` expects and the values each accepts |
| `POST /admin/reload-model` | service, admin | Reloads the production model without a restart |
| `POST /admin/users` | admin | Creates an account |
| `GET /admin/users` | admin | Lists accounts |
| `DELETE /admin/users/{username}` | admin | Disables an account |
| `GET /metrics` | internal (Prometheus) | Metrics in Prometheus format, see [Monitoring](#monitoring) |

`/health` stays public so Docker and the reverse proxy can probe it; every
other endpoint except `/token` requires a token (see
[Authentication](#authentication)).

### Request format

`/predict` takes the **real BAAC values**, keyed by column name, as they
appear in the data.gouv.fr files: `"catv": "7"` for a car, `"lum": "1"` for
daylight. Categorical codes may be sent as strings or numbers, and `null`
means unknown. Column order does not matter.

The API converts these values to the model's internal encoding itself:
XGBoost stores the categories it saw in training inside the model, so the
mapping always matches the version being served and follows each retrain.

Example, a real severe accident from the test set (a driver in a car on an
80 km/h road in Haute-Corse, in daylight):

```json
{
  "features": {
    "place": "1", "catu": "1", "sexe": "2", "secu1": "1", "secu2": "-1",
    "secu3": "-1", "locp": "0", "actp": "0", "etatp": "-1", "senc": "2",
    "catv": "7", "obs": "2", "obsm": "0", "choc": "3", "manv": "13",
    "occutc": null, "jour": 19, "mois": 6, "an": 2023, "lum": "1",
    "dep": "2B", "com": "2B193", "agg": "1", "int": "1", "atm": "1",
    "col": "6", "lat": 42.37663, "long": 9.18949, "catr": "2", "circ": "2",
    "nbv": 2, "vosp": "0", "prof": "1", "plan": "3", "surf": "1",
    "infra": "0", "situ": "3", "vma": 80, "heure": 16, "minute": 57
  }
}
```

Response:

```json
{
  "prediction": 1,
  "probabilities": [0.22261929512023926, 0.7773807048797607],
  "model_version": "6"
}
```

A request that does not match the model is refused with a 422 listing every
problem at once: missing or unknown columns, non-numeric values, and category
values never seen in training (with the accepted ones). `GET /model/schema`
lists the 40 expected columns and their allowed values; the column meanings
are in the variable description PDF in `docs/`.
`model_version` is the MLflow registry version currently served, or `"local"`
when the API fell back to the on-disk model.

## Authentication

The API is meant for emergency call centre staff, so `/predict` is not open:
callers log in and use a short-lived token. Three roles exist:

| Role | Can |
|---|---|
| `operator` | Run predictions. Call centre staff. |
| `service` | Run predictions and reload the production model, but not touch accounts. Used by the Airflow pipeline. |
| `admin` | Everything, including managing accounts. |

Set `JWT_SECRET_KEY` in `.env` before starting. It signs the tokens, so a weak
value would let anyone forge them: the API refuses to start if it is missing
or shorter than 32 characters.

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Accounts live in a SQLite database (`data/api_users.db`, gitignored: it holds
password hashes). Create the first admin from the command line, since
`/admin/users` is itself admin-only:

```bash
docker-compose run --rm api uv run python -m src.api.manage_users create <name> --role admin
```

The same script offers `list` and `disable`. The password is prompted rather
than passed as an argument, so it stays out of the shell history.

Log in, then send the token on every call:

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/token \
  -d "username=<name>&password=<password>" | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"features": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}'
```

Tokens expire after `ACCESS_TOKEN_EXPIRE_MINUTES` (30 by default). Disabling an
account takes effect immediately rather than at expiry: the API re-reads the
account on every request, so a revoked operator cannot keep working with a
token issued moments earlier.

Swagger UI at `/docs` has an **Authorize** button that performs the same login.
Once authorized, `GET /model/schema` shows what `/predict` expects, see
[Request format](#request-format).

## Orchestration (Airflow)

The `ml_pipeline` DAG runs the whole chain, each step in the container it
already ships with, so Airflow adds orchestration without duplicating a single
dependency:

```
dvc_pull -> preprocess -> train -> evaluate -> promote -> reload_api
```

```bash
docker-compose --profile airflow up -d     # http://localhost:8080
```

Trigger it from the UI or the command line:

```bash
docker-compose --profile airflow exec airflow airflow dags trigger ml_pipeline
```

`schedule=None`: the pipeline runs only when someone asks. A model change in a
service used by emergency call centres should be a decision, not the side
effect of a nightly job.

### The release gate

`train` registers a new model version but does not put it in production.
`promote` does, and only when the metrics logged by `evaluate` clear the
thresholds in `src/training/promote.py`:

| Metric | Minimum | Override |
|---|---|---|
| `auc_roc` | 0.85 | `MIN_AUC_ROC` |
| `recall` (severe class) | 0.85 | `MIN_RECALL_SEVERE` |

Recall comes first: reporting a severe accident as minor is worse than
over-flagging a minor one. Both values sit a few points below what a full
training run produces (auc 0.8763, recall 0.8961), so a comparable model passes
and a degraded one does not.

When a metric falls short, `promote` fails, `reload_api` never runs, and the
previous model keeps serving. A red `promote` task means production is
untouched, which is the intended outcome rather than an incident.

Missing metrics count as a failure too: an unmeasured model must not reach
production.

### Requirements

The DAG needs a `service` account for the API (`reload_api` calls
`/admin/reload-model`), and `API_PASSWORD` set in `.env`:

```bash
make users ARGS="create airflow_service --role service"
```

Airflow starts the task containers through the host Docker socket, mounted into
its own container. That is a privileged access: anything running in Airflow can
control Docker on the host. Acceptable locally, to revisit before any real
deployment.

## CI/CD

Four workflows, all in `.github/workflows/`. The first two check and publish
the code; the last two keep the data and the model up to date, see
[Automatic updates](#automatic-updates).

**`ci.yml`** runs on every pull request and every push to `main`:

| Step | Command |
|---|---|
| Lint | `ruff check src/ tests/` |
| Format | `ruff format --check src/ tests/` |
| Types | `mypy src/` |
| Tests | `pytest tests/ --cov=src` |
| Images | builds the three Dockerfiles without pushing (pull requests only) |

**`release.yml`** runs once a change is merged to `main`, and publishes the
three images to the GitHub Container Registry:

```
ghcr.io/hkayne1987/may26_cmlops_accidents/baac-api:latest
ghcr.io/hkayne1987/may26_cmlops_accidents/baac-training:latest
ghcr.io/hkayne1987/may26_cmlops_accidents/baac-dvc:latest
```

Each image is tagged twice: with the commit SHA, which pins a build for good,
and with `latest`, which follows `main`. Pull one with:

```bash
docker pull ghcr.io/hkayne1987/may26_cmlops_accidents/baac-api:latest
```

GHCR rather than Docker Hub: the registry token is provided by GitHub itself
(`secrets.GITHUB_TOKEN`), so there is no secret to create or rotate, and the
images belong to the repository rather than to one member's personal account.
Docker Hub organisations are a paid feature.

The packages are public: anyone can pull them without logging in.

**`ingest.yml`** runs on the 1st of each month and checks data.gouv.fr for new
or corrected BAAC files. **`retrain.yml`** runs when `data/raw.dvc` changes on
`main`. Both are described below.

### Secrets they need

Set in the repository **Settings → Secrets and variables → Actions**:

| Name | Value | Used by |
|---|---|---|
| `DAGSHUB_USER` | a DagsHub username with write access to the repository | `ingest.yml`, `retrain.yml` |
| `DAGSHUB_TOKEN` | that user's DagsHub access token | `ingest.yml`, `retrain.yml` |
| `DATA_PR_TOKEN` | a member's GitHub classic token, `public_repo` scope only | `ingest.yml` |

`DATA_PR_TOKEN` is there because GitHub's own workflow token may not open pull
requests unless the repository owner allows it, which collaborators cannot do.
The data pull requests are therefore opened in that member's name, and
`ingest.yml` fails once the token expires: generate a new one then. Without the
secret, `ingest.yml` falls back to GitHub's token.

## Automatic updates

From a new file on data.gouv.fr to the model and code the API serves, with a
human decision in the middle:

| Step | Where | What happens |
|---|---|---|
| 1. Detection | `ingest.yml`, monthly | compares data.gouv.fr with `data/sources.json` |
| 2. Ingestion | `ingest.yml` | downloads new or corrected files, versions them on DagsHub with DVC |
| 3. Review | pull request | a human checks the new data and merges, or not |
| 4. Retraining | `retrain.yml`, on merge | dvc pull, preprocess, train, evaluate, release gate |
| 5. Model update | the API | sees `@production` move in the registry and reloads, no restart |
| 6. Code update | Watchtower | replaces the running API when CI publishes a new image |

### 1-3. New data

`src/data/ingest.py` asks the data.gouv.fr API which BAAC files exist from
2019 on. File names change every year (`caracteristiques-2019`,
`carcteristiques-2021`, `caract-2023`, `Caract_2024`), so tables are recognised
by prefix and saved under one naming scheme in `data/raw/`.

A file counts as changed when its date or size differs from
`data/sources.json`, the manifest of the files in use. It is then downloaded
and hashed: only a different content is a change, since the API omits the
checksum for some files and a metadata edit alone must not trigger a retrain.
When nothing changed, the workflow stops after one API call, with no pull
request and no notification.

Monthly rather than yearly: a new year comes out around October, but past
years are corrected at any time (`usagers-2022` was republished in 2025).

```bash
uv run python -m src.data.ingest --dry-run   # what changed, nothing downloaded
```

`src/data/preprocess.py` finds the years from the files present, so a new year
needs no code change.

Two things to know:

- On a public repository, GitHub **disables scheduled workflows after 60 days
  without activity**. Re-enable `ingest.yml` from the Actions tab if it stops.
- A file whose metadata changed but content did not is downloaded again each
  month, until the next real data pull request records it. Harmless.

### 4. Retraining

`retrain.yml` runs the same steps as the Airflow `ml_pipeline` DAG, on the
GitHub runner. Each run logs the data version (`raw:<md5 of data/raw>`) to
MLflow. A model that falls short of the [release gate](#the-release-gate) fails
the run and never reaches production. It can also be started by hand from the
Actions tab.

### 5. The API follows the registry

The API checks every `MODEL_POLL_SECONDS` (default 300, `0` disables it) which
version carries `@production`, and reloads when it moved. It keeps answering
with the previous model while the new one downloads. `POST
/admin/reload-model` still forces an immediate reload.

### 6. Watchtower updates the running API

```bash
docker-compose --profile autoupdate up -d
```

[Watchtower](https://github.com/nicholas-fedor/watchtower) (the maintained
fork; the original is archived) checks GHCR every 5 minutes and recreates the
`api` container when a new `baac-api:latest` is out, keeping its settings. It
only touches containers labelled `com.centurylinklabs.watchtower.enable=true`.

It is opt-in because in development it would replace an image you just built
locally with the published one. For the same reason, the Airflow DAGs pull the
published images before each task: set `PULL_IMAGES=false` in `.env` to run
local builds instead.

## Monitoring

The API exposes Prometheus metrics on `/metrics`. Prometheus scrapes it over
the Docker network, and Grafana shows them on a provisioned dashboard.

```bash
docker-compose up -d api
docker-compose --profile monitoring up -d
```

| Service | URL | Login |
|---|---|---|
| Prometheus | http://localhost:9090 | none |
| Grafana | http://localhost:3000/d/baac-api | `admin` / `GRAFANA_PASSWORD` from `.env` |
| Pushgateway | http://localhost:9091 | none |

The data source and the **BAAC severity API** dashboard are provisioned from
`deployment/grafana/`, so there is nothing to set up by hand. UI edits are
kept until Grafana restarts: export the JSON and commit it to keep them.

| Metric | What it tells you |
|---|---|
| `api_http_requests_total`, `api_http_request_duration_seconds` | traffic, errors and latency per route |
| `api_model_version_info`, `api_model_loaded` | which model version is served, and whether one is loaded at all |
| `api_model_load_duration_seconds`, `api_model_reloads_total` | how long the last load took, and reloads triggered by Airflow |
| `api_predictions_total` | predictions by outcome (`severe` / `non_severe`) and model version |
| `api_prediction_severe_probability` | distribution of the severe-class probability, often the first visible sign of drift |
| `api_login_attempts_total` | login successes and failures, to spot password guessing |

Routes are labelled by template (`/admin/users/{username}`), never by raw path,
so the number of time series stays bounded.

`/metrics` is unauthenticated because only Prometheus reads it, from inside
the Docker network. **It must not be exposed publicly**: the reverse proxy
should not forward it.

### Data drift

Whether an accident really was severe is only known once a year, with the
next BAAC release. Until then, the only early sign that the model may be
degrading is a change in what operators send it. That is what drift
detection watches.

1. Every call to `/predict` is logged with the real values received, in
   `data/predictions.db` (gitignored, mounted from the host).
2. `src/monitoring/detection.py` compares the last 7 days of requests with a
   sample of the training data, using [Evidently](https://www.evidentlyai.com/).
3. It writes an HTML report to `reports/drift/drift_report_latest.html` and
   pushes the result to the Pushgateway, so it shows in the **Data drift** row
   of the Grafana dashboard.
4. Airflow runs it daily (`drift_monitoring` DAG). It is safe to schedule: it
   only measures and never touches the served model.

The alert fires when **25% of the compared columns** drift, the same
threshold in Grafana and in the Evidently report. `com` (24k communes, too
sparse to test) and `an` (the year always moves forward) are left out, and a
column empty in the window is skipped. Thresholds and window can be tuned
from `.env`, see `env.example`.

**A drift alert does not retrain the model.** Retraining needs new labelled
data, and retraining on the same data would give the same model. The decision
process is:

| Signal | When | Response |
|---|---|---|
| Drift in the requests | any day | alert in Grafana, a human looks at what changed |
| New BAAC year versioned with DVC | once a year | run `ml_pipeline`; `promote` only releases a model that clears the metric gate |
| Manual decision | any time | run `ml_pipeline` by hand |

#### Trying it out

The API gets no real traffic in this project, so `make replay` sends real
accidents from the test set through it, like operators would. A scenario other
than `normal` restricts them to one kind of accident, making the inputs drift
on purpose:

```bash
docker-compose up -d api
docker-compose --profile monitoring up -d

make replay SCENARIO=normal ROWS=500     # ordinary accidents
make drift                               # -> no drift
make replay SCENARIO=motorway ROWS=500   # now half the week is motorways
make drift                               # -> alert, ~40% of columns drifted
```

Scenarios: `normal`, `motorway`, `pedestrians`, `night`, `two_wheelers`.

## Project structure

`src/data/`: Load, merge BAAC tables, build target, feature engineering and train/test split  
`src/training/`: Train, save model + feature schema, metrics calculation and threshold search  
`src/api/`: FastAPI inference service  
`data/`: Raw BAAC CSVs and train.parquet / test.parquet (versioned with DVC, not in Git)  
`models/`: Trained model + feature schema (metrics are logged to MLflow)  
`docker/`: Dockerfiles for the API and training images  
`docs/`: PDF explaining features  

## Methodology notes

Key choices:

- **Binary target** to absorb the known label noise on the
  hospitalized/slightly-injured boundary since the 2018 reform.
- **2019+ only** to keep a consistent schema (the BAAC format changed in 2019).
- **Group-aware split** (`GroupShuffleSplit` on `Num_Acc`) so that all users of
  the same accident stay on the same side, preventing leakage.
- **Schema drift handled across years** (e.g. `Accident_Id` → `Num_Acc` in 2022,
  `num_veh` not unique in some years — joins use `id_vehicule`).