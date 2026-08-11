
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

- Python 3.10+
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

The BAAC data is **not** versioned in this repo and must be downloaded manually
from [data.gouv.fr](https://www.data.gouv.fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2024/).

For each year **2019 to 2024**, download the 4 CSV files and place them in
`data/raw/` using the naming scheme `type-year.csv`:

data/raw/  
├── caracteristiques-2019.csv  
├── lieux-2019.csv  
├── vehicules-2019.csv  
├── usagers-2019.csv  
└── ... (through 2024 — 24 files total)

The variable description PDF is in `docs/`.

## Pipeline

The pipeline runs in three steps:

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
hyperparameters, metrics and the data version there, and registers the model as
`xgb_severity` with a `production` alias pointing at the latest version.

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

```bash
docker-compose up -d --build api     # inference API on http://localhost:8000
```

The API pulls `models:/xgb_severity@production` from the registry at startup
(~10-15s), falling back to `models/xgb_severity.joblib` if the registry is
unreachable. Check which model it loaded with:

```bash
curl http://localhost:8000/health
# {"status":"healthy","model_loaded":true,"model_version":"2"}
```

Training runs on demand rather than as part of the default stack:

```bash
docker-compose --profile training run --rm training \
  uv run python -m src.training.train --sample
```

This publishes a new model version, moves the `production` alias to it, and the
API serves it after a `docker-compose restart api`.

Prometheus and Grafana sit behind a `monitoring` profile
(`docker-compose --profile monitoring up -d`). They are not wired up yet: the
API's `/metrics` endpoint is still commented out.

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
```bash
GET /health
Returns the health status of the API, whether a model is loaded, and the version
of the model currently served ("local" when loaded from the fallback file).

POST /predict
Performs inference using the trained XGBoost model.
```
Request Body:
The request must contain a list of feature values in the same order used during model training.

Example Request
```bash
{
  "features": [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}
```

Example Response
```bash
{
  "prediction": 0,
  "probabilities": [0.9995688199996948, 0.00043118116445839405],
  "model_version": "2"
}
```
`model_version` is the MLflow registry version currently served, or `"local"`
when the API fell back to the on-disk model.

Important: `features` must list the 40 values in the exact order the model
expects. The authoritative order is the one carried by the served model itself
(`model.get_booster().feature_names`); `models/feature_columns.json` records the
same order and is the easiest way to read it:

```bash
uv run python -c "import json; print(json.load(open('models/feature_columns.json')))"
```

Both come from the same training run, so they match — but if you ever retrain
without committing the updated schema, trust the served model over the file.

## Project structure

`src/data/`: Load, merge BAAC tables, build target, feature engineering and train/test split  
`src/training/`: Train, save model + feature schema, metrics calculation and threshold search  
`src/api/`: FastAPI inference service  
`data/`: Raw BAAC CSVs (downloaded manually), train.parquet / test.parquet  
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