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

- Python 3.9+
- A virtual environment (standard `venv` for now; dependency management with `uv` will be added later)

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
make train        # train XGBoost on train.parquet -> models/
make evaluate     # evaluate on test.parquet, write models/metrics.json
```

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

## Project structure

`src/data/`: Load, merge BAAC tables, build target, feature engineering and train/test split  
`src/train/`: Train, save model + feature schema, metrics calculation and threshold search
`data/`: Raw BAAC CSVs (downloaded manually), train.parquet / test.parquet
`models/`: Trained model + metrics
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