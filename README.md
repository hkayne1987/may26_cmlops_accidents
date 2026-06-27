# MLOps Project Accidents

This repository provides a Machine Learning projects for Accidents project.

## 🏗 Project Architecture

```mermaid
graph TD
    subgraph Client_Layer
        User([User/Client])
    end

    subgraph Deployment_Layer
        Gateway[API Gateway / JWT]
        API[FastAPI Service]
    end

    subgraph Orchestration_Layer
        Prefect[Prefect/Airflow]
    end

    subgraph Data_Model_Management
        DVC[DVC / MinIO]
        MLflow[MLflow Tracking]
    end

    subgraph Monitoring_Layer
        Prom[Prometheus]
        Graf[Grafana]
        Evid[Evidently AI]
    end

    User --> Gateway
    Gateway --> API
    Prefect --> API
    Prefect --> DVC
    Prefect --> MLflow
    API --> MLflow
    API --> Prom
    Prom --> Graf
    Evid --> Graf
```

## 🚀 Quick Start

### 1. Prerequisites
- [UV](https://astral.sh/uv) (Fast Python package manager)
- [Docker & Docker Compose](https://docs.docker.com/)

### 2. Installation
```bash
# Clone the repo
git clone <your-repo-url>
cd mlops-project

# Sync dependencies (dev deps are included automatically)
uv sync
```

### 3. Running the project
This project is built progressively. Depending on your current phase:

- **Phase 1 (API only):**
  ```bash
  docker-compose up api -d
  ```
- **Phase 2-4 (Full Stack):**
  ```bash
  docker-compose up -d
  ```

## 📖 Methodology & Roadmap

The project is designed to be built in 4 iterative phases. For a detailed step-by-step guide, please refer to the documentation:

- 🇫🇷 [French Version](./docs/00_project_methodology_fr.md)
- 🇬🇧 [English Version](./docs/00_project_methodology_en.md)

## 🔧 Hydra Configuration

This project uses [Hydra](https://hydra.cc/) for managing configurations. Configs are located in the `configs/` directory.

### Configuration Files

| Config File | Purpose |
|-------------|----------|
| `train.yaml` | Training pipeline configuration |
| `evaluate.yaml` | Model evaluation configuration |
| `preprocess.yaml` | Data preprocessing configuration |
| `drift.yaml` | Drift detection configuration |
| `problem.yaml` | Problem framing and business metrics |
| `model/*.yaml` | Model-specific hyperparameters |
| `preprocessing/*.yaml` | Preprocessing settings |

### Basic Usage

Each module can be run with Hydra using `python -m`:

```bash
# Preprocess data
python -m training.preprocess

# Train model
python -m training.train

# Evaluate model
python -m training.evaluate

# Detect drift
python -m monitoring.detection
```

### Overriding Config Values

Override any config value from the command line:

```bash
# Override model hyperparameters
python -m training.train model.n_estimators=200 model.max_depth=15

# Override multiple values
python -m training.train model=lightgbm training.cv_folds=10

# Override preprocessing
python -m training.preprocess preprocessing.test_size=0.3 preprocessing.scale_features=false

# Override drift threshold
python -m monitoring.detection drift.threshold=0.3
```

### Using Different Config Groups

```bash
# Use a different model configuration
python -m training.train model=lightgbm

# Use a different preprocessing config
python -m training.preprocess preprocessing=advanced
```

### Multi-run with Hydra

Run experiments with different configurations:

```bash
# Run training with different models
python -m training.train --multirun model=lightgbm,xgboost,randomforest
```

## 📂 Project Structure Recap
- `src/api/`: FastAPI inference code.
- `src/training/`: Training, preprocessing and evaluation scripts.
- `src/monitoring/`: Drift detection and observability logic.
- `src/common/`: Shared utilities used by all packages.
- `deployment/`: Infrastructure configs (Prometheus, Grafana).
- `docker/`: Dockerfiles for each service.
- `configs/`: Problem framing and Hydra configurations.
- `data/`: DVC-managed data folders.
- `reports/`: Generated drift and evaluation reports (HTML).
