# Training Container Dockerfile
# Phase 2: Containerized training pipeline

FROM python:3.10-slim

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install workspace packages (production only, no dev dependencies).
# --frozen: fail the build instead of silently re-resolving if the
# lockfile and pyproject.toml are out of sync.
RUN uv sync --no-dev --frozen

# Copy source after dependency install (changes more often)
COPY src/ ./src/

# Copy preprocessed data (can be mounted instead in production)
COPY data/processed/ ./data/processed/

# Set PYTHONPATH
ENV PYTHONPATH=/app

# MLflow connection. MLFLOW_TRACKING_USERNAME/PASSWORD are never baked into
# the image; inject them at runtime (docker run -e / docker-compose / CI secrets).
# MLFLOW_TRACKING_URI defaults to the team's DagsHub repo below; override with
# file:./mlruns (or unset it, see train.py's fallback) to track locally instead.
ENV MLFLOW_TRACKING_URI=https://dagshub.com/hkayne1987/may26_cmlops_accidents.mlflow

# Default command (can be overridden)
CMD ["uv", "run", "python", "-m", "src.training.train"]