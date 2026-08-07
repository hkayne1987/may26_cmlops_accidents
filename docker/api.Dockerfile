# FastAPI Inference Service Dockerfile
# Phase 1: Containerized inference API

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

# Copy models after install (they change more often).
# Only used as a fallback: the API pulls the production model from the
# MLflow registry at startup when MLFLOW_TRACKING_URI is set.
COPY models/ ./models/

# Set PYTHONPATH so imports work (src.api becomes importable)
ENV PYTHONPATH=/app

# MLflow connection. MLFLOW_TRACKING_USERNAME/PASSWORD are never baked into
# the image; inject them at runtime (docker run --env-file / docker-compose).
ENV MLFLOW_TRACKING_URI=https://dagshub.com/hkayne1987/may26_cmlops_accidents.mlflow

# Expose API port
EXPOSE 8000

# Run the API
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]