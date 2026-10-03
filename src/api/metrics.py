"""Prometheus metrics exposed by the inference API on /metrics.

Three groups:

- HTTP traffic: request count and latency per route and status code.
- Model: which version is served, how predictions split between severe and
  non_severe, and the distribution of the severe-class probability. A shift
  in that distribution is often the first visible sign of data drift.
- Security: login attempts by outcome, so a burst of failures shows up on the
  dashboard.

Labels use route templates (/admin/users/{username}), never raw paths or user
names, to keep the number of time series bounded.
"""

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "api_http_requests_total",
    "HTTP requests handled, by method, route template and status code.",
    ["method", "route", "status"],
)

HTTP_LATENCY = Histogram(
    "api_http_request_duration_seconds",
    "Time spent handling a request, by method and route template.",
    ["method", "route"],
    # Inference takes milliseconds; login is slower on purpose (bcrypt).
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTIONS = Counter(
    "api_predictions_total",
    "Predictions served, by predicted class and model version.",
    ["outcome", "model_version"],
)

SEVERE_PROBABILITY = Histogram(
    "api_prediction_severe_probability",
    "Predicted probability of the severe class.",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

MODEL_LOADED = Gauge(
    "api_model_loaded",
    "1 when a model is loaded and predictions can be served, else 0.",
)

MODEL_VERSION = Gauge(
    "api_model_version_info",
    "Always 1, labelled with the model version currently served.",
    ["model_version"],
)

MODEL_LOAD_SECONDS = Gauge(
    "api_model_load_duration_seconds",
    "Time the last model load took, registry pull included.",
)

MODEL_RELOADS = Counter(
    "api_model_reloads_total",
    "Model reloads by result: success (/admin/reload-model), "
    "auto (new @production found in the registry) or no_model.",
    ["result"],
)

PREDICTION_LOG_FAILURES = Counter(
    "api_prediction_log_failures_total",
    "Predictions served but not written to the drift log.",
)

LOGIN_ATTEMPTS = Counter(
    "api_login_attempts_total",
    "Login attempts on /token, by result.",
    ["result"],
)


def set_served_model(version: str, loaded: bool) -> None:
    """Records which model is served, dropping the previous version label.

    Clearing first keeps a single series at 1: after a reload, a dashboard
    showing the served version must not also list the old one.
    """
    MODEL_VERSION.clear()
    MODEL_VERSION.labels(model_version=version).set(1)
    MODEL_LOADED.set(1 if loaded else 0)
