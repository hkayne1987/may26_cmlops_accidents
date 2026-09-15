"""Promotes a freshly trained model to production, if its metrics allow it.

Training registers a model version but does not promote it. This script is the
release gate: it reads the metrics logged by evaluate.py and moves the
'production' alias only when they clear the thresholds below.

The model answers emergency calls, so the gate is deliberately one-sided: when
anything is unclear (missing metrics, unreachable registry), it refuses to
promote and leaves the current production model in place.

Run: python -m src.training.promote
"""

import logging
import os
import sys
from pathlib import Path

import mlflow

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

MODEL_DIR = Path("models")
RUN_ID_PATH = MODEL_DIR / "run_id.txt"
VERSION_PATH = MODEL_DIR / "model_version.txt"

REGISTERED_MODEL_NAME = "xgb_severity"
PRODUCTION_ALIAS = "production"

# Release thresholds. Recall on the severe class comes first: missing a severe
# accident (telling a call centre it is minor) is far worse than over-flagging
# a minor one.
#
# Calibrated on a full training run measured on data/processed/test.parquet
# at threshold 0.35: auc_roc 0.8763, recall 0.8961, precision 0.3800.
#
# The values below sit a few points under that: a comparable model passes, a
# clearly degraded one does not. Both are overridable to tighten the gate
# without a code change.
MIN_AUC_ROC = float(os.environ.get("MIN_AUC_ROC", "0.85"))
MIN_RECALL_SEVERE = float(os.environ.get("MIN_RECALL_SEVERE", "0.85"))


def read_metrics(client: mlflow.MlflowClient, run_id: str) -> dict[str, float]:
    """Reads the metrics evaluate.py logged on the training run."""
    metrics = client.get_run(run_id).data.metrics
    log.info(f"Metrics on run {run_id}: {metrics}")
    return metrics


def meets_thresholds(metrics: dict[str, float]) -> tuple[bool, list[str]]:
    """Checks the metrics against the release thresholds.

    Returns (passed, reasons for refusal). A missing metric counts as a
    failure: evaluate.py may not have run, and an unmeasured model must not
    reach production.
    """
    reasons = []

    for name, minimum in (("auc_roc", MIN_AUC_ROC), ("recall", MIN_RECALL_SEVERE)):
        value = metrics.get(name)
        if value is None:
            reasons.append(f"{name} missing (did evaluate.py run?)")
        elif value < minimum:
            reasons.append(f"{name}={value:.4f} below the {minimum} threshold")

    return not reasons, reasons


def promote(client: mlflow.MlflowClient, version: str) -> None:
    """Moves the production alias onto the given version."""
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME, alias=PRODUCTION_ALIAS, version=version
    )
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME, version=version, key="stage", value="production"
    )
    log.info(
        f"Promoted {REGISTERED_MODEL_NAME} v{version} to "
        f"@{PRODUCTION_ALIAS}: the API serves it on its next reload"
    )


def main() -> int:
    for path in (RUN_ID_PATH, VERSION_PATH):
        if not path.exists():
            log.error(f"{path} not found: run `python -m src.training.train` first")
            return 1

    run_id = RUN_ID_PATH.read_text().strip()
    version = VERSION_PATH.read_text().strip()

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    client = mlflow.MlflowClient()

    metrics = read_metrics(client, run_id)
    passed, reasons = meets_thresholds(metrics)

    if not passed:
        current = "none"
        try:
            current = client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME, PRODUCTION_ALIAS
            ).version
        except Exception:
            pass
        log.error(
            f"v{version} not promoted: {'; '.join(reasons)}. "
            f"Production still serves v{current}."
        )
        return 1

    promote(client, version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
