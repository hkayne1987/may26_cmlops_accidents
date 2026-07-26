"""
Evaluation of the accident severity model.
Loads the trained model (models/) and the test set (data/processed/test.parquet).
Computes metrics and the precision/recall trade-off by threshold.

Run: python -m src.training.evaluate
"""

import logging
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score

from src.data.preprocess import build_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# --- Configuration ------------------------------------------------------------
TEST_PATH = Path("data/processed/test.parquet")
MODEL_PATH = Path("models/xgb_severity.joblib")
RUN_ID_PATH = Path("models/run_id.txt")  # written by train.py
DECISION_THRESHOLD = 0.30   # retained threshold (favors recall on the severe class)


def load_test() -> pd.DataFrame:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test set not found: {TEST_PATH}. "
                                f"Run `python -m src.data.preprocess` first.")
    df = pd.read_parquet(TEST_PATH)
    log.info(f"Test set loaded: {df.shape[0]} rows")
    return df


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. "
                                f"Run `python -m src.training.train` first.")
    return joblib.load(MODEL_PATH)


def evaluate(model, X_test, y_test, threshold: float = DECISION_THRESHOLD) -> dict:
    """Evaluates the model at the given threshold + AUC (threshold-independent)."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    log.info(f"\n=== Report (threshold = {threshold}) ===\n"
             + classification_report(y_test, y_pred, target_names=["non severe", "severe"]))
    log.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
    log.info(f"AUC-ROC: {auc:.3f}")

    return {
        "threshold": threshold,
        "auc_roc": round(auc, 4),
        "recall_severe": round(recall_score(y_test, y_pred), 4),
        "precision_severe": round(precision_score(y_test, y_pred), 4),
        "f1_severe": round(f1_score(y_test, y_pred), 4),
    }


def threshold_search(model, X_test, y_test):
    """Scans several thresholds to find the precision/recall trade-off."""
    y_proba = model.predict_proba(X_test)[:, 1]
    log.info("\n=== Threshold scan ===")
    rows = []
    for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
        y_pred = (y_proba >= t).astype(int)
        r = recall_score(y_test, y_pred)
        p = precision_score(y_test, y_pred)
        f = f1_score(y_test, y_pred)
        log.info(f"threshold={t:.2f} | recall={r:.3f} | precision={p:.3f} | f1={f:.3f}")
        rows.append({"threshold": t, "recall": round(r, 4),
                     "precision": round(p, 4), "f1": round(f, 4)})
    return rows


if __name__ == "__main__":
    df = load_test()
    X_test, y_test, _ = build_feature_matrix(df)

    model = load_model()
    metrics = evaluate(model, X_test, y_test)
    threshold = threshold_search(model, X_test, y_test)

    # Log metrics into the MLflow run created by train.py, if available
    if RUN_ID_PATH.exists():
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
        run_id = RUN_ID_PATH.read_text().strip()
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("auc_roc", metrics["auc_roc"])
            mlflow.log_metric("recall", metrics["recall_severe"])
            mlflow.log_metric("precision", metrics["precision_severe"])
            mlflow.log_table(
                data=pd.DataFrame(threshold),
                artifact_file="threshold_scan.json",
            )
        log.info(f"Metrics and threshold scan logged to MLflow run {run_id}")
    else:
        log.warning(f"{RUN_ID_PATH} not found: skipping MLflow logging "
                    f"(run `python -m src.training.train` first)")
