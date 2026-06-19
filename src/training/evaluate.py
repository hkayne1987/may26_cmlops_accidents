"""
Évaluation du modèle de gravité des accidents.
Charge le modèle entraîné (models/) et le jeu de test (data/processed/test.parquet),
calcule les métriques et le compromis precision/recall selon le seuil.

Exécution : python -m src.training.evaluate
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score

from src.data.preprocess import build_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# --- Configuration ------------------------------------------------------------
TEST_PATH = Path("data/processed/test.parquet")
MODEL_PATH = Path("models/xgb_gravite.joblib")
METRICS_PATH = Path("models/metrics.json")
DECISION_THRESHOLD = 0.30   # seuil retenu (favorise le recall sur la classe grave)


def load_test() -> pd.DataFrame:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Jeu de test introuvable : {TEST_PATH}. "
                                f"Lance d'abord `python -m src.data.preprocess`.")
    df = pd.read_parquet(TEST_PATH)
    log.info(f"Test chargé : {df.shape[0]} lignes")
    return df


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}. "
                                f"Lance d'abord `python -m src.training.train`.")
    return joblib.load(MODEL_PATH)


def evaluate(model, X_test, y_test, threshold: float = DECISION_THRESHOLD) -> dict:
    """Évalue le modèle au seuil donné + AUC (indépendant du seuil)."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    log.info(f"\n=== Rapport (seuil = {threshold}) ===\n"
             + classification_report(y_test, y_pred, target_names=["non grave", "grave"]))
    log.info(f"Matrice de confusion :\n{confusion_matrix(y_test, y_pred)}")
    log.info(f"AUC-ROC : {auc:.3f}")

    return {
        "threshold": threshold,
        "auc_roc": round(auc, 4),
        "recall_grave": round(recall_score(y_test, y_pred), 4),
        "precision_grave": round(precision_score(y_test, y_pred), 4),
        "f1_grave": round(f1_score(y_test, y_pred), 4),
    }


def threshold_search(model, X_test, y_test):
    """Balaye plusieurs seuils pour trouver le compromis precision/recall."""
    y_proba = model.predict_proba(X_test)[:, 1]
    log.info("\n=== Balayage de seuils ===")
    rows = []
    for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
        y_pred = (y_proba >= t).astype(int)
        r = recall_score(y_test, y_pred)
        p = precision_score(y_test, y_pred)
        f = f1_score(y_test, y_pred)
        log.info(f"seuil={t:.2f} | recall={r:.3f} | precision={p:.3f} | f1={f:.3f}")
        rows.append({"threshold": t, "recall": round(r, 4),
                     "precision": round(p, 4), "f1": round(f, 4)})
    return rows


if __name__ == "__main__":
    df = load_test()
    X_test, y_test, _ = build_feature_matrix(df)

    model = load_model()
    metrics = evaluate(model, X_test, y_test)
    threshold = threshold_search(model, X_test, y_test)

    # Sauvegarde des métriques
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump({"main": metrics, "threshold_search": threshold}, f, indent=2)
    log.info(f"\nMétriques sauvegardées : {METRICS_PATH}")