"""
Entraînement du modèle de gravité des accidents (XGBoost).
Lit le dataset prétraité (data/processed/), split groupé par accident,
entraîne, et sauvegarde le modèle dans models/.

Exécution via 'python -m src.training.train'.
"""

import logging
from pathlib import Path

import pandas as pd

from src.data.preprocess import build_feature_matrix

from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TRAIN_PATH = Path("data/processed/train.parquet")
MODEL_DIR = Path("models")
TEST_SIZE = 0.2
RANDOM_STATE = 0

# Hyperparamètres retenus suite à un GridSearchCV (scoring="f1", 3-fold StratifiedGroupKFold) mené sur l'année 2023 uniquement (~125k usagers).
# Grille testée : max_depth=[3, 5, 7], learning_rate=[0.05, 0.1, 0.2], n_estimators=[50, 100, 200, 300].
# Meilleur score F1 obtenu : 0.586.
# Voir tune_hyperparameters() pour reproduire ou ré-optimiser sur le dataset complet 2019-2024.
HYPERPARAMS = {
    "max_depth": 3,
    "learning_rate": 0.2,
    "n_estimators": 300,
}


def load_train() -> pd.DataFrame:
    """Charge le jeu d'entraînement prétraité."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Jeu d'entraînement introuvable : {TRAIN_PATH}. "
            f"Lance d'abord `python -m src.data.preprocess`."
        )
    df = pd.read_parquet(TRAIN_PATH)
    log.info(f"Train chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def train_model(X_train, y_train) -> XGBClassifier:
    """Entraîne XGBoost avec gestion du déséquilibre (scale_pos_weight)."""
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    log.info(f"scale_pos_weight = {scale:.2f}")

    model = XGBClassifier(
        **HYPERPARAMS,
        scale_pos_weight=scale,
        enable_categorical=True,
        tree_method="hist",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    log.info("Entraînement terminé.")
    return model


def tune_hyperparameters(X_train, y_train, groups_train):
    """Recherche d'hyperparamètres par grid search (optionnel, non exécuté par défaut).
    Prend du temps.
    Usage : python -m src.training.train --tune
    """
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

    scale = (y_train == 0).sum() / (y_train == 1).sum()
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 200, 300],
    }
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        XGBClassifier(enable_categorical=True, tree_method="hist",
                      eval_metric="logloss", random_state=RANDOM_STATE),
        param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=2,
    )
    grid.fit(X_train, y_train, groups=groups_train)

    log.info(f"Meilleurs paramètres : {grid.best_params_}")
    log.info(f"Meilleur score F1 : {grid.best_score_:.4f}")
    return grid.best_params_


def save_model(model, X_train):
    """Sauvegarde le modèle + la liste ordonnée des colonnes attendues à l'inférence."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "xgb_gravite.joblib"
    joblib.dump(model, model_path)
    log.info(f"Modèle sauvegardé : {model_path}")

    # On sauve aussi le schéma des colonnes
    # Indispensable pour l'API qui devra reconstruire un DataFrame avec exactement ces colonnes
    schema_path = MODEL_DIR / "feature_columns.json"
    pd.Series(X_train.columns).to_json(schema_path, orient="values")
    log.info(f"Schéma des features sauvegardé : {schema_path}")


if __name__ == "__main__":
    import sys

    df = load_train()

    if "--sample" in sys.argv:
        df = df.sample(n=10000, random_state=RANDOM_STATE)
        log.info(f"Mode --sample : réduit à {len(df)} lignes")

    X, y, groups = build_feature_matrix(df)
    log.info(f"X={X.shape}, y={y.shape}")

    if "--tune" in sys.argv:
        best_params = tune_hyperparameters(X, y, groups)
        log.info(f"Pour utiliser ces paramètres : {best_params}")
    else:
        model = train_model(X, y)
        save_model(model, X)
        log.info("Entraînement terminé.")