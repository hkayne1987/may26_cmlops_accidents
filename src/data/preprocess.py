import logging
from pathlib import Path

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
SEP = ";"
TABLES = ["caracteristiques", "lieux", "vehicules", "usagers"]

TEST_SIZE = 0.2
RANDOM_STATE = 0
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"

# --- Définition des variables numériques (le reste sera catégoriel) -----------
NUMERIC_FEATURES = ["age", "heure", "minute", "jour", "mois", "an",
                    "vma", "nbv", "lartpc", "larrout", "occutc"]

# Colonnes à exclure des features (cible, identifiants, texte libre, redondances)
DROP_COLS = ["grav", "grave", "Num_Acc", "id_usager", "id_vehicule", "num_veh",
             "an_nais", "adr", "voie", "hrmn"]

# Harmonisation des noms de colonnes entre les années (drift de schéma BAAC)
# 2022 nomme l'identifiant d'accident "Accident_Id" au lieu de "Num_Acc".
RENAME_COLS = {
    "Accident_Id": "Num_Acc",
}


def load_year(year: int):
    """Charge les 4 tables d'une année, en str, avec suppression des espaces devant les nom.

    Retourne un dict {nom_table: DataFrame}.
    """
    tables = {}
    for name in TABLES:
        path = RAW_DIR / f"{name}-{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Fichier manquant : {path}")

        df = pd.read_csv(path, sep=SEP, dtype=str, encoding="utf-8")

        # Strip global sur toutes les colonnes texte
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()
            
        # Harmonisation des noms de colonnes
        df = df.rename(columns=RENAME_COLS)

        tables[name] = df
        log.info(f"{year} | {name:18s} : {df.shape[0]:>7} lignes, {df.shape[1]:>2} colonnes")

    return tables

def merge_tables(tables: dict[str, pd.DataFrame]):
    """Fusionne les 4 tables (1 ligne = 1 personne impliquée).

    - lieux est réduit à 1 ligne par accident (première occurrence : choix
      simple pour les accidents en intersection).
    - usagers x vehicules sur id_vehicule (clé unique sur 2019+).
    - puis x caracteristiques et x lieux sur Num_Acc.
    """
    carac = tables["caracteristiques"]
    lieux = tables["lieux"]
    vehicules = tables["vehicules"]
    usagers = tables["usagers"]

    # lieux : 1 ligne par accident (première occurrence)
    lieux_unique = lieux.drop_duplicates(subset="Num_Acc", keep="first").reset_index(drop=True)

    # On retire num_veh de vehicules car usagers le porte déjà
    veh = vehicules.drop(columns=[c for c in ["num_veh"] if c in vehicules.columns])

    n_usagers = len(usagers)

    # usagers x vehicules sur id_vehicule. On garde Num_Acc des deux côtés pour la jointure suivante : merge sur ["Num_Acc", "id_vehicule"].
    df = usagers.merge(veh, on=["Num_Acc", "id_vehicule"], how="left", validate="m:1")
    # x caracteristiques
    df = df.merge(carac, on="Num_Acc", how="left", validate="m:1")
    # x lieux
    df = df.merge(lieux_unique, on="Num_Acc", how="left", validate="m:1")

    # La fusion ne doit jamais changer le nombre de lignes
    if len(df) != n_usagers:
        raise ValueError(
            f"Fusion incohérente : {len(df)} lignes après merge "
            f"pour {n_usagers} usagers attendus."
        )

    return df



def build_target(df: pd.DataFrame):
    """Crée la cible binaire 'grave' contenant les cibles initiales tué (2) et hospitalisé (3).

    Retire les usagers sans gravité renseignée (grav = -1).
    """
    n_avant = len(df)
    df = df[df["grav"] != "-1"].copy()
    log.info(f"build_target | retirés (grav=-1) : {n_avant - len(df)}")

    df["grave"] = df["grav"].isin(["2", "3"]).astype(int)
    taux = df["grave"].mean()
    log.info(f"build_target | proportion 'grave' : {taux:.1%}")
    return df


def engineer_features(df: pd.DataFrame):
    """Crée âge et dates/horaires, puis applique le type de colonne
    (numérique vs catégoriel) attendu par XGBoost.
    """
    
    df = df.copy()

    # Âge au moment de l'accident
    df["an_nais"] = pd.to_numeric(df["an_nais"], errors="coerce")
    df["age"] = pd.to_numeric(df["an"], errors="coerce") - df["an_nais"]
    # Transforme en NaN les âges aberrants
    df.loc[(df["age"] < 0) | (df["age"] > 120), "age"] = pd.NA

    # Composantes horaires depuis hrmn ("HH:MM")
    df["heure"] = df["hrmn"].str.split(":").str[0]
    df["minute"] = df["hrmn"].str.split(":").str[1]

    return df


def build_feature_matrix(df: pd.DataFrame):
    """Sépare X / y et applique le type de feature attendue.

    Retourne (X, y, groups) :
      - X : features (numériques + catégorielles)
      - y : cible 'grave'
      - groups : Num_Acc (pour le split groupé à l'entraînement)
    """
    y = df["grave"]
    groups = df["Num_Acc"]

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].copy()

    categorical_features = [c for c in X.columns if c not in NUMERIC_FEATURES]

    # Typage : numériques ou catégorielles
    for c in NUMERIC_FEATURES:
        if c in X.columns:
            if X[c].dtype == "object":
                X[c] = pd.to_numeric(X[c].str.replace(",", ".", regex=False), errors="coerce")
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce")

    log.info(f"build_feature_matrix | X={X.shape}, "
             f"{len(categorical_features)} catégorielles, "
             f"{len([c for c in NUMERIC_FEATURES if c in X.columns])} numériques")
    return X, y, groups


def split_train_test(df: pd.DataFrame):
    """Split groupé par accident : tous les usagers d'un accident vont dans le même groupe (train ou test).

    Retourne (df_train, df_test).
    """
    groups = df["Num_Acc"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, df["grave"], groups=groups))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Vérification si accidents partagés entre train et test
    overlap = set(df_train["Num_Acc"]) & set(df_test["Num_Acc"])
    if overlap:
        raise ValueError(f"Fuite : {len(overlap)} accidents dans train ET test.")

    log.info(f"Split | train {len(df_train)} usagers ({df_train['Num_Acc'].nunique()} acc.) "
             f"/ test {len(df_test)} usagers ({df_test['Num_Acc'].nunique()} acc.)")
    log.info(f"Split | proportion 'grave' — train {df_train['grave'].mean():.1%} "
             f"/ test {df_test['grave'].mean():.1%}")
    return df_train, df_test


def preprocess_all():
    """Charge, fusionne et prétraite toutes les années, concatène le tout
    et écrit le dataset final dans data/processed/.
    """
    frames = []
    for year in YEARS:
        tables = load_year(year)
        df = merge_tables(tables)
        df = build_target(df)
        df = engineer_features(df)
        frames.append(df)
        log.info(f"--- {year} traité : {len(df)} usagers ---")

    full = pd.concat(frames, ignore_index=True)
    log.info(f"Concaténation : {full.shape[0]} usagers sur {len(YEARS)} années")
    
    # Fixer les catégories sur l'ensemble complet, avant le split, pour garantir
    # que train et test partagent le même schéma de features
    # (XGBoost categorical rejette toute catégorie vue au test mais absente du train).
    categorical_cols = [c for c in full.columns if c not in NUMERIC_FEATURES and c not in DROP_COLS]
    for c in categorical_cols:
        full[c] = full[c].astype(str).str.strip().astype("category")

    df_train, df_test = split_train_test(full)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(TRAIN_PATH, index=False)
    df_test.to_parquet(TEST_PATH, index=False)
    log.info(f"Écrit : {TRAIN_PATH} ({TRAIN_PATH.stat().st_size / 1e6:.1f} Mo)")
    log.info(f"Écrit : {TEST_PATH} ({TEST_PATH.stat().st_size / 1e6:.1f} Mo)")

    return df_train, df_test


if __name__ == "__main__":
    preprocess_all()