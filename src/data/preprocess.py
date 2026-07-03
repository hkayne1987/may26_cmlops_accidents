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

# --- Numeric variable definitions (the rest will be categorical) ---------------
NUMERIC_FEATURES = ["age", "heure", "minute", "jour", "mois", "an",
                    "vma", "nbv", "lartpc", "larrout", "occutc"]

# Columns to exclude from features (target, identifiers, free text, redundancies)
DROP_COLS = ["grav", "grave", "Num_Acc", "id_usager", "id_vehicule", "num_veh",
             "an_nais", "adr", "voie", "hrmn"]

# Column name harmonization across years (BAAC schema drift)
# 2022 names the accident identifier "Accident_Id" instead of "Num_Acc".
RENAME_COLS = {
    "Accident_Id": "Num_Acc",
}


def load_year(year: int):
    """Loads the 4 tables for a given year as strings, stripping leading spaces from column names.

    Returns a dict {table_name: DataFrame}.
    """
    tables = {}
    for name in TABLES:
        path = RAW_DIR / f"{name}-{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path, sep=SEP, dtype=str, encoding="utf-8")

        # Global strip on all text columns
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()

        # Column name harmonization
        df = df.rename(columns=RENAME_COLS)

        tables[name] = df
        log.info(f"{year} | {name:18s} : {df.shape[0]:>7} rows, {df.shape[1]:>2} columns")

    return tables

def merge_tables(tables: dict[str, pd.DataFrame]):
    """Merges the 4 tables (1 row = 1 person involved).

    - lieux is reduced to 1 row per accident (first occurrence: simple choice for intersection accidents).
    - usagers x vehicules on id_vehicule (unique key from 2019+).
    - then x caracteristiques and x lieux on Num_Acc.
    """
    carac = tables["caracteristiques"]
    lieux = tables["lieux"]
    vehicules = tables["vehicules"]
    usagers = tables["usagers"]

    # lieux: 1 row per accident (first occurrence)
    lieux_unique = lieux.drop_duplicates(subset="Num_Acc", keep="first").reset_index(drop=True)

    # Drop num_veh from vehicules since usagers already carries it
    veh = vehicules.drop(columns=[c for c in ["num_veh"] if c in vehicules.columns])

    n_usagers = len(usagers)

    # usagers x vehicules on id_vehicule. Keep Num_Acc on both sides for the next join: merge on ["Num_Acc", "id_vehicule"].
    df = usagers.merge(veh, on=["Num_Acc", "id_vehicule"], how="left", validate="m:1")
    # x caracteristiques
    df = df.merge(carac, on="Num_Acc", how="left", validate="m:1")
    # x lieux
    df = df.merge(lieux_unique, on="Num_Acc", how="left", validate="m:1")

    # The merge must never change the number of rows
    if len(df) != n_usagers:
        raise ValueError(
            f"Inconsistent merge: {len(df)} rows after merge "
            f"for {n_usagers} expected users."
        )

    return df



def build_target(df: pd.DataFrame):
    """Creates the binary target 'grave' covering the initial targets killed (2) and hospitalized (3).

    Removes users with no recorded severity (grav = -1).
    """
    n_avant = len(df)
    df = df[df["grav"] != "-1"].copy()
    log.info(f"build_target | removed (grav=-1): {n_avant - len(df)}")

    df["grave"] = df["grav"].isin(["2", "3"]).astype(int)
    taux = df["grave"].mean()
    log.info(f"build_target | proportion 'severe': {taux:.1%}")
    return df


def engineer_features(df: pd.DataFrame):
    """Creates age and date/time features, then applies the column types
    (numeric vs categorical) expected by XGBoost.
    """

    df = df.copy()

    # Age at the time of the accident
    df["an_nais"] = pd.to_numeric(df["an_nais"], errors="coerce")
    df["age"] = pd.to_numeric(df["an"], errors="coerce") - df["an_nais"]
    # Set outlier ages to NaN
    df.loc[(df["age"] < 0) | (df["age"] > 120), "age"] = pd.NA

    # Time components from hrmn ("HH:MM")
    df["heure"] = df["hrmn"].str.split(":").str[0]
    df["minute"] = df["hrmn"].str.split(":").str[1]

    return df


def build_feature_matrix(df: pd.DataFrame):
    """Separates X / y and applies the expected feature types.

    Returns (X, y, groups):
      - X: features (numeric + categorical)
      - y: target 'grave'
      - groups: Num_Acc (for grouped split at training time)
    """
    y = df["grave"]
    groups = df["Num_Acc"]

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].copy()

    categorical_features = [c for c in X.columns if c not in NUMERIC_FEATURES]

    # Typing: numeric or categorical
    for c in NUMERIC_FEATURES:
        if c in X.columns:
            if X[c].dtype == "object":
                X[c] = pd.to_numeric(X[c].str.replace(",", ".", regex=False), errors="coerce")
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce")

    log.info(f"build_feature_matrix | X={X.shape}, "
             f"{len(categorical_features)} categorical, "
             f"{len([c for c in NUMERIC_FEATURES if c in X.columns])} numeric")
    return X, y, groups


def split_train_test(df: pd.DataFrame):
    """Grouped split by accident: all users from an accident go into the same group (train or test).

    Returns (df_train, df_test).
    """
    groups = df["Num_Acc"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, df["grave"], groups=groups))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Check for accidents shared between train and test
    overlap = set(df_train["Num_Acc"]) & set(df_test["Num_Acc"])
    if overlap:
        raise ValueError(f"Data leak: {len(overlap)} accidents in both train AND test.")

    log.info(f"Split | train {len(df_train)} users ({df_train['Num_Acc'].nunique()} acc.) "
             f"/ test {len(df_test)} users ({df_test['Num_Acc'].nunique()} acc.)")
    log.info(f"Split | proportion 'severe' — train {df_train['grave'].mean():.1%} "
             f"/ test {df_test['grave'].mean():.1%}")
    return df_train, df_test


def preprocess_all():
    """Loads, merges and preprocesses all years, concatenates everything
    and writes the final dataset to data/processed/.
    """
    frames = []
    for year in YEARS:
        tables = load_year(year)
        df = merge_tables(tables)
        df = build_target(df)
        df = engineer_features(df)
        frames.append(df)
        log.info(f"--- {year} processed: {len(df)} users ---")

    full = pd.concat(frames, ignore_index=True)
    log.info(f"Concatenation: {full.shape[0]} users over {len(YEARS)} years")

    # Fix categories on the full dataset before splitting, to ensure
    # that train and test share the same feature schema
    # (XGBoost categorical rejects any category seen in test but absent from train).
    categorical_cols = [c for c in full.columns if c not in NUMERIC_FEATURES and c not in DROP_COLS]
    for c in categorical_cols:
        full[c] = full[c].astype(str).str.strip().astype("category")

    df_train, df_test = split_train_test(full)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(TRAIN_PATH, index=False)
    df_test.to_parquet(TEST_PATH, index=False)
    log.info(f"Written: {TRAIN_PATH} ({TRAIN_PATH.stat().st_size / 1e6:.1f} MB)")
    log.info(f"Written: {TEST_PATH} ({TEST_PATH.stat().st_size / 1e6:.1f} MB)")

    return df_train, df_test


if __name__ == "__main__":
    preprocess_all()
