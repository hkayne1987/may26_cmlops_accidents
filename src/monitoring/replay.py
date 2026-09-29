"""Replays real BAAC accidents through the API, as call centre operators would.

The API gets no real traffic in this project, so drift detection has nothing
to look at. This sends rows of the test set to /predict, which logs them like
any real request. A scenario restricts the rows to one kind of accident to
make the incoming data drift on purpose, for demonstration.

Run: python -m src.monitoring.replay --rows 500 --scenario motorway
Credentials come from API_USERNAME / API_PASSWORD, as for the Airflow account.
"""

import argparse
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.preprocess import build_feature_matrix

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

TEST_PATH = Path("data/processed/test.parquet")
# Not read from API_URL: that variable is Airflow's address for the API inside
# the Docker network (http://api:8000), unreachable from the host.
DEFAULT_URL = "http://localhost:8000"

# Each scenario keeps one kind of accident. "normal" keeps everything, so the
# requests look like the training data and should not drift.
SCENARIOS: dict[str, tuple[str, Any]] = {
    "normal": ("", None),
    "motorway": ("catr", {"1"}),
    "pedestrians": ("catu", {"3"}),
    "night": ("lum", {"2", "3", "4", "5"}),
    "two_wheelers": ("catv", {"2", "30", "31", "32", "33", "34"}),
}


def select_rows(scenario: str, n_rows: int, seed: int = 0) -> pd.DataFrame:
    """Picks test-set rows for the scenario, typed as the model sees them."""
    X, _, _ = build_feature_matrix(pd.read_parquet(TEST_PATH))
    column, values = SCENARIOS[scenario]
    if column:
        X = X[X[column].astype(str).isin(values)]
    return X.sample(min(n_rows, len(X)), random_state=seed)


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        return value
    number = float(value)
    return int(number) if number.is_integer() else number


def build_requests(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as operators would send them: real values, null when unknown."""
    categorical = {c for c in rows.columns if str(rows[c].dtype) == "category"}
    requests: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        requests.append(
            {
                str(c): (None if pd.isna(v) else str(v))
                if c in categorical
                else _json_value(v)
                for c, v in row.items()
            }
        )
    return requests


def _post(url: str, path: str, data: bytes, headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(
        url + path, data=data, headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def login(url: str, username: str, password: str) -> str:
    body = urllib.parse.urlencode({"username": username, "password": password})
    return _post(
        url,
        "/token",
        body.encode(),
        {"Content-Type": "application/x-www-form-urlencoded"},
    )["access_token"]


def replay(url: str, requests: list[dict[str, Any]], token: str) -> tuple[int, int]:
    """Sends the requests. Returns (severe predictions, failures)."""
    severe = failures = 0
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    for i, features in enumerate(requests, 1):
        try:
            result = _post(
                url, "/predict", json.dumps({"features": features}).encode(), headers
            )
            severe += result["prediction"]
        except urllib.error.HTTPError as e:
            failures += 1
            log.warning(f"Request {i} refused ({e.code}): {e.read()[:200]!r}")
        if i % 100 == 0:
            log.info(f"{i}/{len(requests)} sent")
    return severe, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="normal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--url", default=DEFAULT_URL, help="API base URL")
    args = parser.parse_args()

    username = os.environ.get("API_USERNAME", "")
    password = os.environ.get("API_PASSWORD", "")
    if not username or not password:
        raise SystemExit("Set API_USERNAME and API_PASSWORD (see env.example)")

    rows = select_rows(args.scenario, args.rows, args.seed)
    log.info(f"Replaying {len(rows)} '{args.scenario}' accidents to {args.url}")
    token = login(args.url, username, password)
    severe, failures = replay(args.url, build_requests(rows), token)
    log.info(
        f"Done: {len(rows) - failures} predictions, {severe} flagged severe, "
        f"{failures} refused"
    )


if __name__ == "__main__":
    main()
