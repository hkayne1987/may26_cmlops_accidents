"""Constants and helpers shared by the API test modules."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# The API refuses to start without a strong signing key, so set one before
# the app is imported anywhere.
os.environ.setdefault("JWT_SECRET_KEY", "test-key-that-is-long-enough-for-hs256-abc")

OPERATOR_PW = "operator-password"
ADMIN_PW = "admin-password-1"
SERVICE_PW = "service-password"

# A small stand-in for the real 40-column schema, used with the stub model:
# two categorical BAAC columns and one numeric one are enough to exercise
# validation and conversion.
STUB_CATEGORIES = {"catv": ["-1", "1", "16", "7"], "lum": ["1", "2", "3"]}
STUB_NAMES = ["catv", "lum", "vma"]

FEATURES = {"features": {"catv": "7", "lum": "1", "vma": 80}}


def token_for(client, username, password) -> str:
    response = client.post("/token", data={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def auth_header(client, username, password) -> dict:
    return {"Authorization": f"Bearer {token_for(client, username, password)}"}
