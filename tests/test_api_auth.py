"""Tests for API authentication and authorization."""

import os
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# The API refuses to start without a strong signing key, so set one before
# importing the app.
os.environ.setdefault("JWT_SECRET_KEY", "test-key-that-is-long-enough-for-hs256-abc")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from src.api import auth as auth_module  # noqa: E402
from src.api import main as main_module  # noqa: E402

OPERATOR_PW = "operator-password"
ADMIN_PW = "admin-password-1"


@pytest.fixture
def client(monkeypatch):
    """API client backed by an in-memory database and a stub model."""
    engine = auth_module.get_engine(":memory:")

    def override_session():
        with Session(engine) as session:
            yield session

    main_module.app.dependency_overrides[auth_module.get_session] = override_session

    with Session(engine) as session:
        auth_module.create_user(session, "operator1", OPERATOR_PW, auth_module.Role.OPERATOR)
        auth_module.create_user(session, "admin1", ADMIN_PW, auth_module.Role.ADMIN)

    # Stub model: two classes, probability of the severe class above the
    # 0.30 threshold so the prediction is deterministic. predict_proba must
    # return a numpy array, since the endpoint calls .tolist() on the row.
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.8]])
    model.n_features_in_ = 40

    # lifespan would otherwise pull the real model from MLflow and overwrite
    # the stub, so stub the loader itself rather than the module globals.
    monkeypatch.setattr(main_module, "load_model", lambda: (model, "test"))

    with TestClient(main_module.app, raise_server_exceptions=False) as c:
        yield c

    main_module.app.dependency_overrides.clear()


def token_for(client, username, password) -> str:
    response = client.post("/token", data={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def auth_header(client, username, password) -> dict:
    return {"Authorization": f"Bearer {token_for(client, username, password)}"}


FEATURES = {"features": [0.0] * 40}


# --- Login -------------------------------------------------------------


def test_login_returns_token(client):
    response = client.post(
        "/token", data={"username": "operator1", "password": OPERATOR_PW}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["expires_in"] > 0


def test_login_wrong_password_is_rejected(client):
    response = client.post(
        "/token", data={"username": "operator1", "password": "wrong-password"}
    )
    assert response.status_code == 401


def test_login_unknown_user_gives_same_error_as_wrong_password(client):
    """The message must not reveal whether the username exists."""
    unknown = client.post("/token", data={"username": "ghost", "password": "whatever1234"})
    wrong = client.post("/token", data={"username": "operator1", "password": "wrong-pass12"})
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json()["detail"] == wrong.json()["detail"]


def test_login_disabled_account_is_rejected(client):
    headers = auth_header(client, "admin1", ADMIN_PW)
    assert client.delete("/admin/users/operator1", headers=headers).status_code == 200

    response = client.post(
        "/token", data={"username": "operator1", "password": OPERATOR_PW}
    )
    assert response.status_code == 401


# --- Protected endpoints ------------------------------------------------


def test_predict_without_token_is_rejected(client):
    assert client.post("/predict", json=FEATURES).status_code == 401


def test_predict_with_invalid_token_is_rejected(client):
    response = client.post(
        "/predict", json=FEATURES, headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert response.status_code == 401


def test_predict_with_valid_token_succeeds(client):
    response = client.post(
        "/predict", json=FEATURES, headers=auth_header(client, "operator1", OPERATOR_PW)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1  # 0.8 >= 0.30
    assert body["model_version"] == "test"


def test_expired_token_is_rejected(client, monkeypatch):
    monkeypatch.setattr(auth_module, "ACCESS_TOKEN_EXPIRE_MINUTES", -1)
    token, _ = auth_module.create_access_token("operator1", "operator")
    response = client.post(
        "/predict", json=FEATURES, headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()


def test_token_signed_with_another_key_is_rejected(client):
    import jwt

    forged = jwt.encode({"sub": "admin1", "role": "admin"}, "another-key", algorithm="HS256")
    response = client.post(
        "/predict", json=FEATURES, headers={"Authorization": f"Bearer {forged}"}
    )
    assert response.status_code == 401


def test_health_stays_public(client):
    assert client.get("/health").status_code == 200


# --- Roles --------------------------------------------------------------


def test_operator_cannot_create_users(client):
    response = client.post(
        "/admin/users",
        json={"username": "newbie", "password": "some-password", "role": "operator"},
        headers=auth_header(client, "operator1", OPERATOR_PW),
    )
    assert response.status_code == 403


def test_admin_can_create_and_list_users(client):
    headers = auth_header(client, "admin1", ADMIN_PW)
    created = client.post(
        "/admin/users",
        json={"username": "newbie", "password": "some-password", "role": "operator"},
        headers=headers,
    )
    assert created.status_code == 201
    assert created.json()["username"] == "newbie"

    listed = client.get("/admin/users", headers=headers)
    assert listed.status_code == 200
    assert "newbie" in [u["username"] for u in listed.json()]


def test_admin_cannot_deactivate_itself(client):
    response = client.delete(
        "/admin/users/admin1", headers=auth_header(client, "admin1", ADMIN_PW)
    )
    assert response.status_code == 400


def test_me_returns_the_caller(client):
    response = client.get("/me", headers=auth_header(client, "operator1", OPERATOR_PW))
    assert response.status_code == 200
    assert response.json() == {
        "username": "operator1",
        "role": "operator",
        "is_active": True,
    }


# --- Input validation ---------------------------------------------------


def test_wrong_feature_count_is_rejected(client):
    response = client.post(
        "/predict",
        json={"features": [0.0] * 10},
        headers=auth_header(client, "operator1", OPERATOR_PW),
    )
    assert response.status_code == 422
    assert "40" in response.json()["detail"]


def test_short_password_is_rejected(client):
    response = client.post(
        "/admin/users",
        json={"username": "newbie", "password": "short", "role": "operator"},
        headers=auth_header(client, "admin1", ADMIN_PW),
    )
    assert response.status_code == 422


def test_prediction_errors_do_not_leak_internals(client, monkeypatch):
    """A failure inside the model must not return a traceback to the caller."""
    headers = auth_header(client, "operator1", OPERATOR_PW)

    # Patch after the client is built: lifespan has already run, so the
    # module global is the model actually used by the endpoint.
    broken = MagicMock()
    broken.predict_proba.side_effect = RuntimeError("/secret/path/model.joblib missing")
    broken.n_features_in_ = 40
    monkeypatch.setattr(main_module, "model", broken)

    response = client.post("/predict", json=FEATURES, headers=headers)
    assert response.status_code == 500
    assert "secret" not in response.text
