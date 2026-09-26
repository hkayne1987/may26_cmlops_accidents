"""Tests for API authentication and authorization."""

from unittest.mock import MagicMock

from src.api import auth as auth_module
from src.api import main as main_module
from tests.api_helpers import (
    ADMIN_PW,
    FEATURES,
    OPERATOR_PW,
    SERVICE_PW,
    auth_header,
)

# The `client` fixture lives in tests/conftest.py, which also imports
# api_helpers first so JWT_SECRET_KEY is set before the app loads.


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
    unknown = client.post(
        "/token", data={"username": "ghost", "password": "whatever1234"}
    )
    wrong = client.post(
        "/token", data={"username": "operator1", "password": "wrong-pass12"}
    )
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
    assert body["prediction"] == 1  # 0.8 >= 0.35
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

    forged = jwt.encode(
        {"sub": "admin1", "role": "admin"}, "another-key", algorithm="HS256"
    )
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


def test_service_can_reload_the_model(client):
    response = client.post(
        "/admin/reload-model", headers=auth_header(client, "airflow1", SERVICE_PW)
    )
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_service_cannot_manage_users(client):
    """The pipeline account must not be able to touch operator accounts."""
    response = client.post(
        "/admin/users",
        json={"username": "newbie", "password": "some-password", "role": "operator"},
        headers=auth_header(client, "airflow1", SERVICE_PW),
    )
    assert response.status_code == 403


def test_operator_cannot_reload_the_model(client):
    response = client.post(
        "/admin/reload-model", headers=auth_header(client, "operator1", OPERATOR_PW)
    )
    assert response.status_code == 403


def test_reload_requires_a_token(client):
    assert client.post("/admin/reload-model").status_code == 401


def test_service_can_predict(client):
    response = client.post(
        "/predict", json=FEATURES, headers=auth_header(client, "airflow1", SERVICE_PW)
    )
    assert response.status_code == 200


def test_empty_local_model_file_does_not_crash(tmp_path, monkeypatch):
    """An interrupted training run can leave a 0-byte model file behind."""
    empty = tmp_path / "xgb_severity.joblib"
    empty.write_bytes(b"")
    monkeypatch.setattr(main_module, "model_path", empty)

    loaded, version = main_module.load_model_from_disk()
    assert loaded is None
    assert version == "unknown"


def test_corrupt_local_model_file_does_not_crash(tmp_path, monkeypatch):
    corrupt = tmp_path / "xgb_severity.joblib"
    corrupt.write_bytes(b"not a joblib file")
    monkeypatch.setattr(main_module, "model_path", corrupt)

    loaded, version = main_module.load_model_from_disk()
    assert loaded is None
    assert version == "unknown"


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
