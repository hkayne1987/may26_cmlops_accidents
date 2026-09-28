"""Fixtures shared across test modules."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.api_helpers import (
    ADMIN_PW,
    OPERATOR_PW,
    SERVICE_PW,
    STUB_CATEGORIES,
    STUB_NAMES,
)


@pytest.fixture
def client(monkeypatch):
    """API client backed by an in-memory database and a stub model."""
    # Imported here so only the tests that use the API pay for loading it.
    from fastapi.testclient import TestClient
    from sqlalchemy.orm import Session

    from src.api import auth as auth_module
    from src.api import main as main_module
    from src.api.schema import FeatureSchema

    engine = auth_module.get_engine(":memory:")

    def override_session():
        with Session(engine) as session:
            yield session

    main_module.app.dependency_overrides[auth_module.get_session] = override_session

    with Session(engine) as session:
        auth_module.create_user(
            session, "operator1", OPERATOR_PW, auth_module.Role.OPERATOR
        )
        auth_module.create_user(session, "admin1", ADMIN_PW, auth_module.Role.ADMIN)
        auth_module.create_user(
            session, "airflow1", SERVICE_PW, auth_module.Role.SERVICE
        )

    # Stub model: two classes, probability of the severe class above the
    # 0.35 threshold so the prediction is deterministic. predict_proba must
    # return a numpy array, since the endpoint calls .tolist() on the row.
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.8]])

    # lifespan would otherwise pull the real model from MLflow and overwrite
    # the stub, so stub the loader itself rather than the module globals.
    # A stub has no XGBoost booster to read a schema from, so give it one.
    monkeypatch.setattr(main_module, "load_model", lambda: (model, "test"))
    monkeypatch.setattr(
        main_module,
        "schema_from_model",
        lambda m: FeatureSchema(names=STUB_NAMES, categories=STUB_CATEGORIES),
    )

    with TestClient(main_module.app, raise_server_exceptions=False) as c:
        yield c

    main_module.app.dependency_overrides.clear()
