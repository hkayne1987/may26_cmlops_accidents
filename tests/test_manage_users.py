"""Tests for the account maintenance helpers behind src/api/manage_users.py."""

import pytest
from sqlalchemy.orm import Session

import tests.api_helpers  # noqa: F401  (sets JWT_SECRET_KEY first)
from src.api import auth


@pytest.fixture
def session():
    with Session(auth.get_engine(":memory:")) as s:
        auth.create_user(s, "alice", "first-password-1", auth.Role.OPERATOR)
        yield s


def test_reset_password_replaces_the_old_one(session):
    auth.set_password(session, "alice", "second-password-2")
    assert auth.authenticate_user(session, "alice", "second-password-2") is not None
    assert auth.authenticate_user(session, "alice", "first-password-1") is None


def test_reset_password_keeps_the_length_rule(session):
    with pytest.raises(ValueError, match="at least"):
        auth.set_password(session, "alice", "short")


def test_disable_then_enable(session):
    auth.set_active(session, "alice", False)
    assert auth.authenticate_user(session, "alice", "first-password-1") is None
    auth.set_active(session, "alice", True)
    assert auth.authenticate_user(session, "alice", "first-password-1") is not None


def test_unknown_user_is_reported(session):
    with pytest.raises(ValueError, match="not found"):
        auth.set_password(session, "bob", "any-password-123")
    with pytest.raises(ValueError, match="not found"):
        auth.set_active(session, "bob", True)
