"""Authentication and authorization for the inference API.

Users live in a SQLite database so accounts can be created and disabled
without restarting the service: emergency call centre staff changes over
time, and a redeploy per arrival or departure is not workable.

Login exchanges a username and password for a signed JWT (OAuth2 password
flow). Protected endpoints then expect an `Authorization: Bearer <token>`
header. Roles are carried in the token and checked per endpoint.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from sqlalchemy import Boolean, DateTime, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool

log = logging.getLogger(__name__)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "api_users.db"

# bcrypt hashes at most 72 bytes and raises above that, so reject longer
# passwords up front instead of silently truncating them.
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LENGTH = 12


class Role(str, Enum):
    """Who can do what.

    operator: runs predictions. Emergency call centre staff, and machine
              accounts such as Airflow, which are told apart by their
              username rather than by a separate role.
    admin:    manages user accounts on top of operator rights.
    """

    OPERATOR = "operator"
    ADMIN = "admin"


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(64), primary_key=True)
    hashed_password: Mapped[str] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(String(16), default=Role.OPERATOR.value)
    # Accounts are disabled rather than deleted, so past predictions stay
    # attributable to a real user.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserInfo(BaseModel):
    username: str
    role: Role
    is_active: bool


def get_secret_key() -> str:
    """Returns the JWT signing key, refusing to start without a strong one.

    A default value here would mean anyone reading the repo could mint valid
    tokens, so an unset or too-short key is a startup failure, not a warning.
    """
    key = os.environ.get("JWT_SECRET_KEY", "")
    if len(key) < 32:
        raise RuntimeError(
            "JWT_SECRET_KEY must be set to at least 32 characters. Generate "
            'one with: python -c "import secrets; print(secrets.token_urlsafe(48))"'
        )
    return key


def hash_password(password: str) -> str:
    """Hashes a password with bcrypt (per-password salt included)."""
    encoded = password.encode()
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise ValueError(f"Password must be at most {MAX_PASSWORD_BYTES} bytes")
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    """Checks a password against its bcrypt hash."""
    encoded = password.encode()
    if len(encoded) > MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(encoded, hashed.encode())
    except ValueError:
        # Malformed hash in the database: treat as a failed login.
        return False


def get_engine(db_path: str | None = None):
    """Creates the SQLite engine and the users table if needed.

    check_same_thread=False is required because FastAPI runs synchronous
    endpoints in a thread pool, so a connection is not always reused from
    the thread that opened it. StaticPool keeps an in-memory database alive
    across those threads, which would otherwise be empty on every new
    connection.
    """
    path = db_path or os.environ.get("API_USERS_DB", str(DEFAULT_DB_PATH))
    in_memory = path == ":memory:"
    if not in_memory:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{path}",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool if in_memory else None,
    )
    Base.metadata.create_all(engine)
    return engine


_engine = None


def get_session() -> Session:
    """Yields a database session (FastAPI dependency)."""
    global _engine
    if _engine is None:
        _engine = get_engine()
    with Session(_engine) as session:
        yield session


def create_user(session: Session, username: str, password: str, role: Role) -> User:
    """Creates a user. Raises ValueError if the name is taken."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters"
        )
    if session.get(User, username) is not None:
        raise ValueError(f"User {username!r} already exists")

    user = User(
        username=username,
        hashed_password=hash_password(password),
        role=role.value,
    )
    session.add(user)
    session.commit()
    log.info(f"User created: {username} (role={role.value})")
    return user


def authenticate_user(session: Session, username: str, password: str) -> User | None:
    """Returns the user when the credentials are valid, else None."""
    user = session.get(User, username)
    if user is None:
        # Hash anyway so a missing user and a wrong password take a similar
        # amount of time, which avoids leaking which usernames exist.
        bcrypt.hashpw(b"dummy", bcrypt.gensalt())
        return None
    if not user.is_active:
        log.warning(f"Login refused for disabled account: {username}")
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(username: str, role: str) -> tuple[str, int]:
    """Signs a JWT for the given user. Returns (token, lifetime in seconds)."""
    expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.now(timezone.utc) + expires_delta
    payload = {"sub": username, "role": role, "exp": expire}
    token = jwt.encode(payload, get_secret_key(), algorithm=ALGORITHM)
    return token, int(expires_delta.total_seconds())


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

_CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> User:
    """Resolves the caller from the bearer token (FastAPI dependency)."""
    try:
        payload = jwt.decode(token, get_secret_key(), algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise _CREDENTIALS_ERROR

    username = payload.get("sub")
    if not username:
        raise _CREDENTIALS_ERROR

    # Re-read the account: a token stays valid until it expires, so a
    # disabled user must be rejected here rather than trusted from the token.
    user = session.get(User, username)
    if user is None or not user.is_active:
        raise _CREDENTIALS_ERROR
    return user


def require_role(*allowed: Role):
    """Builds a dependency that only lets the given roles through."""

    def checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in {r.value for r in allowed}:
            log.warning(
                f"Access denied for {user.username} (role={user.role}), "
                f"needs one of {[r.value for r in allowed]}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient privileges for this operation",
            )
        return user

    return checker
