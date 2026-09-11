"""Command-line user management for the inference API.

Needed to create the first admin account: /admin/users is itself
admin-only, so the database would otherwise stay empty and unusable.

Usage:
    python -m src.api.manage_users create <username> --role admin
    python -m src.api.manage_users list
    python -m src.api.manage_users disable <username>

The password is read interactively so it never lands in the shell history.
"""

import argparse
import getpass
import logging
import sys

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.auth import MIN_PASSWORD_LENGTH, Role, User, create_user, get_engine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def prompt_password() -> str:
    """Asks for a password twice and checks the two entries match."""
    password = getpass.getpass("Password: ")
    if len(password) < MIN_PASSWORD_LENGTH:
        sys.exit(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
    if password != getpass.getpass("Confirm password: "):
        sys.exit("Passwords do not match")
    return password


def cmd_create(args) -> None:
    with Session(get_engine()) as session:
        try:
            user = create_user(
                session, args.username, prompt_password(), Role(args.role)
            )
        except ValueError as e:
            sys.exit(str(e))
        print(f"Created {user.username} (role={user.role})")


def cmd_list(args) -> None:
    with Session(get_engine()) as session:
        users = session.scalars(select(User).order_by(User.username)).all()
        if not users:
            print("No users yet. Create one with: create <username> --role admin")
            return
        for u in users:
            state = "active" if u.is_active else "disabled"
            print(f"{u.username:20s} {u.role:10s} {state}")


def cmd_disable(args) -> None:
    with Session(get_engine()) as session:
        user = session.get(User, args.username)
        if user is None:
            sys.exit(f"User {args.username!r} not found")
        user.is_active = False
        session.commit()
        print(f"Disabled {user.username}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage API user accounts")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="create a user")
    p_create.add_argument("username")
    p_create.add_argument(
        "--role", choices=[r.value for r in Role], default=Role.OPERATOR.value
    )
    p_create.set_defaults(func=cmd_create)

    p_list = sub.add_parser("list", help="list users")
    p_list.set_defaults(func=cmd_list)

    p_disable = sub.add_parser("disable", help="disable a user")
    p_disable.add_argument("username")
    p_disable.set_defaults(func=cmd_disable)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
