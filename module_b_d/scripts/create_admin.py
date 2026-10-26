"""Create the initial administrator account (explicit, one-time setup step).

Credentials are NEVER hard-coded. Provide them either through environment
variables or interactively:

    # Option 1 - interactive (password is not echoed)
    python -m scripts.create_admin

    # Option 2 - environment variables (e.g. in CI or a setup script)
    QGENAI_ADMIN_USERNAME=admin QGENAI_ADMIN_PASSWORD='...' python -m scripts.create_admin

The script refuses to run if any administrator already exists, so it cannot
be used to silently take over an installed system.
"""
import argparse
import getpass
import os
import sys

from sqlalchemy import select

from app.config import get_settings
from app.database import SessionLocal, init_db
from app.models.user import Role, User
from app.services.audit_service import append_event, audited_transaction
from app.services.authentication import hash_password, validate_password_strength
from app.services.policy_service import ensure_default_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the first Q-GenAI administrator.")
    parser.add_argument("--username", default=os.environ.get("QGENAI_ADMIN_USERNAME"))
    parser.add_argument("--full-name", default=os.environ.get("QGENAI_ADMIN_FULL_NAME", "Administrator"))
    parser.add_argument("--site", default=os.environ.get("QGENAI_ADMIN_SITE", "default"))
    args = parser.parse_args()

    get_settings().storage_dir.mkdir(parents=True, exist_ok=True)
    init_db()

    username = args.username or input("Admin username: ").strip()
    password = os.environ.get("QGENAI_ADMIN_PASSWORD")
    if not password:
        password = getpass.getpass("Admin password (min 10 chars, letters + digits): ")
        if password != getpass.getpass("Repeat password: "):
            print("Passwords do not match.", file=sys.stderr)
            return 1
    try:
        validate_password_strength(password)
    except ValueError as exc:
        print(f"Weak password: {exc}", file=sys.stderr)
        return 1

    with SessionLocal() as db:
        ensure_default_policy(db)
        if db.execute(select(User.id).where(User.role == Role.ADMIN)).first():
            print("An administrator already exists. Use the admin UI/API to manage users.",
                  file=sys.stderr)
            return 1
        if db.execute(select(User.id).where(User.username == username)).first():
            print("That username is already taken.", file=sys.stderr)
            return 1
        with audited_transaction(db):
            user = User(username=username, full_name=args.full_name, role=Role.ADMIN,
                        site=args.site, password_hash=hash_password(password))
            db.add(user)
            db.flush()
            append_event(db, "user.bootstrap_admin_created", target_type="user", target_id=user.id,
                         metadata={"username": username, "role": "admin", "via": "create_admin script"})
    print(f"Administrator '{username}' created.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
