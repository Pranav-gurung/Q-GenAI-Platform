"""Section D: password hashing (Argon2id) and short-lived JWT access tokens."""
from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models.user import RevokedToken, User

_hasher = PasswordHasher()  # Argon2id with the library's recommended parameters
# Verified against when the username does not exist, so the response time
# does not reveal which usernames are valid.
_DUMMY_HASH = _hasher.hash(secrets.token_urlsafe(16))

TOKEN_ISSUER = "q-genai"
PASSWORD_MIN_LENGTH = 10


class AuthError(Exception):
    """Generic authentication failure - message is safe to show to clients."""


def validate_password_strength(password: str) -> None:
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError(f"Password must be at least {PASSWORD_MIN_LENGTH} characters long.")
    if len(password) > 128:
        raise ValueError("Password must be at most 128 characters long.")
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        raise ValueError("Password must contain both letters and digits.")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def authenticate(db: Session, username: str, password: str) -> User | None:
    user = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if user is None:
        verify_password(password, _DUMMY_HASH)  # constant-ish timing
        return None
    if not verify_password(password, user.password_hash):
        return None
    if not user.is_active:
        return None
    if _hasher.check_needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
    return user


@dataclass
class TokenClaims:
    user_id: str
    jti: str
    token_version: int
    expires_at: datetime


def create_access_token(user: User, settings: Settings) -> tuple[str, datetime]:
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {
        "sub": user.id,
        "iss": TOKEN_ISSUER,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int(expires.timestamp()),
        "jti": secrets.token_urlsafe(16),
        "ver": user.token_version,
        # NOTE: the role is intentionally NOT trusted from the token. The
        # current role is always re-read from the database on every request.
    }
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return token, expires


def decode_access_token(token: str, settings: Settings) -> TokenClaims:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],  # never accept "none" or other algs
            issuer=TOKEN_ISSUER,
            options={"require": ["sub", "exp", "iat", "jti", "iss"]},
        )
    except jwt.ExpiredSignatureError:
        raise AuthError("Session expired. Please log in again.") from None
    except jwt.PyJWTError:
        raise AuthError("Invalid authentication token.") from None
    return TokenClaims(
        user_id=str(payload["sub"]),
        jti=str(payload["jti"]),
        token_version=int(payload.get("ver", 0)),
        expires_at=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
    )


def is_revoked(db: Session, jti: str) -> bool:
    return db.get(RevokedToken, jti) is not None


def revoke_token(db: Session, claims: TokenClaims) -> None:
    if db.get(RevokedToken, claims.jti) is None:
        db.add(RevokedToken(jti=claims.jti, expires_at=claims.expires_at))


def purge_expired_revocations(db: Session) -> None:
    db.query(RevokedToken).filter(RevokedToken.expires_at < datetime.now(timezone.utc)).delete()
    db.commit()
