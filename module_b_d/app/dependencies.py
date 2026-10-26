"""Shared FastAPI dependencies: current user + permission guards."""
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.models.user import User
from app.services.authentication import AuthError, TokenClaims, decode_access_token, is_revoked
from app.services.authorization import Permission, has_permission

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

_UNAUTHORIZED = {"WWW-Authenticate": "Bearer"}


@dataclass
class AuthContext:
    user: User
    claims: TokenClaims


def get_auth_context(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AuthContext:
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated.", headers=_UNAUTHORIZED)
    try:
        claims = decode_access_token(token, settings)
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc), headers=_UNAUTHORIZED) from None

    if is_revoked(db, claims.jti):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session has been logged out.",
                            headers=_UNAUTHORIZED)
    user = db.get(User, claims.user_id)
    # Disabled users and tokens issued before a role/password change are rejected.
    if user is None or not user.is_active or user.token_version != claims.token_version:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid authentication token.",
                            headers=_UNAUTHORIZED)
    return AuthContext(user=user, claims=claims)


def get_current_user(ctx: AuthContext = Depends(get_auth_context)) -> User:
    return ctx.user


def require_permission(permission: Permission):
    """Dependency factory: `user = Depends(require_permission(Permission.X))`."""

    def _guard(user: User = Depends(get_current_user)) -> User:
        if not has_permission(user, permission):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "You do not have permission for this action.")
        return user

    return _guard


def require_any_permission(*permissions: Permission):
    def _guard(user: User = Depends(get_current_user)) -> User:
        if not any(has_permission(user, p) for p in permissions):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "You do not have permission for this action.")
        return user

    return _guard
