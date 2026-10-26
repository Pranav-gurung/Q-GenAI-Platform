"""Authentication endpoints: login, logout, current user."""
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.dependencies import AuthContext, get_auth_context
from app.schemas.auth import MeResponse, MessageResponse, TokenResponse
from app.services.audit_service import append_event, audited_transaction, record_standalone
from app.services.authentication import authenticate, create_access_token, revoke_token
from app.services.authorization import permissions_for
from app.services.rate_limiter import LoginRateLimiter

router = APIRouter(prefix="/auth", tags=["Authentication"])


@lru_cache
def get_login_limiter() -> LoginRateLimiter:
    s = get_settings()
    return LoginRateLimiter(s.login_max_attempts, s.login_window_seconds)


def _client_ip(request: Request) -> str:
    # Behind a reverse proxy, configure uvicorn --proxy-headers so this is the real client IP.
    return request.client.host if request.client else "unknown"


@router.post("/login", response_model=TokenResponse,
             summary="Log in (public) - returns a short-lived JWT bearer token")
def login(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    limiter: LoginRateLimiter = Depends(get_login_limiter),
):
    username = form.username.strip()[:64]
    ip = _client_ip(request)
    key = f"{ip}|{username.lower()}"

    retry_after = limiter.retry_after(key)
    if retry_after:
        record_standalone(db, "auth.login_blocked",
                          metadata={"username_attempted": username, "client_ip": ip})
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            "Too many failed login attempts. Try again later.",
                            headers={"Retry-After": str(retry_after)})

    user = authenticate(db, username, form.password)
    if user is None:
        db.rollback()
        limiter.record_failure(key)
        # Credentials are never logged - only the attempted username and IP.
        record_standalone(db, "auth.login_failed",
                          metadata={"username_attempted": username, "client_ip": ip})
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect username or password.",
                            headers={"WWW-Authenticate": "Bearer"})

    token, expires = create_access_token(user, settings)
    with audited_transaction(db):  # login only succeeds if it is audit-logged
        append_event(db, "auth.login_success", actor=user, target_type="user", target_id=user.id,
                     metadata={"client_ip": ip})
    limiter.reset(key)
    return TokenResponse(access_token=token, expires_at=expires, role=user.role.value,
                         username=user.username)


@router.post("/logout", response_model=MessageResponse, summary="Log out (revokes the current token)")
def logout(ctx: AuthContext = Depends(get_auth_context), db: Session = Depends(get_db)):
    with audited_transaction(db):
        revoke_token(db, ctx.claims)
        append_event(db, "auth.logout", actor=ctx.user, target_type="user", target_id=ctx.user.id)
    return MessageResponse(detail="Logged out.")


@router.get("/me", response_model=MeResponse, summary="Current user, role and permissions")
def me(ctx: AuthContext = Depends(get_auth_context)):
    u = ctx.user
    return MeResponse(id=u.id, username=u.username, full_name=u.full_name, role=u.role.value,
                      site=u.site, permissions=permissions_for(u),
                      token_expires_at=ctx.claims.expires_at)
