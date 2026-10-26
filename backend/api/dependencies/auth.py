"""Minimal auth hook. If API_KEY is configured, write endpoints require header `X-API-Key`.

This is a placeholder structure, not a user/role system; replace this dependency with real
operator authentication (Part 7) without touching the routes.
"""
from __future__ import annotations

import hmac

from fastapi import Depends, Header

from backend.api.dependencies.services import get_settings_dep
from backend.core.config import Settings
from backend.core.exceptions import AuthenticationError


def require_api_key(x_api_key: str | None = Header(default=None),
                    settings: Settings = Depends(get_settings_dep)) -> None:
    if settings.api_key is None:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), settings.api_key.encode()):
        raise AuthenticationError("Missing or invalid API key")
