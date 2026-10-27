"""HTTP hardening for the existing app: request-size limit, security headers, request-id hygiene, and a simple
in-process rate limit for the expensive endpoints. Deliberately small: no gateway, no Redis, no distributed state."""
from __future__ import annotations

import json
import re
import threading
import time
from collections import defaultdict

from fastapi import Depends, Request

from backend.api.dependencies import get_settings_dep
from backend.core.config import Settings
from backend.core.exceptions import AppError

_REQUEST_ID = re.compile(r"[^A-Za-z0-9._-]")
BODY_OVERHEAD_BYTES = 1024 * 1024  # multipart framing + form fields on top of MAX_UPLOAD_MB

SECURITY_HEADERS = {
    b"x-content-type-options": b"nosniff",
    b"x-frame-options": b"DENY",
    b"referrer-policy": b"no-referrer",
    b"permissions-policy": b"camera=(), microphone=(), geolocation=()",
}
# The bundled UI loads only its own files; inline style attributes are used for chart widths. /api/docs (Swagger UI)
# loads its assets from a CDN, so it is excluded from the CSP.
CSP = (b"default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; "
       b"connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


def clean_request_id(value: str | None) -> str:
    """Client-supplied X-Request-ID is echoed and logged: keep only safe characters (no log / header injection)."""
    return _REQUEST_ID.sub("", value or "")[:64]


class RequestTooLarge(AppError):
    code = "PAYLOAD_TOO_LARGE"
    http_status = 413


class RateLimited(AppError):
    code = "RATE_LIMITED"
    http_status = 429


class HardeningMiddleware:
    """ASGI middleware: (1) rejects bodies above the limit BEFORE the multipart parser spools them to disk, using
    Content-Length when present and counting streamed bytes otherwise; (2) adds security headers to every response."""

    def __init__(self, app, *, max_body_bytes: int, max_upload_bytes: int, docs_prefix: str) -> None:
        self.app, self.max_body, self.max_upload, self.docs_prefix = app, max_body_bytes, max_upload_bytes, docs_prefix

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        length = dict(scope.get("headers") or []).get(b"content-length")
        if length is not None and (not length.isdigit() or int(length) > self.max_body):
            return await self._reject(scope, send)
        received = 0
        csp = not scope.get("path", "").startswith(self.docs_prefix)

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body:
                    raise _BodyTooLarge
            return message

        async def secured_send(message):
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() not in SECURITY_HEADERS]
                headers += list(SECURITY_HEADERS.items())
                if csp:
                    headers.append((b"content-security-policy", CSP))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, limited_receive, secured_send)
        except BaseException as exc:  # noqa: B036 - only our own signal is handled, everything else re-raised
            if not _is_body_too_large(exc):
                raise
            await self._reject(scope, send)

    async def _reject(self, scope, send):
        # Image uploads keep the API's existing contract (422 IMAGE_TOO_LARGE, as the route itself returns);
        # any other oversized body is 413 PAYLOAD_TOO_LARGE.
        if scope.get("method") == "POST" and scope.get("path", "").endswith("/image"):
            status, code = 422, "IMAGE_TOO_LARGE"
            message = f"Image exceeds maximum size of {self.max_upload / (1024 * 1024):g} MB"
        else:
            status, code, message = 413, "PAYLOAD_TOO_LARGE", "Request body exceeds the maximum allowed size"
        body = json.dumps({"success": False, "error": {"code": code, "message": message,
                                                       "details": {"max_bytes": self.max_upload}, "request_id": None}})
        body = body.encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                                (b"connection", b"close"), *SECURITY_HEADERS.items()]})
        await send({"type": "http.response.body", "body": body})


class _BodyTooLarge(BaseException):
    """BaseException on purpose: the multipart parser's caller converts any Exception into a generic 400."""


def _is_body_too_large(exc: BaseException) -> bool:
    if isinstance(exc, _BodyTooLarge):
        return True
    return isinstance(exc, BaseExceptionGroup) and any(_is_body_too_large(e) for e in exc.exceptions)


class RateLimiter:
    """Fixed one-minute window per client key. Thread-safe, in-process, bounded memory (old windows dropped)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._window = 0
        self._counts: dict[str, int] = defaultdict(int)

    def hit(self, key: str, limit: int, now: float | None = None) -> int | None:
        """Count one request; return seconds to wait if the limit is exceeded, else None."""
        now = time.time() if now is None else now
        window = int(now // 60)
        with self._lock:
            if window != self._window:
                self._window, self._counts = window, defaultdict(int)
            self._counts[key] += 1
            if self._counts[key] > limit:
                return 60 - int(now % 60)
        return None


def expensive_endpoint(request: Request, settings: Settings = Depends(get_settings_dep)) -> None:
    """Dependency for endpoints that run CV or call the LLM."""
    limit = settings.rate_limit_per_minute
    if not limit:
        return
    client = request.client.host if request.client else "unknown"
    wait = request.app.state.rate_limiter.hit(client, limit)
    if wait is not None:
        raise RateLimited(f"Too many requests to this endpoint; limit is {limit} per minute",
                          details={"retry_after_seconds": wait})
