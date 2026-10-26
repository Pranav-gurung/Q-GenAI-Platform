"""Application errors. Each maps to a stable machine-readable code and an HTTP status.

API responses for these errors use the envelope
    {"success": false, "error": {"code": ..., "message": ..., "details": ...}}
(see backend/api/errors.py). Internal details/stack traces are never returned to clients.
"""
from __future__ import annotations

from typing import Any


class AppError(Exception):
    code: str = "APP_ERROR"
    http_status: int = 400

    def __init__(self, message: str, *, code: str | None = None, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        self.details = details


class NotFoundError(AppError):
    code = "NOT_FOUND"
    http_status = 404


class ValidationFailedError(AppError):
    """Domain-level validation that cannot be expressed in the request schema (e.g. unknown store)."""
    code = "VALIDATION_ERROR"
    http_status = 422


class ConflictError(AppError):
    code = "CONFLICT"
    http_status = 409


class InvalidStateTransitionError(ConflictError):
    code = "INVALID_STATE_TRANSITION"


class InvalidImageError(AppError):
    """Codes: UNSUPPORTED_IMAGE_TYPE, IMAGE_TOO_LARGE, CORRUPT_IMAGE, EMPTY_FILE, INVALID_IMAGE."""
    code = "INVALID_IMAGE"
    http_status = 422


class AuthenticationError(AppError):
    code = "UNAUTHORIZED"
    http_status = 401


class StorageError(AppError):
    code = "STORAGE_ERROR"
    http_status = 500
