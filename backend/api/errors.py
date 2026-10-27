"""Exception -> consistent JSON error envelope. Stack traces are logged, never returned."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.core.exceptions import AppError
from backend.core.logging import request_id_var
from backend.schemas.common import ErrorBody, ErrorResponse

logger = logging.getLogger("qgenai.api")

_HTTP_CODES = {401: "UNAUTHORIZED", 403: "FORBIDDEN", 404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED",
               413: "PAYLOAD_TOO_LARGE", 415: "UNSUPPORTED_MEDIA_TYPE"}


def error_response(request: Request, status: int, code: str, message: str, details=None) -> JSONResponse:
    # request.state survives into the outermost (500) handler; the context var may already be reset there.
    rid = getattr(request.state, "request_id", None) or request_id_var.get()
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details, request_id=rid))
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        log = logger.error if exc.http_status >= 500 else logger.info
        log("%s %s -> %d %s: %s", request.method, request.url.path, exc.http_status, exc.code, exc.message,
            exc_info=exc.http_status >= 500)
        return error_response(request, exc.http_status, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Echo location + message only (no raw input values).
        details = [{"field": ".".join(str(p) for p in err["loc"]), "message": err["msg"], "type": err["type"]}
                   for err in exc.errors()]
        logger.info("%s %s -> 422 VALIDATION_ERROR (%d issues)", request.method, request.url.path, len(details))
        return error_response(request, 422, "VALIDATION_ERROR", "Request validation failed", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "HTTP_ERROR")
        return error_response(request, exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s %s", request.method, request.url.path,
                         extra={"request_id": getattr(request.state, "request_id", None)})
        return error_response(request, 500, "INTERNAL_ERROR", "An internal error occurred")


# For OpenAPI docs
ERROR_RESPONSES = {
    401: {"model": ErrorResponse, "description": "Missing/invalid API key (only when API_KEY is configured)"},
    404: {"model": ErrorResponse, "description": "Resource not found"},
    409: {"model": ErrorResponse, "description": "Invalid pipeline state for this operation"},
    422: {"model": ErrorResponse, "description": "Validation error (request, ID, image, stage, ...)"},
    500: {"model": ErrorResponse, "description": "Internal error (details are logged, not returned)"},
}
