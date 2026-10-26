"""FastAPI surface for Module C.  Run:  uvicorn module_c.api:create_app --factory --port 8001

All business logic lives in module_c.rca.RCAService (testable without FastAPI). This file only
handles HTTP concerns: structural validation (Pydantic), authentication hook, status-code mapping.

HTTP mapping
  200  valid request, status = success | insufficient_evidence  (insufficient evidence is NOT an HTTP error)
  401/403 authentication/authorization hook failed
  422  invalid request (schema, bbox, confidence, unknown class, too many defects, ...) - nothing analysed
  429  LLM rate limited (Retry-After set when known)      body: RCAResponse with status="error"
  502  LLM returned malformed/unvalidated output          body: RCAResponse with status="error"
  503  LLM not configured / unavailable                   body: RCAResponse with status="error"
  504  LLM timeout                                        body: RCAResponse with status="error"
  500  audit failure under RCA_AUDIT_FAIL_MODE=fail_closed (result withheld)
"""
from __future__ import annotations

import hmac
import os
import uuid
from typing import Any, Optional, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError as FastAPIValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .audit import AuditEvent
from .config import Settings
from .rca import RCAService, build_service


class DefectIn(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    defect_class: str = Field(alias="class", description="Defect class name; must match the shared class map (Module A).")
    confidence: float = Field(description="Detector confidence in [0, 1].")
    bbox: Optional[list[float]] = Field(None, description="[x_min, y_min, x_max, y_max] in ORIGINAL-image pixels.")
    mask_ref: Optional[str] = Field(None, description="Reference to a segmentation mask, if the detector produces one.")


class RCARequestIn(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "inspection_id": "QC-0001", "image_id": "IMG-0001", "model_version": "v1", "status": "review_required",
        "defects": [{"class": "scratch", "confidence": 0.94, "bbox": [120, 80, 260, 190]}]}})
    inspection_id: str
    image_id: str
    model_version: str
    status: Optional[str] = None
    defects: list[DefectIn]
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    part_type: Optional[str] = None
    material: Optional[str] = None
    production_process: Optional[str] = None
    inspection_context: Optional[str] = None


class RCAResponseOut(BaseModel):
    """Documented response shape (see schemas/rca_response.schema.json). Extra keys are allowed."""
    model_config = ConfigDict(extra="allow")
    request_id: str
    inspection_id: Optional[str] = None
    image_id: Optional[str] = None
    model_version: Optional[str] = None
    status: str = Field(description="success | insufficient_evidence | error")
    summary: str = ""
    observations: list[str] = []
    possible_causes: list[dict[str, Any]] = []
    recommended_actions: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    defect_analyses: list[dict[str, Any]] = []
    unknowns: list[str] = []
    notes: list[str] = []
    limitations: str = ""
    error: Optional[dict[str, str]] = None
    generated_at: Optional[str] = None


class Authenticator(Protocol):
    """Module D plugs in here. Return an actor id (str) or raise HTTPException(401/403)."""
    def __call__(self, request: Request) -> Optional[str]: ...


class NoAuth:
    def __call__(self, request: Request) -> Optional[str]:
        return None


class StaticApiKeyAuth:
    """Stop-gap shared-secret check (header X-API-Key vs env RCA_API_KEY). NOT a replacement for Module D."""

    def __call__(self, request: Request) -> Optional[str]:
        expected = os.environ.get("RCA_API_KEY")
        if not expected:
            raise HTTPException(503, "RCA_AUTH_MODE=static_api_key but RCA_API_KEY is not set")
        given = request.headers.get("x-api-key", "")
        if not hmac.compare_digest(given.encode(), expected.encode()):
            raise HTTPException(401, "invalid or missing API key")
        return "static-api-key-client"


def create_app(settings: Optional[Settings] = None, service: Optional[RCAService] = None,
               authenticator: Optional[Authenticator] = None) -> FastAPI:
    s = settings or Settings.from_env()
    svc = service or build_service(s)
    auth = authenticator or (StaticApiKeyAuth() if s.auth_mode == "static_api_key" else NoAuth())
    app = FastAPI(title="Module C - GenAI Root-Cause Analysis", version="0.1.0",
                  description="Document-grounded RCA suggestions for detector output. Not proof of root cause.")

    @app.exception_handler(FastAPIValidationError)
    async def _structural_422(request: Request, exc: FastAPIValidationError):
        req_id = uuid.uuid4().hex
        if request.url.path == "/rca":
            try:
                svc.audit.record(AuditEvent("rca.rejected", req_id, http_status=422, error_code="invalid_request"))
            except Exception:  # audit visibility: logged by the sink owner; request is rejected anyway
                pass
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()], "request_id": req_id})

    @app.get("/health")
    def health() -> dict:
        provider = svc.provider
        return {"status": "ok", "indexed_chunks": svc.retriever.store.count(),
                "embedding_backend": svc.retriever.embedder.name,
                "llm_provider": provider.name, "llm_model": getattr(provider, "model", ""),
                "llm_configured": bool(getattr(provider, "configured", True)),
                "supported_classes": sorted(s.supported_classes), "audit_sink": s.audit_sink,
                "audit_fail_mode": s.audit_fail_mode, "auth_mode": s.auth_mode}

    def _auth_dep(request: Request) -> Optional[str]:
        return auth(request)

    @app.post("/rca", response_model=RCAResponseOut, responses={422: {}, 429: {}, 500: {}, 502: {}, 503: {}, 504: {}})
    def rca(body: RCARequestIn, request: Request, actor: Optional[str] = Depends(_auth_dep)):
        outcome = svc.handle_raw(body.model_dump(by_alias=True, exclude_none=True), actor=actor)
        headers = {"Retry-After": str(int(outcome.retry_after))} if outcome.retry_after else None
        return JSONResponse(status_code=outcome.http_status, content=outcome.response, headers=headers)

    return app
