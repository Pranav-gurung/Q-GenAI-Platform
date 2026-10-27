from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from backend import __version__
from backend.db.session import check_connection

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str  # ok | degraded
    version: str
    environment: str
    database: str  # ok | unavailable
    storage: str   # ok | unavailable
    llm: dict = Field(description="GenAI RCA configuration: enabled, provider, model, status "
                                  "(disabled | configured | not_configured). Static check: the provider is not called, "
                                  "and no credentials are ever included.")
    rag: dict = Field(description="RAG / SOP intelligence: enabled, embedding provider/model, vector store, documents, "
                                  "chunks, status (disabled | ready | unavailable) and error code. No document text or "
                                  "credentials.")
    pipeline_modules: dict[str, dict] = Field(description=(
        "Per slot: name, implemented, and for artifact-backed modules (CV) also available, model_loaded, "
        "model_version, device, error. A configured module that cannot run makes status 'degraded'."))


@router.get("/health", response_model=HealthResponse, summary="Service health and module status")
def health(request: Request) -> HealthResponse:
    state = request.app.state
    db_ok = check_connection(state.engine)
    storage_ok = state.storage.healthy()
    modules = state.orchestrator.modules.describe()
    modules_ok = all(m.get("available", True) for m in modules.values())
    rag = state.rca_service.rag
    modules_ok = modules_ok and (not rag.settings.rag_enabled or rag.index is not None)
    return HealthResponse(
        status="ok" if db_ok and storage_ok and modules_ok else "degraded",
        version=__version__, environment=state.settings.environment.value,
        database="ok" if db_ok else "unavailable", storage="ok" if storage_ok else "unavailable",
        llm=state.rca_service.health(),
        rag=rag.health(),
        pipeline_modules=modules,
    )
