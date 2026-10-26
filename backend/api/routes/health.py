from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from backend import __version__
from backend.db.session import check_connection

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str  # ok | degraded
    version: str
    environment: str
    database: str  # ok | unavailable
    storage: str   # ok | unavailable
    pipeline_modules: dict[str, dict]


@router.get("/health", response_model=HealthResponse, summary="Service health and module status")
def health(request: Request) -> HealthResponse:
    state = request.app.state
    db_ok = check_connection(state.engine)
    storage_ok = state.storage.healthy()
    return HealthResponse(
        status="ok" if db_ok and storage_ok else "degraded",
        version=__version__, environment=state.settings.environment.value,
        database="ok" if db_ok else "unavailable", storage="ok" if storage_ok else "unavailable",
        pipeline_modules=state.orchestrator.modules.describe(),
    )
