"""Application factory.  Run:  uvicorn backend.main:create_app --factory --reload"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend import __version__
from backend.api.errors import register_exception_handlers
from backend.api.routes import health, inspections, reference
from backend.core.config import Settings, get_settings
from backend.core.logging import configure_logging, request_id_var
from backend.db.session import build_engine, build_session_factory
from backend.pipeline.orchestrator import PipelineOrchestrator
from backend.pipeline.registry import build_modules
from backend.pipeline.states import PipelineState
from backend.rca.service import RCAService
from backend.services.reference_service import is_empty, seed_sample_data
from backend.services.storage import build_storage

logger = logging.getLogger("qgenai")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

API_DESCRIPTION = f"""
Computer-vision-first quality inspection pipeline for quick-commerce dark stores.

**Modules** are selected by configuration (see `GET /api/health` for what is active and whether it can run):
image quality gate (`basic`), computer vision (`yolov8`), decision rules (`rules`), fulfillment stage
analysis (`deterministic`), GenAI root-cause analysis (`genai`) and RCA-derived actions (`rca`). CV reports visual
evidence; PASS/FAIL/REVIEW comes from deterministic rules; stage attribution is deterministic; the LLM only adds
labelled hypotheses. Inspections without a real decision end in `REQUIRES_REVIEW`.

**Pipeline states:** {", ".join(s.value for s in PipelineState)}

**Errors** always use: `{{"success": false, "error": {{"code", "message", "details", "request_id"}}}}`
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.seed_sample_data and settings.sample_data_path.exists():
            with app.state.session_factory() as session:
                try:
                    if is_empty(session):
                        seed_sample_data(session, settings.sample_data_path)
                except Exception:  # e.g. migrations not applied yet; health will report it
                    logger.exception("sample-data seeding skipped")
        logger.info("Q-GenAI %s started (env=%s, modules=%s)", __version__, settings.environment.value,
                    {k: v["name"] for k, v in app.state.orchestrator.modules.describe().items()})
        yield
        app.state.engine.dispose()

    app = FastAPI(title="Q-GenAI API", version=__version__, description=API_DESCRIPTION, lifespan=lifespan,
                  docs_url=f"{settings.api_prefix}/docs", openapi_url=f"{settings.api_prefix}/openapi.json",
                  redoc_url=None)

    app.state.settings = settings
    app.state.engine = build_engine(settings.database_url)
    app.state.session_factory = build_session_factory(app.state.engine)
    app.state.storage = build_storage(settings)
    app.state.orchestrator = PipelineOrchestrator(build_modules(settings))
    rca_engine = app.state.orchestrator.modules.rca_engine
    app.state.rca_service = getattr(rca_engine, "service", None) or RCAService(settings)  # shared by GET /rca

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID", "")[:64] or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        if request.url.path.startswith(settings.api_prefix):
            logging.getLogger("qgenai.http").info(
                "%s %s %d %.1fms", request.method, request.url.path, response.status_code,
                (time.perf_counter() - start) * 1000, extra={"request_id": rid})
        return response

    register_exception_handlers(app)
    for module in (health, reference, inspections):
        app.include_router(module.router, prefix=settings.api_prefix)

    if settings.serve_frontend and FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app

