"""Q-GenAI FastAPI application (Sections B + D).

Run:  uvicorn app.main:app --host 127.0.0.1 --port 8000
Docs: http://127.0.0.1:8000/docs
"""
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select

from app.config import get_settings
from app.database import SessionLocal, init_db
from app.dependencies import get_current_user
from app.models.model_version import ModelVersion
from app.models.user import User
from app.routers import admin, auth, inspections, quality
from app.services.audit_service import AuditWriteError
from app.services.authentication import purge_expired_revocations
from app.services.detector import detector_manager
from app.services.policy_service import ensure_default_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("qgenai")

TAGS = [
    {"name": "General", "description": "**Public** - no authentication required (except /model-info)."},
    {"name": "Authentication", "description": "`/auth/login` is **public**; the rest require a bearer token."},
    {"name": "Inspection", "description": "**Authenticated.** Section B - defect detection."},
    {"name": "Quality management", "description": "**Authenticated + role checks.**"},
    {"name": "Administration", "description": "**Administrator only.** Section D."},
    {"name": "Integration (Section C)", "description": "**Authenticated.** Image-free data for RCA."},
]


def load_model_at_startup() -> None:
    settings = get_settings()
    with SessionLocal() as db:
        active = db.execute(select(ModelVersion).where(ModelVersion.is_active.is_(True))).scalar_one_or_none()
    if active:
        weights, version, approved = active.weights_file, active.version_label, active.approved
    else:  # fall back to the .env configuration (not registered -> not approved)
        weights, version, approved = settings.model_weights, settings.model_version, False
    info = detector_manager.load(settings.models_dir, weights, version, settings.inference_device,
                                 settings.inference_image_size, approved)
    log.info("Model status: %s - %s", info.status, info.message)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    settings.validate_for_startup()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    with SessionLocal() as db:
        ensure_default_policy(db)
        purge_expired_revocations(db)
    load_model_at_startup()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=("Q-GenAI - AI-powered industrial quality control. Section B (computer vision) "
                     "and Section D (security & compliance). Click **Authorize** and log in to "
                     "try authenticated endpoints."),
        openapi_tags=TAGS,
        lifespan=lifespan,
        debug=settings.debug,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,  # explicit list, never "*"
        allow_credentials=False,                  # bearer tokens, no cookies
        allow_methods=["GET", "POST", "PUT", "PATCH"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if not request.url.path.startswith(("/docs", "/openapi.json", "/redoc")):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(AuditWriteError)
    async def audit_failure(_request: Request, exc: AuditWriteError):
        log.error("Audit write failure: %s", exc)
        return JSONResponse(status_code=500, content={
            "detail": "The action could not be recorded in the audit log, so it was rolled back "
                      "and NOT saved. Please retry or contact an administrator."})

    @app.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception):
        ref = uuid.uuid4().hex[:12]
        log.exception("Unhandled error (ref %s)", ref)  # details stay in the server log
        return JSONResponse(status_code=500,
                            content={"detail": f"Internal server error (reference {ref})."})

    @app.get("/health", tags=["General"], summary="Liveness check (public)")
    def health():
        current = detector_manager.current
        return {"status": "ok",
                "model_status": current.info.status if current else "unavailable"}

    @app.get("/model-info", tags=["General"], summary="Active model details (authenticated)")
    def model_info(_user: User = Depends(get_current_user)):
        current = detector_manager.current
        if current is None:
            return {"model_status": "unavailable", "message": "No detector initialised."}
        return current.info.public_dict()

    app.include_router(auth.router)
    app.include_router(inspections.router)
    app.include_router(quality.router)
    app.include_router(admin.router)
    return app


app = create_app()
