"""Database foundation: migrations + persistence through the service layer.

Runs against SQLite by default. Set TEST_DATABASE_URL=postgresql+psycopg://... to also run
against a real (disposable!) PostgreSQL database; the schema is dropped afterwards.
"""
import os

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from backend.db.session import build_engine, build_session_factory
from backend.pipeline.orchestrator import PipelineOrchestrator
from backend.pipeline.registry import build_modules
from backend.pipeline.states import PipelineState
from backend.schemas.feedback import FeedbackCreate
from backend.schemas.inspection import InspectionCreate
from backend.services.feedback_service import FeedbackService
from backend.services.inspection_service import InspectionFilters, InspectionService
from backend.services.reference_service import seed_sample_data
from backend.services.storage import LocalImageStorage
from tests.conftest import ROOT
from tests.utils import image_bytes

EXPECTED_TABLES = {"stores", "skus", "inspections", "inspection_images", "pipeline_events",
                   "inspection_results", "feedback", "alembic_version"}

BACKENDS = ["sqlite"] + (["postgres"] if os.environ.get("TEST_DATABASE_URL") else [])


@pytest.fixture(params=BACKENDS)
def db_url(request, tmp_path):
    url = f"sqlite:///{tmp_path / 'mig.db'}" if request.param == "sqlite" else os.environ["TEST_DATABASE_URL"]
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "backend/db/migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "head")
    yield url
    command.downgrade(cfg, "base")
    if request.param == "postgres":
        engine = build_engine(url)
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
        engine.dispose()


def test_migrations_create_all_tables(db_url):
    engine = build_engine(db_url)
    assert EXPECTED_TABLES <= set(inspect(engine).get_table_names())
    engine.dispose()


def test_inspection_lifecycle_persists(db_url, tmp_path, settings):
    engine = build_engine(db_url)
    factory = build_session_factory(engine)
    storage = LocalImageStorage(tmp_path / "img")
    orchestrator = PipelineOrchestrator(build_modules(settings))

    with factory() as s:
        seed_sample_data(s, settings.sample_data_path)
        svc = InspectionService(s, storage, orchestrator, settings)
        insp = svc.create(InspectionCreate(store_id="DS-MUM-001", sku_id="SKU-EGGS-6", stage="DISPATCH",
                                           order_id="ORD-9", metadata={"rider": "R-1"}))
        svc.attach_image_and_run(insp.inspection_id, image_bytes("JPEG"), "eggs.jpg", "image/jpeg")
        FeedbackService(s, svc).submit(insp.inspection_id, FeedbackCreate(operator_id="OP-2", outcome="FAIL",
                                                                          defect_types=["CRUSHED_PACKAGE"]))
        iid = insp.inspection_id

    # Fresh session: answer the questions the schema must support.
    with factory() as s:
        svc = InspectionService(s, storage, orchestrator, settings)
        insp = svc.get(iid)
        assert (insp.store_id, insp.sku_id, insp.stage.value) == ("DS-MUM-001", "SKU-EGGS-6", "DISPATCH")
        assert insp.created_at.tzinfo is not None
        assert insp.current_image.width == 64 and insp.current_image.mime_type == "image/jpeg"
        assert insp.status is PipelineState.COMPLETED and insp.decision.value == "FAIL"
        assert insp.result.cv["status"] == "NOT_IMPLEMENTED" and insp.result.cv_model_version is None
        assert insp.metadata_ == {"rider": "R-1"}
        assert len(insp.events) >= 10
        fbs = FeedbackService(s, svc).list_for(iid)
        assert fbs[0].defect_types == ["CRUSHED_PACKAGE"] and fbs[0].operator_id == "OP-2"
        items, total = svc.list(InspectionFilters(store_id="DS-MUM-001"), 10, 0)
        assert total == 1
    engine.dispose()
