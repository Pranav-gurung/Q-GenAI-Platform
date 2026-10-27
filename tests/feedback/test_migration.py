"""Migration 0003 (structured feedback review): upgrade keeps existing feedback, downgrade removes only its columns.
SQLite always; PostgreSQL too when TEST_DATABASE_URL is set (same convention as tests/integration)."""
import os

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from backend.db.session import build_engine
from tests.conftest import ROOT

NEW = {"decision_correct", "defect_correct", "stage_correct", "rca_useful", "action_useful", "ai_defect_types",
       "ai_stage", "ai_snapshot"}
BACKENDS = ["sqlite"] + (["postgres"] if os.environ.get("TEST_DATABASE_URL") else [])
NOW = "2026-10-27 00:00:00+00:00"


@pytest.fixture(params=BACKENDS)
def alembic(request, tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}" if request.param == "sqlite" else os.environ["TEST_DATABASE_URL"]
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "backend/db/migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.attributes["configure_logger"] = False
    engine = build_engine(url)
    yield cfg, engine
    command.downgrade(cfg, "base")
    if request.param == "postgres":
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
    engine.dispose()


def columns(engine):
    return {c["name"] for c in inspect(engine).get_columns("feedback")}


def test_upgrade_preserves_existing_feedback_and_downgrade_reverts(alembic):
    cfg, engine = alembic
    command.upgrade(cfg, "0002")
    assert not NEW & columns(engine)
    with engine.begin() as c:
        c.execute(text(f"INSERT INTO stores (store_id, name, created_at) VALUES ('S-1', 'Store', '{NOW}')"))
        c.execute(text(f"INSERT INTO skus (sku_id, name, created_at) VALUES ('K-1', 'Milk', '{NOW}')"))
        c.execute(text("INSERT INTO inspections (inspection_id, store_id, sku_id, stage, status, source, metadata, "
                       f"created_at, updated_at) VALUES ('INS-1', 'S-1', 'K-1', 'PACKING', 'COMPLETED', 'api', '{{}}', "
                       f"'{NOW}', '{NOW}')"))
        c.execute(text("INSERT INTO feedback (feedback_id, inspection_id, operator_id, outcome, defect_types, "
                       "corrected_detections, is_correction, model_versions, learning_status, created_at) VALUES "
                       f"('FB-1', 'INS-1', 'OP-1', 'FAIL', '[]', '[]', false, '{{}}', 'PENDING', '{NOW}')"))

    command.upgrade(cfg, "head")
    assert NEW <= columns(engine)
    assert "ix_feedback_ai_stage" in {i["name"] for i in inspect(engine).get_indexes("feedback")}
    with engine.connect() as c:
        row = c.execute(text("SELECT outcome, decision_correct, stage_correct, ai_stage, ai_snapshot FROM feedback")).one()
    assert row[0] == "FAIL" and row[1] is None and row[2] is None and row[3] is None
    assert row[4] in ({}, "{}")  # empty snapshot for pre-Part-6 rows

    command.downgrade(cfg, "0002")
    assert not NEW & columns(engine)
    with engine.connect() as c:
        assert c.execute(text("SELECT feedback_id, outcome FROM feedback")).one() == ("FB-1", "FAIL")
    command.upgrade(cfg, "head")  # re-upgrade works
    assert NEW <= columns(engine)
