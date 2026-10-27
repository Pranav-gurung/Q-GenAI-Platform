import os
from dataclasses import replace

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from backend.core.config import Settings
from backend.db.session import build_engine
from backend.domain.enums import DecisionOutcome
from backend.pipeline.orchestrator import PipelineOrchestrator
from backend.pipeline.registry import build_modules
from backend.pipeline.states import PipelineState as S
from tests.conftest import ROOT
from tests.unit.test_orchestrator import FixedDecision, make_ctx, transitions

MODULES = build_modules(Settings(_env_file=None, pipeline_analysis_engine="deterministic"))


@pytest.mark.parametrize("outcome,final,code", [
    (DecisionOutcome.REVIEW, S.REQUIRES_REVIEW, "DECISION_REVIEW"),
    (DecisionOutcome.FAIL, S.REQUIRES_REVIEW, "NO_CORRECTIVE_ACTION"),
])
def test_review_and_fail_decisions_get_stage_analysis(outcome, final, code):
    ctx = PipelineOrchestrator(replace(MODULES, decision_engine=FixedDecision(outcome))).run(make_ctx())
    assert S.PIPELINE_ANALYSIS in transitions(ctx) and ctx.state is final and ctx.history[-1].error_code == code
    assert ctx.pipeline_analysis.is_real and ctx.pipeline_analysis.trace_analysis is not None
    if outcome is DecisionOutcome.REVIEW:
        assert ctx.rca is None and ctx.action is None  # REVIEW does not go on to RCA / action


def test_pass_skips_stage_analysis():
    ctx = PipelineOrchestrator(replace(MODULES, decision_engine=FixedDecision(DecisionOutcome.PASS))).run(make_ctx())
    assert ctx.state is S.COMPLETED and ctx.pipeline_analysis is None


@pytest.mark.parametrize("backend", ["sqlite"] + (["postgres"] if os.environ.get("TEST_DATABASE_URL") else []))
def test_migration_0002_adds_item_id(backend, tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}" if backend == "sqlite" else os.environ["TEST_DATABASE_URL"]
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "backend/db/migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "0001")
    engine = build_engine(url)
    assert "item_id" not in {c["name"] for c in inspect(engine).get_columns("inspections")}
    command.upgrade(cfg, "head")
    insp = inspect(engine)
    assert "item_id" in {c["name"] for c in insp.get_columns("inspections")}
    assert "ix_inspections_item_id" in {i["name"] for i in insp.get_indexes("inspections")}
    command.downgrade(cfg, "base")
    engine.dispose()
