"""The SYNTHETIC demo operations seed through the real configured pipeline (trained YOLO demo model + rules + Part 3
+ mock RCA + RAG) and the dashboard numbers it produces. Skipped when the demo model / dataset is absent."""
import pytest
from fastapi.testclient import TestClient

from backend.core.config import get_settings
from scripts import seed_demo_operations
from tests.cv.conftest import DEMO_ARTIFACT, DEMO_DATASET

pytestmark = pytest.mark.skipif(not DEMO_ARTIFACT.is_file() or not DEMO_DATASET.is_file(),
                                reason="demo model / synthetic dataset not available")


def test_seed_is_deterministic_and_feeds_the_dashboard(tmp_path, monkeypatch, demo_index, capsys):
    pytest.importorskip("ultralytics")
    env = {"DATABASE_URL": f"sqlite:///{tmp_path / 'demo.db'}", "STORAGE_PATH": str(tmp_path / "img"),
           "ENVIRONMENT": "development", "CV_ENGINE": "yolov8", "CV_MODEL_PATH": str(DEMO_ARTIFACT),
           "CV_DEVICE": "cpu", "QUALITY_GATE_ENGINE": "basic", "DECISION_ENGINE": "rules",
           "DECISION_AUTO_PASS": "true", "PIPELINE_ANALYSIS_ENGINE": "deterministic", "RCA_ENGINE": "genai",
           "ACTION_ENGINE": "rca", "LLM_ENABLED": "true", "LLM_PROVIDER": "mock", "RAG_ENABLED": "true",
           "RAG_INDEX_PATH": str(demo_index), "LOG_LEVEL": "WARNING"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    try:
        from alembic import command
        from alembic.config import Config
        from tests.conftest import ROOT
        cfg = Config(str(ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(ROOT / "backend/db/migrations"))
        cfg.set_main_option("sqlalchemy.url", env["DATABASE_URL"])
        cfg.attributes["configure_logger"] = False
        command.upgrade(cfg, "head")
        assert seed_demo_operations.main(["--dataset", str(DEMO_DATASET.parent)]) == 0
        assert "seeded 102 SYNTHETIC demo inspections" in capsys.readouterr().out
        assert seed_demo_operations.main(["--dataset", str(DEMO_DATASET.parent)]) == 0  # second run: no-op
        assert "already present" in capsys.readouterr().out
        from backend.main import create_app
        with TestClient(create_app()) as c:
            o = c.get("/api/analytics/overview").json()
    finally:
        get_settings.cache_clear()
    assert o["demo_seed_inspections"] == o["inspections"]["total"] == 102
    assert o["period"]["end"] == "2026-10-18" and o["period"]["defaulted"] is True
    assert o["defects"][0]["defect_type"] == "LEAKING_PACKAGE"
    assert o["hotspots"][0]["stage"] == "PACKING" and o["investigation"]["stage"] == "PACKING"
    assert o["feedback"]["feedback_records"] == 22 and o["feedback"]["decision_overrides"] == 2
    assert o["feedback"]["stage_corrections_by_ai_stage"] == {"PACKING": 3}
