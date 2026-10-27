"""Part 6 fixtures: the real app + a deterministic test-double CV, so inspections carry a real AI result
(rules decision, Part 3 attribution, mock-LLM RCA, Part 5 SOP grounding) that operators can review."""
import io
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.db.base import Base
from backend.domain.enums import DefectType, ModuleStatus
from backend.main import create_app
from backend.schemas.results import CVResult, Detection
from backend.services.reference_service import seed_sample_data
from tests.rag.conftest import demo_index  # noqa: F401  (session-scoped SOP index)
from tests.rag.helpers import KNOWLEDGE

BASE = {"store_id": "DS-BLR-001", "sku_id": "SKU-MILK-500ML"}
LEAK_COLOR, CLEAN_COLOR = (220, 40, 40), (90, 160, 60)


class ColorCV:
    """Test double: a red-dominant image 'shows' a HIGH-severity leak, anything else shows nothing."""
    name, implemented = "test-color-cv", True

    def health(self):
        return {}

    def detect(self, image):
        r, g, _ = Image.open(io.BytesIO(image.load_bytes())).convert("RGB").getpixel((1, 1))
        dets = [Detection(defect_type=DefectType.LEAKING_PACKAGE, confidence=0.94, severity="HIGH")] if r > 200 else []
        return CVResult(module=self.name, module_version="test-cv-1", status=ModuleStatus.COMPLETED, detections=dets)


def png(color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def ai_client(settings, demo_index):  # noqa: F811
    s = settings.model_copy(update={
        "decision_engine": "rules", "decision_auto_pass": True, "pipeline_analysis_engine": "deterministic",
        "rca_engine": "genai", "action_engine": "rca", "llm_enabled": True, "llm_provider": "mock",
        "rag_enabled": True, "rag_index_path": demo_index, "rag_knowledge_path": KNOWLEDGE})
    app = create_app(s)
    orch = app.state.orchestrator
    orch.modules = replace(orch.modules, cv_engine=ColorCV())
    Base.metadata.create_all(app.state.engine)
    with app.state.session_factory() as db:
        seed_sample_data(db, s.sample_data_path)
    with TestClient(app) as c:
        yield c


def inspect(c, stage, color, item="UNIT-FB"):
    iid = c.post("/api/inspections", json={**BASE, "stage": stage, "item_id": item}).json()["inspection_id"]
    r = c.post(f"/api/inspections/{iid}/image", files={"file": ("p.png", png(color), "image/png")})
    assert r.status_code == 200, r.text
    return r.json()


def canonical(c, item="UNIT-FB"):
    """RECEIVING / STORAGE / PICKING PASS (CV + rules), PACKING: CV leak -> FAIL -> PACKING, HIGH -> RCA -> SOP."""
    for i, st in enumerate(("RECEIVING", "STORAGE", "PICKING")):  # distinct images: Part 3 ignores duplicates
        assert inspect(c, st, (40 + i, 160, 60), item)["decision"] == "PASS"
    return inspect(c, "PACKING", LEAK_COLOR, item)
