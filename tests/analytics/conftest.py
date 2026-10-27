"""Part 7 fixtures: a small, fully scripted dataset run through the REAL pipeline (test-double CV, rules decision,
Part 3, mock-LLM RCA, Part 5 SOP grounding) plus Part 6 operator reviews, with inspection timestamps moved to fixed
days so trends are deterministic. Expected numbers in the tests are derived by hand from this script."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import update

from backend.db.models import Inspection, InspectionImage
from tests.feedback.conftest import LEAK_COLOR, ai_client, demo_index, png  # noqa: F401
from tests.rag.helpers import KNOWLEDGE  # noqa: F401

DAY1, DAY2, DAY3 = datetime(2026, 10, 20, 8, tzinfo=timezone.utc), datetime(2026, 10, 21, 8, tzinfo=timezone.utc), \
    datetime(2026, 10, 22, 8, tzinfo=timezone.utc)
_shade = iter(range(30, 200))


def run(c, store, sku, item, stages, leak_at, when):
    """One item through `stages`; CV sees a leak from `leak_at` on. Timestamps: `when` + 1 h per stage."""
    ids = []
    for i, stage in enumerate(stages):
        color = LEAK_COLOR if leak_at and stages.index(leak_at) <= i else (next(_shade), 160, 60)
        iid = c.post("/api/inspections", json={"store_id": store, "sku_id": sku, "stage": stage,
                                               "item_id": item}).json()["inspection_id"]
        r = c.post(f"/api/inspections/{iid}/image", files={"file": ("p.png", png(color), "image/png")})
        assert r.status_code == 200, r.text
        ids.append(iid)
    with c.app.state.session_factory() as db:
        for i, iid in enumerate(ids):
            ts = when + timedelta(hours=i)
            db.execute(update(Inspection).where(Inspection.inspection_id == iid).values(created_at=ts))
            db.execute(update(InspectionImage).where(InspectionImage.inspection_id == iid).values(created_at=ts))
        db.commit()
    return ids


def review(c, iid, **body):
    r = c.post(f"/api/inspections/{iid}/feedback", json={"operator_id": "OP-1", **body})
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture
def dataset(ai_client):  # noqa: F811
    """11 inspections, 5 items, 2 stores, 2 SKUs, 3 days:
    A BLR/MILK day1 R S P K, leak at PACKING  -> reviewed all correct
    B BLR/MILK day1 R S P,   leak at PICKING  -> stage corrected PICKING -> STORAGE, action not useful
    C DEL/EGGS day2 R,       leak at RECEIVING (pre-existing)
    D DEL/EGGS day2 R K,     leak at PACKING  -> operator corrects defect to TORN_PACKAGE, RCA not useful
    E BLR/MILK day3 R,       clean
    """
    c = ai_client
    a = run(c, "DS-BLR-001", "SKU-MILK-500ML", "UNIT-A", ["RECEIVING", "STORAGE", "PICKING", "PACKING"], "PACKING", DAY1)
    b = run(c, "DS-BLR-001", "SKU-MILK-500ML", "UNIT-B", ["RECEIVING", "STORAGE", "PICKING"], "PICKING",
            DAY1 + timedelta(hours=5))
    cc = run(c, "DS-DEL-001", "SKU-EGGS-6", "UNIT-C", ["RECEIVING"], "RECEIVING", DAY2)
    d = run(c, "DS-DEL-001", "SKU-EGGS-6", "UNIT-D", ["RECEIVING", "PACKING"], "PACKING", DAY2 + timedelta(hours=3))
    e = run(c, "DS-BLR-001", "SKU-MILK-500ML", "UNIT-E", ["RECEIVING"], None, DAY3)
    review(c, a[-1], decision_correct=True, defect_correct=True, stage_correct=True, rca_useful=True,
           action_useful=True, comment="Recommendation matches the packing SOP.")
    review(c, b[-1], decision_correct=True, stage_correct=False, suspected_stage="STORAGE", rca_useful=True,
           action_useful=False)
    review(c, d[-1], decision_correct=True, defect_correct=False, defect_types=["TORN_PACKAGE"], rca_useful=False)
    return {"A": a, "B": b, "C": cc, "D": d, "E": e}
