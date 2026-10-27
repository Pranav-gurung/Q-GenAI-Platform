"""GET /api/analytics/overview on the scripted dataset (see conftest): every expected number is derived by hand."""
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from backend.db.base import Base
from backend.db.models import Feedback, Inspection, InspectionResult
from backend.main import create_app
from backend.services.reference_service import seed_sample_data
from tests.analytics.conftest import review
from tests.utils import image_bytes

PERIOD = "start_date=2026-10-20&end_date=2026-10-22"


def overview(c, query=PERIOD):
    r = c.get(f"/api/analytics/overview?{query}")
    assert r.status_code == 200, r.text
    assert "NaN" not in r.text and "Infinity" not in r.text
    return r.json()


# ------------------------------------------------------------------------------------------------ overview
def test_inspection_counts_and_rates(dataset, ai_client):
    o = overview(ai_client)
    i = o["inspections"]
    assert (i["total"], i["decided"], i["pass"], i["fail"], i["review"]) == (11, 11, 7, 4, 0)
    assert (i["defective"], i["fail_rate"], i["observed_defect_rate"]) == (4, 0.3636, 0.3636)
    assert i["operator_verdicts"] == 3 and i["in_progress"] == 0
    assert o["has_data"] and o["period"] == {"start": "2026-10-20", "end": "2026-10-22", "days": 3,
                                             "granularity": "day", "defaulted": False}
    assert "not be interpreted as causal proof" in o["disclaimer"]
    assert "INSPECTED items" in o["definitions"]["observed_defect_rate"]


def test_defects_by_type_and_severity(dataset, ai_client):
    o = overview(ai_client)
    assert (o["unique_defects"], o["defect_observations"]) == (4, 4)
    leak, torn = o["defects"]  # ranked: top observed quality issues first
    assert (leak["defect_type"], leak["unique_defects"], leak["share_of_unique"]) == ("LEAKING_PACKAGE", 3, 0.75)
    assert leak["severity"] == {"HIGH": 3}  # CV severity kept when an operator confirmed the CV defect
    assert leak["most_common_first_stage"] == "RECEIVING"  # 1/1/1 tie -> earliest in flow order (deterministic)
    assert (torn["defect_type"], torn["unique_defects"], torn["most_common_first_stage"]) == (
        "TORN_PACKAGE", 1, "PACKING")
    assert torn["severity"] == {"UNSPECIFIED": 1}  # operator-only label: no CV severity is invented


def test_stage_metrics_and_hotspots(dataset, ai_client):
    o = overview(ai_client)
    stages = {s["stage"]: s for s in o["stages"]}
    assert set(stages) == {"RECEIVING", "STORAGE", "PICKING", "PACKING"}  # nothing invented for DISPATCH/DELIVERY
    assert (stages["RECEIVING"]["inspections"], stages["RECEIVING"]["defective_inspections"]) == (5, 1)
    assert stages["PACKING"]["observed_defect_rate"] == 1.0 and stages["STORAGE"]["first_observed_defects"] == 0
    hot = [(h["rank"], h["stage"], h["first_observed_defects"]) for h in o["hotspots"]]
    assert hot == [(1, "PACKING", 2), (2, "PICKING", 1), (3, "RECEIVING", 1)]  # PICKING beats RECEIVING on rate
    assert o["hotspots"][0]["evidence_levels"] == {"HIGH": 1, "INSUFFICIENT": 1}
    inv = o["investigation"]
    assert inv["stage"] == "PACKING" and "highest observed quality hotspot" in inv["headline"]
    assert "not a root-cause finding" in inv["caveat"]
    text = json.dumps(inv).lower()
    assert "caused" not in text and "root cause is" not in text


def test_defect_stage_matrix(dataset, ai_client):
    m = overview(ai_client)["matrix"]
    assert m["stages"] == ["RECEIVING", "STORAGE", "PICKING", "PACKING", "DISPATCH", "DELIVERY"]
    rows = {r["defect_type"]: r for r in m["rows"]}
    assert set(rows) == {"LEAKING_PACKAGE", "TORN_PACKAGE"}  # only observed defect types
    assert rows["LEAKING_PACKAGE"]["counts"] == {"RECEIVING": 1, "STORAGE": 0, "PICKING": 1, "PACKING": 1,
                                                 "DISPATCH": None, "DELIVERY": None}  # null = not inspected
    assert rows["TORN_PACKAGE"]["total"] == 1


def test_store_and_sku_metrics(dataset, ai_client):
    o = overview(ai_client)
    stores = {s["key"]: s for s in o["stores"]}
    assert (stores["DS-BLR-001"]["inspections"], stores["DS-BLR-001"]["fail"], stores["DS-BLR-001"]["fail_rate"]) == (
        8, 2, 0.25)
    assert stores["DS-DEL-001"]["observed_defect_rate"] == 0.6667 and stores["DS-DEL-001"]["unique_defects"] == 2
    assert stores["DS-BLR-001"]["name"].startswith("Sample Dark Store")
    assert [s["key"] for s in o["stores"]] == ["DS-DEL-001", "DS-BLR-001"]  # same volume -> higher rate first
    skus = {s["key"]: s for s in o["skus"]}
    assert skus["SKU-EGGS-6"]["category"] == "Dairy & Eggs" and skus["SKU-MILK-500ML"]["inspections"] == 8
    assert {c["key"] for c in o["categories"]} == {"Dairy", "Dairy & Eggs"}


def test_filters(dataset, ai_client):
    o = overview(ai_client, f"{PERIOD}&store_id=DS-DEL-001")
    assert o["inspections"]["total"] == 3 and [s["key"] for s in o["stores"]] == ["DS-DEL-001"]
    assert o["feedback"]["feedback_records"] == 1  # only D's review is in this store
    o = overview(ai_client, f"{PERIOD}&sku_id=SKU-MILK-500ML")
    assert o["inspections"]["total"] == 8 and o["unique_defects"] == 2
    o = overview(ai_client, "start_date=2026-10-21&end_date=2026-10-21")
    assert o["inspections"]["total"] == 3 and len(o["trends"]["buckets"]) == 1
    o = overview(ai_client, f"{PERIOD}&store_id=DS-MUM-001")  # valid id, no data
    assert o["has_data"] is False and o["inspections"]["total"] == 0 and o["investigation"] is None


# ------------------------------------------------------------------------------------------------ trends
def test_daily_trend(dataset, ai_client):
    t = overview(ai_client)["trends"]
    assert t["granularity"] == "day"
    assert [(b["start"], b["inspections"], b["defective_inspections"], b["observed_defect_rate"]) for b in t["buckets"]] \
        == [("2026-10-20", 7, 2, 0.2857), ("2026-10-21", 3, 2, 0.6667), ("2026-10-22", 1, 0, 0.0)]
    d = t["direction"]
    assert (d["direction"], d["first_half_rate"], d["second_half_rate"]) == ("decreased", 0.2857, 0.0)
    assert "observed defect rate decreased" in d["statement"] and "not their cause" in d["statement"]


def test_weekly_trend_and_gap_days(dataset, ai_client):
    t = overview(ai_client, f"{PERIOD}&granularity=week")["trends"]
    assert [(b["start"], b["inspections"]) for b in t["buckets"]] == [("2026-10-19", 11)]  # ISO week (Monday)
    assert t["direction"]["direction"] == "insufficient_data"
    days = overview(ai_client, "start_date=2026-10-18&end_date=2026-10-23")["trends"]["buckets"]
    assert [b["inspections"] for b in days] == [0, 0, 7, 3, 1, 0]
    assert days[0]["observed_defect_rate"] is None  # no inspections: null, not 0


def test_default_period_ends_at_latest_inspection(dataset, ai_client):
    o = overview(ai_client, "")
    assert o["period"]["defaulted"] is True and o["period"]["end"] == "2026-10-22"
    assert o["period"]["start"] == "2026-10-20"  # starts at the first inspection, not 30 empty days earlier
    assert o["inspections"]["total"] == 11


def test_empty_period(dataset, ai_client):
    o = overview(ai_client, "start_date=2025-01-01&end_date=2025-01-02")
    assert o["has_data"] is False and o["message"] == "No inspection data available for this period."
    assert o["inspections"]["fail_rate"] is None and o["inspections"]["observed_defect_rate"] is None
    assert [b["observed_defect_rate"] for b in o["trends"]["buckets"]] == [None, None]
    assert o["feedback"]["decision_override_rate"] is None and o["feedback"]["rca"]["useful_rate"] is None


# ------------------------------------------------------------------------------------------------ feedback
def test_feedback_rates(dataset, ai_client):
    f = overview(ai_client)["feedback"]
    assert (f["feedback_records"], f["decision_reviews"], f["decision_overrides"], f["decision_override_rate"]) == (
        3, 3, 0, 0.0)
    assert (f["defect_reviews"], f["defect_corrections"], f["defect_correction_rate"]) == (2, 1, 0.5)
    assert (f["stage_reviews"], f["stage_corrections"], f["stage_correction_rate"]) == (2, 1, 0.5)
    assert f["stage_corrections_by_ai_stage"] == {"PICKING": 1}
    assert f["rca"] == {"rated": 3, "useful": 2, "not_useful": 1, "useful_rate": 0.6667}
    assert f["actions"] == {"rated": 2, "useful": 1, "not_useful": 1, "useful_rate": 0.5}
    assert f["sop_grounded_actions"]["rated"] == 2  # both rated inspections had GROUNDED SOP actions


def test_decision_override_rate(dataset, ai_client):
    review(ai_client, dataset["C"][0], decision_correct=False, outcome="REVIEW")  # override on item C
    o = overview(ai_client)
    assert (o["feedback"]["decision_overrides"], o["feedback"]["decision_override_rate"]) == (1, 0.25)
    assert o["inspections"]["review"] == 1 and o["inspections"]["fail"] == 3  # operator verdict is the outcome


def test_zero_denominators_without_any_data(client):
    o = client.get("/api/analytics/overview").json()
    assert o["period"] is None and o["has_data"] is False and o["message"] == "No inspection data available."
    f = o["feedback"]
    assert None is f["decision_override_rate"] is f["defect_correction_rate"] is f["stage_correction_rate"]
    assert f["actions"]["useful_rate"] is None and o["inspections"]["observed_defect_rate"] is None


# ------------------------------------------------------------------------------------------------ outcome classes
def test_rejected_undecided_and_in_progress_are_excluded_from_rates(settings, valid_payload):
    app = create_app(settings.model_copy(update={"quality_gate_engine": "basic"}))
    Base.metadata.create_all(app.state.engine)
    with app.state.session_factory() as db:
        seed_sample_data(db, settings.sample_data_path)
    with TestClient(app) as c:
        rejected = c.post("/api/inspections", json=valid_payload).json()["inspection_id"]
        c.post(f"/api/inspections/{rejected}/image", files={"file": ("a.png", image_bytes(), "image/png")})  # 64x48
        c.post("/api/inspections", json=valid_payload)  # never gets an image: in progress
        o = c.get("/api/analytics/overview").json()
    i = o["inspections"]
    assert (i["total"], i["image_rejected"], i["in_progress"], i["decided"]) == (1, 1, 1, 0)
    assert i["fail_rate"] is None and o["has_data"] is False


def test_stub_pipeline_counts_as_no_decision(client, valid_payload):
    iid = client.post("/api/inspections", json=valid_payload).json()["inspection_id"]
    client.post(f"/api/inspections/{iid}/image", files={"file": ("a.png", image_bytes(), "image/png")})
    i = client.get("/api/analytics/overview").json()["inspections"]
    assert (i["total"], i["no_decision"], i["decided"]) == (1, 1, 0)
    client.post(f"/api/inspections/{iid}/feedback", json={"operator_id": "OP-1", "outcome": "FAIL",
                                                          "defect_types": ["DAMAGED_BOX"]})
    i = client.get("/api/analytics/overview").json()["inspections"]
    assert (i["decided"], i["fail"], i["operator_verdicts"], i["defective"]) == (1, 1, 1, 1)


# ------------------------------------------------------------------------------------------------ API / security
@pytest.mark.parametrize("query,status", [
    ("start_date=2026-13-01", 422),
    ("start_date=yesterday", 422),
    ("start_date=2026-10-22&end_date=2026-10-20", 422),
    ("start_date=2024-01-01&end_date=2026-01-01", 422),
    ("granularity=hour", 422),
    ("store_id=DS-BLR-001' OR '1'='1", 422),
    ("store_id=DS-BLR-001;DROP TABLE inspections", 422),
    ("sku_id=%25", 422),
    ("store_id=" + "A" * 65, 422),
])
def test_invalid_parameters(dataset, ai_client, query, status):
    r = ai_client.get(f"/api/analytics/overview?{query}")
    assert r.status_code == status and r.json()["success"] is False and "Traceback" not in r.text
    assert overview(ai_client)["inspections"]["total"] == 11  # nothing was touched


def test_date_range_error_code(ai_client):
    r = ai_client.get("/api/analytics/overview?start_date=2026-10-22&end_date=2026-10-20")
    assert r.json()["error"]["code"] == "INVALID_DATE_RANGE"


def test_analytics_is_read_only(dataset, ai_client):
    def snapshot():
        with ai_client.app.state.session_factory() as db:
            return (db.scalar(select(func.count()).select_from(Inspection)),
                    db.scalar(select(func.count()).select_from(Feedback)),
                    [(r.inspection_id, r.decision, r.cv) for r in db.scalars(select(InspectionResult)
                                                                             .order_by(InspectionResult.result_id))],
                    [(i.inspection_id, i.status, i.decision) for i in db.scalars(select(Inspection)
                                                                                .order_by(Inspection.inspection_id))])
    before = snapshot()
    for q in (PERIOD, f"{PERIOD}&granularity=week", ""):
        overview(ai_client, q)
    assert snapshot() == before


def test_health_reports_analytics(ai_client):
    h = ai_client.get("/api/health").json()
    assert h["analytics"] == {"enabled": True, "status": "ready", "mode": "deterministic"}


def test_unlinked_inspections_are_their_own_trace(ai_client):
    """No item / order id: each inspection is analysed alone, so its defects are first observed at its own stage."""
    from tests.feedback.conftest import LEAK_COLOR, png
    for stage in ("PICKING", "DISPATCH"):
        iid = ai_client.post("/api/inspections", json={"store_id": "DS-BLR-002", "sku_id": "SKU-COLA-300ML",
                                                       "stage": stage}).json()["inspection_id"]
        ai_client.post(f"/api/inspections/{iid}/image", files={"file": (f"{stage}.png", png(LEAK_COLOR if stage ==
                       "PICKING" else (230, 30, 30)), "image/png")})
    o = overview(ai_client, "")
    assert {h["stage"]: h["first_observed_defects"] for h in o["hotspots"]} == {"PICKING": 1, "DISPATCH": 1}



def test_later_review_without_types_keeps_operator_defect(client, valid_payload):
    """Part 8 regression: verdict FAIL + LEAKING_PACKAGE, then a review restating FAIL without types (stub CV)."""
    iid = client.post("/api/inspections", json=valid_payload).json()["inspection_id"]
    client.post(f"/api/inspections/{iid}/image", files={"file": ("a.png", image_bytes(), "image/png")})
    for body in ({"outcome": "FAIL", "defect_types": ["LEAKING_PACKAGE"]}, {"outcome": "FAIL", "rca_useful": True}):
        assert client.post(f"/api/inspections/{iid}/feedback", json={"operator_id": "OP-1", **body}).status_code == 201
    o = client.get("/api/analytics/overview").json()
    assert o["inspections"]["defective"] == 1 and o["defects"][0]["defect_type"] == "LEAKING_PACKAGE"
    assert client.get(f"/api/inspections/{iid}/pipeline").json()["defect_analyses"][0]["defect_type"] == "LEAKING_PACKAGE"
