"""Persisted observations -> GET /api/inspections/{id}/pipeline, and the PIPELINE_ANALYSIS stage in the real
pipeline (stub CV here; the real-CV path is in tests/cv/test_fulfillment_full_path.py)."""
import pytest
from fastapi.testclient import TestClient

from backend.db.base import Base
from backend.main import create_app
from backend.services.reference_service import seed_sample_data
from tests.utils import image_bytes

BASE = {"store_id": "DS-BLR-001", "sku_id": "SKU-MILK-500ML"}
STAGES = ["RECEIVING", "STORAGE", "PICKING", "PACKING", "DISPATCH"]
_color = iter(range(10, 250, 3))


@pytest.fixture
def client(settings):
    app = create_app(settings.model_copy(update={"pipeline_analysis_engine": "deterministic"}))
    Base.metadata.create_all(app.state.engine)
    with app.state.session_factory() as s:
        seed_sample_data(s, settings.sample_data_path)
    with TestClient(app) as c:
        yield c


def inspect(c, stage, *, item="UNIT-0001", order=None, outcome=None, defects=(), upload=True, **extra):
    body = {**BASE, "stage": stage, **({"item_id": item} if item else {}), **({"order_id": order} if order else {}), **extra}
    ins = c.post("/api/inspections", json=body).json()
    iid = ins["inspection_id"]
    if upload:  # distinct image per inspection (identical photos would be de-duplicated)
        r = c.post(f"/api/inspections/{iid}/image", files={"file": ("p.png", image_bytes(color=(next(_color), 90, 40)), "image/png")})
        assert r.status_code == 200, r.text
    if outcome:
        r = c.post(f"/api/inspections/{iid}/feedback", json={"operator_id": "OP-1", "outcome": outcome, "defect_types": list(defects)})
        assert r.status_code == 201, r.text
    return iid


def pipeline(c, iid):
    r = c.get(f"/api/inspections/{iid}/pipeline")
    assert r.status_code == 200, r.text
    return r.json()


def test_case2_through_api_with_persisted_observations(client):
    for st in STAGES[:3]:
        inspect(client, st, outcome="PASS")
    packing = inspect(client, "PACKING", outcome="FAIL", defects=["CRUSHED_PACKAGE"])
    dispatch = inspect(client, "DISPATCH", outcome="FAIL", defects=["CRUSHED_PACKAGE"])
    inspect(client, "PACKING", item="UNIT-OTHER", outcome="FAIL", defects=["LEAKING_PACKAGE"])  # other item: excluded

    r = pipeline(client, packing)
    assert r["trace"] == {"key_type": "ITEM", "store_id": "DS-BLR-001", "sku_id": "SKU-MILK-500ML",
                          "item_id": "UNIT-0001", "order_id": None}
    assert [(t["stage"], t["status"]) for t in r["timeline"]] == [
        ("RECEIVING", "PASS"), ("STORAGE", "PASS"), ("PICKING", "PASS"), ("PACKING", "FAIL"), ("DISPATCH", "FAIL"),
        ("DELIVERY", "NOT_OBSERVED")]
    (a,) = r["defect_analyses"]
    assert (a["defect_type"], a["classification"], a["first_observed_stage"], a["last_known_good_stage"],
            a["evidence_level"], a["attribution_status"], a["current_stage_attribution"]) == (
        "CRUSHED_PACKAGE", "NEW_DEFECT", "PACKING", "PICKING", "HIGH", "ATTRIBUTED", "SUPPORTED")
    assert len(r["observations"]) == 5 and all(o["decision_source"] == "OPERATOR" for o in r["observations"])
    assert "image_sha256" not in r["observations"][0] and "does not prove physical causality" in r["disclaimer"]
    assert pipeline(client, dispatch)["defect_analyses"][0]["current_stage_attribution"] == "NOT_SUPPORTED"


def test_case3_and_case4_via_api(client):
    inspect(client, "RECEIVING", item="U-3", outcome="FAIL", defects=["CRUSHED_PACKAGE"])
    k = inspect(client, "PACKING", item="U-3", outcome="FAIL", defects=["CRUSHED_PACKAGE"])
    a = pipeline(client, k)["defect_analyses"][0]
    assert a["classification"] == "PRE_EXISTING" and a["current_stage_attribution"] == "NOT_SUPPORTED"

    inspect(client, "RECEIVING", item="U-4", outcome="PASS")
    k = inspect(client, "PACKING", item="U-4", outcome="FAIL", defects=["CRUSHED_PACKAGE"])
    a = pipeline(client, k)["defect_analyses"][0]
    assert a["classification"] == "INSUFFICIENT_EVIDENCE" and a["missing_stages"] == ["STORAGE", "PICKING"]


def test_unreviewed_inspections_are_review_not_pass(client):
    """Stub CV -> no decision -> REQUIRES_REVIEW. Without operator feedback these are REVIEW observations."""
    inspect(client, "RECEIVING", item="U-5")
    k = inspect(client, "PACKING", item="U-5", outcome="FAIL", defects=["DENTED_CONTAINER"])
    r = pipeline(client, k)
    assert r["timeline"][0]["status"] == "REVIEW"
    assert r["observations"][0]["decision_source"] == "NO_DECISION"
    assert r["defect_analyses"][0]["last_known_good_stage"] is None


def test_order_trace_and_in_flight_inspections(client):
    p = inspect(client, "PICKING", item=None, order="ORD-77", outcome="PASS")
    in_flight = inspect(client, "PACKING", item=None, order="ORD-77", upload=False)  # CREATED: not evidence yet
    inspect(client, "PACKING", item=None, order="ORD-78", outcome="FAIL", defects=["LEAKING_PACKAGE"])  # other order
    r = pipeline(client, in_flight)
    assert r["trace"]["key_type"] == "ORDER" and r["inspected_stages"] == ["PICKING"]
    assert r["status"] == "NOT_ATTRIBUTED" and r["explanation_code"] == "NO_DEFECT_OBSERVED"
    assert pipeline(client, p)["observations"][0]["inspection_id"] == p


def test_no_identifier_is_isolated(client):
    k = inspect(client, "PACKING", item=None, outcome="FAIL", defects=["LEAKING_PACKAGE"])
    r = pipeline(client, k)
    assert r["trace"]["key_type"] == "NONE" and "NO_TRACE_IDENTIFIER" in r["flags"]
    assert r["defect_analyses"][0]["explanation_code"] == "NO_PRIOR_OBSERVATION"


def test_pipeline_stage_snapshot_and_health(client):
    inspect(client, "RECEIVING", item="U-9", outcome="PASS")
    k = inspect(client, "PACKING", item="U-9")
    body = client.get(f"/api/inspections/{k}").json()
    pa = body["result"]["pipeline_analysis"]
    assert pa["status"] == "COMPLETED" and pa["module"] == "deterministic"
    assert pa["trace_analysis"]["inspected_stages"] == ["RECEIVING", "PACKING"]
    assert pa["stage_likelihoods"] == {} and pa["confidence"] is None  # no probabilities claimed
    assert body["model_versions"]["pipeline_analysis"] == "deterministic:1"
    h = client.get("/api/health").json()["pipeline_modules"]["pipeline_analyzer"]
    assert h == {"name": "deterministic", "implemented": True, "available": True, "version": "1"}


def test_validation_and_errors(client):
    assert client.get("/api/inspections/INS-0000000000000000/pipeline").json()["error"]["code"] == "INSPECTION_NOT_FOUND"
    assert client.get("/api/inspections/bad/pipeline").status_code == 422
    r = client.post("/api/inspections", json={**BASE, "stage": "PACKING", "item_id": "bad id/../x"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"
    r = client.post("/api/inspections", json={**BASE, "stage": "SHIPPING", "item_id": "U-1"})
    assert r.status_code == 422


def test_list_filters_by_item_and_order(client):
    inspect(client, "RECEIVING", item="U-LIST", upload=False)
    inspect(client, "PICKING", item=None, order="ORD-LIST", upload=False)
    assert client.get("/api/inspections", params={"item_id": "U-LIST"}).json()["total"] == 1
    assert client.get("/api/inspections", params={"order_id": "ORD-LIST"}).json()["items"][0]["order_id"] == "ORD-LIST"
    assert client.get("/api/inspections", params={"item_id": "bad id"}).status_code == 422
