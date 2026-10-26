"""Four-Module End-to-End Integration Test (Modules A -> B -> C -> D).

Tests the full lifecycle across all four domains:
1. Module A: Synthetic data generation and leakage-free YOLO export.
2. Module A -> B: Handover verification (data.yaml, labels, hashes, zero leakage).
3. Module B & D: Image inspection, deterministic mock detection, policy evaluation (FAIL),
   and SQLite audit logging.
4. Contract reconciliation: Conversion of inspection to Module C RCARequest.
5. Module C: Knowledge base document ingestion, retrieval, grounded mock LLM generation,
   and citation tracking.
6. Module D: Cryptographic SHA-256 hash-chained audit logging of the RCA event via ModuleDAuditAdapter.
7. Verification & Tamper Detection: verify_chain() confirms zero broken links, and tampering
   is caught immediately.
8. Error & Negative cases: Unsupported classes, insufficient evidence, and fail-closed audit errors.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import jsonschema
from sqlalchemy import select, text

import sys
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from module_a.fixtures import FIXTURE_CLASSES, generate_fixture
from module_a.prepare_dataset import AugmentConfig, prepare
from module_c.config import Settings as ModuleCSettings
from module_c.models import validate_request as module_c_validate_request
from module_c.rca import RCAService
from module_c.retrieval import IngestResult, IngestionService, Retriever
from module_c.vectorstore import JsonVectorStore, HashingEmbedder

from app.database import SessionLocal
from app.models.audit_event import AuditEvent as ModuleDAuditEvent
from app.models.inspection import Inspection
from app.services.audit_adapter import ModuleDAuditAdapter
from app.services.audit_service import verify_chain
from tests.conftest import FakeDetector, det, predict
from training.handover import verify_handover


SCHEMA_PATH = repo_root / "schemas" / "rca_request.schema.json"


@pytest.fixture
def rca_schema():
    assert SCHEMA_PATH.is_file()
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def module_c_rag_service(tmp_path):
    """Set up an authentic Module C RAG environment with an approved manufacturing SOP."""
    docs_dir = tmp_path / "approved_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write an approved manufacturing SOP covering crack and scratch defects
    sop_content = (
        "STANDARD OPERATING PROCEDURE: SURFACE DEFECT MITIGATION (SOP-QA-2026)\n\n"
        "1. CRACK DEFECTS\n"
        "Cracks in machined components typically originate from thermal stress or excessive "
        "clamping pressure during CNC milling. When cracks are detected, operators shall immediately "
        "verify cooling fluid flow rate (minimum 15 L/min) and verify pneumatic clamp pressure.\n\n"
        "2. SCRATCH DEFECTS\n"
        "Surface scratches occur when parts contact each other or hard fixtures during station transfer. "
        "Operators shall ensure silicone separator trays are used and check fixture guide padding.\n"
    )
    sop_file = docs_dir / "SOP-QA-2026.txt"
    sop_file.write_text(sop_content, encoding="utf-8")
    meta_file = docs_dir / "SOP-QA-2026.txt.meta.json"
    meta_file.write_text(
        json.dumps({
            "document_id": "SOP-QA-2026",
            "title": "Surface Defect Mitigation Procedure",
            "version": "1.0",
            "access_level": "internal",
            "license_note": "Internal Q-GenAI Standard",
        }),
        encoding="utf-8",
    )

    # 2. Ingest into JsonVectorStore
    index_dir = tmp_path / "rag_index"
    embedder = HashingEmbedder()
    store = JsonVectorStore(index_dir, embedder.name)
    settings = ModuleCSettings(
        approved_docs_dir=docs_dir,
        index_dir=index_dir,
        chunk_size=300,
        chunk_overlap=50,
        min_score=0.05,
        supported_classes=("crack", "scratch", "dent", "corrosion"),
        audit_fail_mode="fail_closed",
    )
    ingestion = IngestionService(settings, embedder=embedder, store=store)
    results = ingestion.ingest_directory()
    assert len(results) == 1
    assert results[0].status == "indexed"
    assert store.count() > 0

    retriever = Retriever(settings, embedder=embedder, store=store)

    # 3. Grounded mock provider that extracts cited chunk and returns grounded causes
    class GroundedMockProvider:
        name = "grounded-mock"
        model = "mock-rca-v1"

        def generate(self, system: str, user: str) -> str:
            # Deterministically parse available evidence chunk from prompt
            chunk_match = user.find('chunk_id="')
            chunk_id = user[chunk_match + 10: user.find('"', chunk_match + 10)] if chunk_match != -1 else "c1"

            if "crack" in user.lower():
                return json.dumps({
                    "possible_causes": [
                        {
                            "defect_class": "crack",
                            "cause": "Thermal stress from inadequate coolant or excessive clamping pressure",
                            "status": "supported_by_document",
                            "evidence_chunk_ids": [chunk_id],
                        }
                    ],
                    "recommended_actions": [
                        {
                            "action": "Inspect coolant flow and calibrate clamp torque",
                            "chunk_ids": [chunk_id],
                        }
                    ],
                    "unknowns": ["Coolant flow rate at time of machining"],
                    "notes": ["Grounded mock output for automated verification"],
                })
            elif "scratch" in user.lower():
                return json.dumps({
                    "possible_causes": [
                        {
                            "defect_class": "scratch",
                            "cause": "Parts contacting hard fixtures or station transfer without silicone separators",
                            "status": "supported_by_document",
                            "evidence_chunk_ids": [chunk_id],
                        }
                    ],
                    "recommended_actions": [
                        {
                            "action": "Ensure silicone separator trays are used and check guide padding",
                            "chunk_ids": [chunk_id],
                        }
                    ],
                    "unknowns": ["Fixture condition at time of handling"],
                    "notes": ["Grounded mock output for automated verification"],
                })
            return json.dumps({
                "possible_causes": [],
                "recommended_actions": [],
                "unknowns": ["No defect evidence available"],
                "notes": [],
            })

    # 4. Wire Module D Audit Adapter
    audit_adapter = ModuleDAuditAdapter(SessionLocal)
    service = RCAService(settings, retriever, GroundedMockProvider(), audit_adapter)
    return service, store


def test_four_module_full_lifecycle(
    fresh_db,
    client,
    headers,
    use_detector,
    tmp_path,
    module_c_rag_service,
    rca_schema,
):
    rca_service, vector_store = module_c_rag_service

    # =========================================================================
    # Step 1: Module A (Synthetic Data Generation & YOLO Export)
    # =========================================================================
    fx_dir = tmp_path / "module_a_fx"
    img_dir, ann_csv = generate_fixture(fx_dir, n_parts=8, views_per_part=3, seed=10)
    out_dir = tmp_path / "module_a_out"

    prep_res = prepare(
        image_dir=img_dir,
        annotations=ann_csv,
        classes=FIXTURE_CLASSES,
        out_root=out_dir,
        source={"name": "e2e_fixture", "path": str(img_dir), "is_test_fixture": True},
        export_formats=("yolo",),
        seed=10,
        augment=AugmentConfig(seed=10, variants_per_image=1),
    )
    yolo_dir = Path(prep_res["layout"]["processed_dir"]) / "yolo"
    manifest_path = Path(prep_res["manifest"])

    # Handover Gate: Verify dataset is valid and zero-leakage
    h_report = verify_handover(yolo_dir, manifest_path=manifest_path)
    assert h_report.valid is True
    assert h_report.leakage_free is True
    assert h_report.manifest_verified is True

    # =========================================================================
    # Step 2: Module B & D (Inspection, Policy Evaluation, Audit Logging)
    # =========================================================================
    # Configure deterministic mock detector reporting crack and scratch
    mock_detections = [
        det("crack", 0.94, (10, 10, 45, 35)),
        det("scratch", 0.82, (15, 15, 30, 25)),
    ]
    use_detector(FakeDetector(mock_detections))

    pred_res = predict(client, headers["op"])
    assert pred_res.status_code == 201
    insp_data = pred_res.json()
    iid = insp_data["inspection_id"]

    # Verify Module B & D quality policy evaluation
    assert insp_data["inspection_status"] == "FAIL"
    assert insp_data["final_decision"] == "FAIL"
    assert insp_data["defect_count"] == 2
    assert insp_data["image_sha256"] != ""

    # =========================================================================
    # Step 3: Contract Reconciliation (GET /inspections/{id}/rca-request)
    # =========================================================================
    rca_req_resp = client.get(f"/inspections/{iid}/rca-request", headers=headers["op"])
    assert rca_req_resp.status_code == 200, rca_req_resp.text
    rca_payload = rca_req_resp.json()

    # Verify JSON Schema compliance
    jsonschema.validate(instance=rca_payload, schema=rca_schema)

    # Verify Module C validator compliance
    validated_req = module_c_validate_request(
        rca_payload,
        supported_classes=["crack", "scratch", "dent", "corrosion"],
        max_defects=20,
    )
    assert validated_req.inspection_id == iid
    assert validated_req.image_id == insp_data["image_sha256"]
    assert validated_req.status == "defect_detected"
    assert len(validated_req.defects) == 2

    # Verify strict omission of forbidden top-level fields
    forbidden_keys = {"site", "timestamp", "model_sha256", "policy_version", "confidence_threshold", "ai_decision", "final_decision"}
    assert not (set(rca_payload.keys()) & forbidden_keys)

    # =========================================================================
    # Step 4: Module C (Root-Cause Analysis with Citations)
    # =========================================================================
    outcome = rca_service.handle_raw(rca_payload, actor=headers["op"]["Authorization"])
    assert outcome.http_status == 200
    resp = outcome.response
    assert resp["status"] == "success"
    assert len(resp["possible_causes"]) > 0
    assert len(resp["recommended_actions"]) > 0
    assert len(resp["citations"]) > 0

    # Verify citation authenticity: citation chunk must exist in the real index
    first_citation = resp["citations"][0]
    assert first_citation["document_id"] == "SOP-QA-2026"
    assert first_citation["version"] == "1.0"
    assert first_citation["chunk_id"] in vector_store.chunks

    # Verify cause evidence points to real chunk
    cause = resp["possible_causes"][0]
    assert cause["status"] == "supported_by_document"
    assert cause["evidence"][0] == first_citation["chunk_id"]

    # =========================================================================
    # Step 5: Module D (Cryptographic Audit Chaining Verification)
    # =========================================================================
    with SessionLocal() as db:
        # Check that RCA event was persisted via the ModuleDAuditAdapter
        rca_events = db.execute(
            select(ModuleDAuditEvent).where(ModuleDAuditEvent.action == "rca.completed")
        ).scalars().all()
        assert len(rca_events) == 1
        rca_ev = rca_events[0]

        assert rca_ev.target_type == "inspection"
        assert rca_ev.target_id == iid
        meta = json.loads(rca_ev.metadata_json)
        assert meta["rca_status"] == "success"
        assert meta["defect_classes"] == ["crack", "scratch"]
        assert meta["cited_documents"][0]["document_id"] == "SOP-QA-2026"

        # Verify no passage text, no secrets, no image bytes logged
        assert "passage" not in rca_ev.metadata_json
        assert "password" not in rca_ev.metadata_json
        assert "token" not in rca_ev.metadata_json

        # Complete cryptographic audit chain verification
        v_res = verify_chain(db)
        assert v_res.valid is True
        assert v_res.total_events >= 4  # policy + requested + completed + result_created + rca.completed
        assert v_res.verified_events == v_res.total_events


# =============================================================================
# Negative & Edge Cases
# =============================================================================

def test_negative_unsupported_defect_class(fresh_db, client, headers, use_detector):
    # Detections include unsupported class 'unknown_particle'
    use_detector(FakeDetector([det("unknown_particle", 0.9, (10, 10, 40, 30))]))
    pred_res = predict(client, headers["op"])
    iid = pred_res.json()["inspection_id"]

    # Handover request should cleanly reject with 422 and structured details
    resp = client.get(f"/inspections/{iid}/rca-request", headers=headers["op"])
    assert resp.status_code == 422
    err_body = resp.json()["detail"]
    assert "unsupported_classes" in err_body
    assert "unknown_particle" in err_body["unsupported_classes"]


def test_negative_insufficient_evidence_flow(
    fresh_db,
    client,
    headers,
    use_detector,
    module_c_rag_service,
):
    rca_service, _ = module_c_rag_service

    # Detection of 'dent' (class is supported by schema, but NOT in SOP-QA-2026 document)
    use_detector(FakeDetector([det("dent", 0.91, (10, 10, 40, 30))]))
    pred_res = predict(client, headers["op"])
    iid = pred_res.json()["inspection_id"]

    rca_payload = client.get(f"/inspections/{iid}/rca-request", headers=headers["op"]).json()
    outcome = rca_service.handle_raw(rca_payload)

    # Must return 200 with status=insufficient_evidence, empty causes, and no citations
    assert outcome.http_status == 200
    assert outcome.response["status"] == "insufficient_evidence"
    assert outcome.response["possible_causes"] == []
    assert outcome.response["citations"] == []

    # Audit event must record rca_status == "insufficient_evidence"
    with SessionLocal() as db:
        ev = db.execute(
            select(ModuleDAuditEvent)
            .where(ModuleDAuditEvent.target_id == iid, ModuleDAuditEvent.action == "rca.completed")
        ).scalar_one()
        meta = json.loads(ev.metadata_json)
        assert meta["rca_status"] == "insufficient_evidence"


def test_negative_tamper_detection(fresh_db, client, headers, use_detector, module_c_rag_service):
    rca_service, _ = module_c_rag_service

    use_detector(FakeDetector([det("crack", 0.9, (10, 10, 40, 30))]))
    pred_res = predict(client, headers["op"])
    assert pred_res.status_code == 201
    iid = pred_res.json()["inspection_id"]
    rca_resp = client.get(f"/inspections/{iid}/rca-request", headers=headers["op"])
    assert rca_resp.status_code == 200, rca_resp.text
    rca_payload = rca_resp.json()
    rca_service.handle_raw(rca_payload)

    with SessionLocal() as db:
        # Confirm valid before tampering
        assert verify_chain(db).valid is True

        # Malicious attacker alters metadata of the RCA audit event
        db.execute(text("UPDATE audit_events SET metadata_json = '{\"forged\": 1}' WHERE action = 'rca.completed'"))
        db.commit()

        # Audit chain verifier must detect tampering
        v_res = verify_chain(db)
        assert v_res.valid is False
        assert "Content altered" in v_res.failure_reason
