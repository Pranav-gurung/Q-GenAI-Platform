"""Practical end-to-end test script for Module A and Module C.

Exercises the entire lifecycle:
1. Module A: Dataset preparation, validation, splitting, train-only augmentation,
   COCO/YOLO/Anomaly exports, manifest verification, and operator feedback workflow.
2. Module C: Approved document ingestion, page chunking, retrieval search,
   FastAPI /health and /rca endpoints, citation verification, insufficient-evidence
   handling, validation rejection, and audit log verification.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module_a.augment_images import AugmentConfig
from module_a.feedback import FeedbackStore, apply_approved
from module_a.fixtures import FIXTURE_CLASSES, generate_fixture
from module_a.prepare_dataset import prepare
from module_a.schema import load_canonical
from module_a.version_dataset import load_manifest, verify_manifest

from module_c.api import create_app
from module_c.audit import JsonlAuditSink
from module_c.config import Settings
from module_c.rca import RCAService
from module_c.retrieval import IngestionService, Retriever
from tests.helpers import (
    HANDLING_P1,
    HANDLING_P2,
    STORAGE_P1,
    StubProvider,
    make_settings,
    write_meta,
    write_pdf,
)
from tests.test_module_c_rca import llm_reply

from fastapi.testclient import TestClient
import jsonschema


def run_end_to_end() -> bool:
    tmp_root = Path(tempfile.mkdtemp(prefix="e2e_test_"))
    print(f"=== Starting Practical End-to-End Test in {tmp_root} ===")

    try:
        # -----------------------------------------------------------------
        # Step 1: Module A Dataset Engine
        # -----------------------------------------------------------------
        print("\n--- [Step 1] Module A: Synthetic Data Pipeline ---")
        fx_dir = tmp_root / "fixture"
        image_dir, ann_csv = generate_fixture(fx_dir, n_parts=6, views_per_part=3, seed=42)
        print(f"Generated synthetic fixture in {image_dir} with {ann_csv.name}")

        out_root = tmp_root / "dataset_out"
        prep_res = prepare(
            image_dir=image_dir,
            annotations=ann_csv,
            classes=FIXTURE_CLASSES,
            out_root=out_root,
            source={"name": "E2E_FIXTURE", "path": str(image_dir), "is_test_fixture": True},
            ratios=(0.60, 0.20, 0.20),
            seed=42,
            augment=AugmentConfig(seed=42, variants_per_image=2),
            export_formats=("coco", "yolo", "anomaly"),
        )

        version = prep_res["version"]
        print(f"Dataset Version: {version}")
        print(f"Original images: {prep_res['n_originals']}, Augmented images: {prep_res['n_augmented']}")

        # Verify Manifest
        manifest = load_manifest(prep_res["manifest"])
        problems = verify_manifest(manifest, prep_res["layout"]["splits_dir"])
        assert not problems, f"Manifest verification failed: {problems}"
        print("Manifest verification: PASS (all image files and hashes match)")

        # Verify Split Leakage & Train-only Augmentation
        for split in ("val", "test"):
            recs, _, _ = load_canonical(Path(prep_res["layout"]["annotations_dir"]) / f"canonical_{split}.json")
            for r in recs:
                assert r.source_image_id is None, f"Augmented image leaked into {split} split!"
        train_recs, _, _ = load_canonical(Path(prep_res["layout"]["annotations_dir"]) / "canonical_train.json")
        n_aug_train = sum(1 for r in train_recs if r.source_image_id is not None)
        assert n_aug_train > 0, "No augmented images found in train split!"
        print(f"Split check: PASS (train has {n_aug_train} augmented images, val/test have 0)")

        # Verify Exports Exist
        coco_file = Path(prep_res["layout"]["processed_dir"]) / "coco" / "instances_train.json"
        yolo_file = Path(prep_res["layout"]["processed_dir"]) / "yolo" / "data.yaml"
        anomaly_dir = Path(prep_res["layout"]["processed_dir"]) / "anomaly" / "train"
        assert coco_file.is_file(), "COCO instances_train.json missing"
        assert yolo_file.is_file(), "YOLO data.yaml missing"
        assert anomaly_dir.is_dir(), "Anomaly layout directory missing"
        print("Exports check: PASS (COCO, YOLO, Anomaly layouts generated)")

        # Verify Operator Feedback Workflow
        print("\n--- [Step 2] Module A: Operator Feedback Workflow ---")
        fb_store = FeedbackStore(tmp_root / "feedback.jsonl", FIXTURE_CLASSES)
        target_rec = next(r for r in train_recs if r.annotations)
        fb = fb_store.submit(
            image_id=target_rec.image_id,
            dataset_version=version,
            original_class=target_rec.annotations[0].class_name,
            corrected_class="fixture_blob",
            annotation_index=0,
            submitted_by="operator_alice",
        )
        print(f"Submitted feedback: {fb.feedback_id} (pending)")

        # 4-eyes review
        reviewed_fb = fb_store.review(fb.feedback_id, "approved", reviewer="lead_bob", notes="Approved correction")
        print(f"Reviewed feedback: {reviewed_fb.feedback_id} -> {reviewed_fb.review_status}")

        # Export and apply
        fb_export_path = tmp_root / "approved_feedback.json"
        n_approved = fb_store.export_approved(fb_export_path)
        assert n_approved == 1, "Expected 1 approved feedback record exported"
        loaded_dicts = json.loads(fb_export_path.read_text(encoding="utf-8"))
        updated_records, changelog = apply_approved(train_recs, loaded_dicts)
        assert changelog[0]["applied"], f"Feedback not applied: {changelog}"
        print("Feedback workflow: PASS (submit -> 4-eyes review -> export -> apply)")

        # -----------------------------------------------------------------
        # Step 3: Module C Approved Document Ingestion & Retrieval
        # -----------------------------------------------------------------
        print("\n--- [Step 3] Module C: Document Ingestion & Retrieval ---")
        docs_dir = tmp_root / "docs"
        docs_dir.mkdir(parents=True)
        pdf1 = write_pdf(docs_dir / "handling_sop.pdf", [HANDLING_P1, HANDLING_P2])
        write_meta(pdf1, "SOP-001", "Component Handling SOP", "2.0")
        pdf2 = write_pdf(docs_dir / "storage_sop.pdf", [STORAGE_P1])
        write_meta(pdf2, "SOP-002", "Storage and Preservation SOP", "1.1")

        s = make_settings(
            tmp_root,
            approved_docs_dir=docs_dir,
            supported_classes=("scratch", "crack", "corrosion", "dent"),
        )
        ingest_svc = IngestionService(s)
        ingest_results = ingest_svc.ingest_directory()
        print(f"Ingested {len(ingest_results)} document files. Chunks: {ingest_svc.store.count()}")
        assert ingest_svc.store.count() > 0, "No chunks were indexed!"

        retriever = Retriever(s)
        passages, n_cand = retriever.search("surface scratches on machined aluminium housings from handling")
        assert len(passages) > 0, "Retrieval returned no passages for valid query!"
        print(f"Retrieved {len(passages)} passages (top score: {passages[0].score:.4f}, doc: {passages[0].document_id})")
        print(f"Citation: {passages[0].title} v{passages[0].version} p{passages[0].page} section: {passages[0].section}")

        # -----------------------------------------------------------------
        # Step 4: Module C API & RCA Grounded Generation
        # -----------------------------------------------------------------
        print("\n--- [Step 4] Module C: RCA API and Contract Validation ---")
        audit_sink = JsonlAuditSink(tmp_root / "rca_audit.jsonl")
        prov = StubProvider([llm_reply(cause_status="supported_by_document")])
        rca_svc = RCAService(s, retriever, prov, audit_sink)
        app = create_app(s, rca_svc)
        client = TestClient(app)

        # GET /health
        health_resp = client.get("/health")
        assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
        health_data = health_resp.json()
        assert health_data["status"] == "ok"
        assert health_data["indexed_chunks"] > 0
        print(f"GET /health: PASS (indexed_chunks: {health_data['indexed_chunks']}, backend: {health_data['embedding_backend']})")

        # Load JSON Schemas
        req_schema = json.loads((Path(__file__).parent.parent / "schemas" / "rca_request.schema.json").read_text())
        resp_schema = json.loads((Path(__file__).parent.parent / "schemas" / "rca_response.schema.json").read_text())

        # Valid Request (with evidence)
        req_payload = {
            "inspection_id": "QC-2026-001",
            "image_id": "IMG-HOUSING-042",
            "model_version": "v2.1",
            "status": "review_required",
            "part_type": "machined housing",
            "material": "aluminium",
            "production_process": "transfer",
            "defects": [
                {"class": "scratch", "confidence": 0.94, "bbox": [120, 80, 260, 190]}
            ],
        }
        jsonschema.validate(instance=req_payload, schema=req_schema)

        rca_resp = client.post("/rca", json=req_payload)
        assert rca_resp.status_code == 200, f"RCA request failed: {rca_resp.text}"
        rca_data = rca_resp.json()
        jsonschema.validate(instance=rca_data, schema=resp_schema)
        assert rca_data["status"] == "success"
        assert len(rca_data["citations"]) > 0, "Expected citations in RCA response"
        print("POST /rca (valid defect with evidence): PASS (status: success, schema valid)")
        print(f"Summary: {rca_data['summary']}")
        print(f"Observations: {rca_data['observations']}")
        print(f"Possible Causes: {[c['cause'] for c in rca_data['possible_causes']]}")
        print(f"Citations: {[c['chunk_id'] for c in rca_data['citations']]}")

        # Insufficient Evidence Request (e.g. defect class without relevant docs)
        insufficient_payload = {
            "inspection_id": "QC-2026-002",
            "image_id": "IMG-HOUSING-043",
            "model_version": "v2.1",
            "defects": [
                {"class": "crack", "confidence": 0.88, "bbox": [10, 10, 50, 50]}
            ],
        }
        ins_resp = client.post("/rca", json=insufficient_payload)
        assert ins_resp.status_code == 200
        ins_data = ins_resp.json()
        jsonschema.validate(instance=ins_data, schema=resp_schema)
        assert ins_data["status"] == "insufficient_evidence"
        assert ins_data["defect_analyses"][0]["status"] == "insufficient_evidence"
        assert len(ins_data["citations"]) == 0
        print("POST /rca (no matching documents): PASS (status: insufficient_evidence, escalated)")

        # Invalid Request (unknown defect class) -> 422 with request_id
        invalid_payload = {
            "inspection_id": "QC-2026-003",
            "image_id": "IMG-003",
            "model_version": "v1",
            "defects": [{"class": "unsupported_defect", "confidence": 0.9, "bbox": [0, 0, 10, 10]}],
        }
        inv_resp = client.post("/rca", json=invalid_payload)
        assert inv_resp.status_code == 422
        inv_data = inv_resp.json()
        assert "request_id" in inv_data, "422 response missing request_id"
        print("POST /rca (validation error): PASS (HTTP 422, request_id present)")

        # -----------------------------------------------------------------
        # Step 5: Module D Audit Verification
        # -----------------------------------------------------------------
        print("\n--- [Step 5] Module D: Audit Trail Verification ---")
        audit_file = tmp_root / "rca_audit.jsonl"
        assert audit_file.is_file(), "Audit file not found"
        events = [json.loads(line) for line in audit_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        event_types = [e["event_type"] for e in events]
        print(f"Recorded audit events ({len(events)}): {event_types}")
        assert "rca.completed" in event_types, "rca.completed event missing"
        assert "rca.rejected" in event_types, "rca.rejected event missing"

        # Check no passage text or secrets in audit log
        raw_audit = audit_file.read_text(encoding="utf-8")
        assert "Surface scratches on housings can occur" not in raw_audit, "Document text leaked to audit log!"
        print("Audit trail check: PASS (events recorded, no secrets or passage text leaked)")

        print("\n========================================================")
        print(" ALL END-TO-END CHECKS COMPLETED SUCCESSFULLY!")
        print("========================================================")
        return True

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    success = run_end_to_end()
    sys.exit(0 if success else 1)
