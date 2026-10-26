#!/usr/bin/env python3
"""Q-GenAI Platform — Standalone Four-Module End-to-End Platform Test.

Executes the complete cross-module integration lifecycle:
  Step 1: Module A (Synthetic Data Engine) -> Generate fixture & export YOLO dataset.
  Step 2: Handover Gate (A -> B) -> Verify YOLO layout, manifest, and cross-split leakage.
  Step 3: Module B & D -> Image inspection, mock YOLO defect detection, quality decision.
  Step 4: Contract Reconciliation (B -> C) -> Convert inspection into canonical Module C payload.
  Step 5: Module C -> Ingest approved manufacturing SOP, retrieve passages, generate grounded RCA.
  Step 6: Module D -> Record RCA outcome into cryptographic SHA-256 hash-chained audit ledger.
  Step 7: Audit Integrity -> Cryptographic verification of the complete hash chain.
  Step 8: Negative Verification -> Tamper detection and unsupported class rejection.

Run:
  cd <project-root>
  .venv-bd/bin/python scripts/e2e_platform_test.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

# Add repo root and module_b_d to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_BD_DIR = REPO_ROOT / "module_b_d"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_BD_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_BD_DIR))

# Safe environment configuration for test run
os.environ["QGENAI_ENV"] = "development"
os.environ["QGENAI_JWT_SECRET_KEY"] = "platform-e2e-test-jwt-secret-key-32-chars-min"
os.environ["QGENAI_AUDIT_CHECKPOINT_KEY"] = "platform-e2e-test-checkpoint-key-32-chars-min"

import jsonschema
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from module_a.fixtures import FIXTURE_CLASSES, generate_fixture
from module_a.prepare_dataset import AugmentConfig, prepare
from module_c.config import Settings as ModuleCSettings
from module_c.models import validate_request as module_c_validate_request
from module_c.rca import RCAService
from module_c.retrieval import IngestionService, Retriever
from module_c.vectorstore import HashingEmbedder, JsonVectorStore

from app.database import Base
from app.services.detector import get_detector
from app.main import app as bd_app
from app.models.audit_event import AuditEvent as ModuleDAuditEvent
from app.models.user import User
from app.services.audit_adapter import ModuleDAuditAdapter
from app.services.audit_service import verify_chain
from app.services.authentication import hash_password
from app.services.detector import detector_manager
from app.services.policy_service import ensure_default_policy
from tests.conftest import FakeDetector, det, image_bytes
from training.handover import verify_handover


def print_step(num: int, title: str):
    print(f"\n\033[1;34m[Step {num}] {title}\033[0m")


def print_pass(msg: str):
    print(f"  \033[32m✔ PASS:\033[0m {msg}")


def print_fail(msg: str):
    print(f"  \033[31m✘ FAIL:\033[0m {msg}")


def main() -> int:
    print("\033[1;36m=================================================================\033[0m")
    print("\033[1;36m    Q-GenAI Platform — Full 4-Module End-to-End Platform Test    \033[0m")
    print("\033[1;36m=================================================================\033[0m")

    tmp_dir = Path(tempfile.mkdtemp(prefix="qgenai_e2e_"))
    try:
        # Set up isolated SQLite database
        db_path = tmp_dir / "test_platform.db"
        engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)

        # Seed admin user and default quality policy
        with TestingSessionLocal() as db:
            user = User(
                id=str(uuid.uuid4()),
                username="test_operator",
                full_name="Test Operator",
                password_hash=hash_password("Secret123!"),
                role="operator",
                site="default",
                is_active=True,
            )
            db.add(user)
            db.commit()
            ensure_default_policy(db)

        # Wire client and override db session
        from app.database import get_db
        def override_get_db():
            db = TestingSessionLocal()
            try:
                yield db
            finally:
                db.close()

        bd_app.dependency_overrides[get_db] = override_get_db
        client = TestClient(bd_app)

        # Obtain JWT token for test operator
        login_resp = client.post("/auth/login", data={"username": "test_operator", "password": "Secret123!"})
        assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
        auth_header = {"Authorization": f"Bearer {login_resp.json()['access_token']}"}

        # ---------------------------------------------------------------------
        # Step 1: Module A (Synthetic Data Generation & YOLO Export)
        # ---------------------------------------------------------------------
        print_step(1, "Module A: Synthetic Data Pipeline & YOLO Export")
        fx_dir = tmp_dir / "module_a_fx"
        img_dir, ann_csv = generate_fixture(fx_dir, n_parts=6, views_per_part=3, seed=42)
        out_dir = tmp_dir / "module_a_out"

        prep_res = prepare(
            image_dir=img_dir,
            annotations=ann_csv,
            classes=FIXTURE_CLASSES,
            out_root=out_dir,
            source={"name": "e2e_test", "path": str(img_dir), "is_test_fixture": True},
            export_formats=("yolo",),
            seed=42,
            augment=AugmentConfig(seed=42, variants_per_image=1),
        )
        yolo_dir = Path(prep_res["layout"]["processed_dir"]) / "yolo"
        manifest_path = Path(prep_res["manifest"])
        print_pass(f"Generated synthetic dataset: {prep_res['n_originals']} originals, {prep_res['n_augmented']} augmented")
        print_pass(f"YOLO layout exported to {yolo_dir}")

        # ---------------------------------------------------------------------
        # Step 2: Handover Gate (Module A -> Module B)
        # ---------------------------------------------------------------------
        print_step(2, "Handover Gate: Module A -> Module B Dataset Validation")
        h_report = verify_handover(yolo_dir, manifest_path=manifest_path)
        assert h_report.valid is True, f"Handover validation failed: {h_report.errors}"
        assert h_report.leakage_free is True, f"Leakage detected: {h_report.leakage_findings}"
        assert h_report.manifest_verified is True
        print_pass(f"Handover report: valid=True, class_names={h_report.class_names}")
        print_pass(f"Splits verified: {list(h_report.splits.keys())}")
        print_pass("Cross-split leakage check: 0 duplicate hashes, 0 part-group leaks")

        # ---------------------------------------------------------------------
        # Step 3: Module B & D (Inspection & Mock Defect Detection)
        # ---------------------------------------------------------------------
        print_step(3, "Module B & D: Inspection Ingestion, Mock YOLO Detection & Policy Evaluation")
        mock_detections = [
            det("crack", 0.94, (10, 10, 45, 35)),
            det("scratch", 0.82, (15, 15, 30, 25)),
        ]
        bd_app.dependency_overrides[get_detector] = lambda: FakeDetector(mock_detections)

        img_payload = image_bytes()
        pred_resp = client.post(
            "/predict",
            files={"file": ("component.png", img_payload, "image/png")},
            headers=auth_header,
        )
        assert pred_resp.status_code == 201, f"Predict failed: {pred_resp.text}"
        insp_data = pred_resp.json()
        iid = insp_data["inspection_id"]
        assert insp_data["final_decision"] == "FAIL"
        assert insp_data["defect_count"] == 2
        print_pass(f"Inspection created: {iid} | Decision: {insp_data['final_decision']} | Defects: {insp_data['defect_count']}")

        # ---------------------------------------------------------------------
        # Step 4: Contract Reconciliation (GET /inspections/{id}/rca-request)
        # ---------------------------------------------------------------------
        print_step(4, "Contract Reconciliation: B -> C Normalized RCA Request")
        rca_req_resp = client.get(f"/inspections/{iid}/rca-request", headers=auth_header)
        assert rca_req_resp.status_code == 200, f"rca-request failed: {rca_req_resp.text}"
        rca_payload = rca_req_resp.json()

        # Validate against JSON Schema
        schema_path = REPO_ROOT / "schemas" / "rca_request.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(instance=rca_payload, schema=schema)
        print_pass("Validated payload against schemas/rca_request.schema.json")

        # Validate against Module C Pydantic validator
        val_req = module_c_validate_request(
            rca_payload,
            supported_classes=["crack", "scratch", "dent", "corrosion"],
            max_defects=20,
        )
        assert val_req.inspection_id == iid
        assert val_req.image_id == insp_data["image_sha256"]
        print_pass(f"Validated with Module C validator (image_id: {val_req.image_id[:16]}..., status: {val_req.status})")

        # ---------------------------------------------------------------------
        # Step 5: Module C (Approved Document Ingestion & Grounded RCA)
        # ---------------------------------------------------------------------
        print_step(5, "Module C: Approved Document Ingestion, Semantic Retrieval & Grounded RCA")
        docs_dir = tmp_dir / "approved_docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        sop_file = docs_dir / "SOP-QA-2026.txt"
        sop_file.write_text(
            "STANDARD OPERATING PROCEDURE: SURFACE DEFECT MITIGATION (SOP-QA-2026)\n\n"
            "1. CRACK DEFECTS\n"
            "Cracks in machined components typically originate from thermal stress or excessive "
            "clamping pressure during CNC milling. When cracks are detected, operators shall immediately "
            "verify cooling fluid flow rate (minimum 15 L/min) and verify pneumatic clamp pressure.\n\n"
            "2. SCRATCH DEFECTS\n"
            "Surface scratches occur when parts contact each other or hard fixtures during station transfer. "
            "Operators shall ensure silicone separator trays are used and check fixture guide padding.\n",
            encoding="utf-8",
        )
        meta_file = docs_dir / "SOP-QA-2026.txt.meta.json"
        meta_file.write_text(json.dumps({
            "document_id": "SOP-QA-2026",
            "title": "Surface Defect Mitigation Procedure",
            "version": "1.0",
            "access_level": "internal",
        }), encoding="utf-8")

        index_dir = tmp_dir / "rag_index"
        embedder = HashingEmbedder()
        store = JsonVectorStore(index_dir, embedder.name)
        c_settings = ModuleCSettings(
            approved_docs_dir=docs_dir,
            index_dir=index_dir,
            chunk_size=300,
            chunk_overlap=50,
            min_score=0.05,
            supported_classes=("crack", "scratch", "dent", "corrosion"),
            audit_fail_mode="fail_closed",
        )
        ingest_svc = IngestionService(c_settings, embedder=embedder, store=store)
        ingest_res = ingest_svc.ingest_directory()
        print_pass(f"Ingested approved SOP: {ingest_res[0].filename} -> {store.count()} chunks indexed")

        retriever = Retriever(c_settings, embedder=embedder, store=store)

        # Grounded mock provider
        class GroundedMockProvider:
            name = "grounded-mock"
            model = "mock-rca-v1"

            def generate(self, system: str, user: str) -> str:
                chunk_match = user.find('chunk_id="')
                chunk_id = user[chunk_match + 10: user.find('"', chunk_match + 10)] if chunk_match != -1 else "c1"
                if "crack" in user.lower():
                    return json.dumps({
                        "possible_causes": [{
                            "defect_class": "crack",
                            "cause": "Thermal stress from inadequate coolant or excessive clamping pressure",
                            "status": "supported_by_document",
                            "evidence_chunk_ids": [chunk_id],
                        }],
                        "recommended_actions": [{
                            "action": "Inspect coolant flow and calibrate clamp torque",
                            "chunk_ids": [chunk_id],
                        }],
                        "unknowns": ["Coolant flow rate at time of machining"],
                        "notes": ["Automated grounded mock output"],
                    })
                elif "scratch" in user.lower():
                    return json.dumps({
                        "possible_causes": [{
                            "defect_class": "scratch",
                            "cause": "Parts contacting hard fixtures during station transfer",
                            "status": "supported_by_document",
                            "evidence_chunk_ids": [chunk_id],
                        }],
                        "recommended_actions": [{
                            "action": "Ensure silicone separator trays are used and check guide padding",
                            "chunk_ids": [chunk_id],
                        }],
                        "unknowns": ["Fixture condition at handling station"],
                        "notes": ["Automated grounded mock output"],
                    })
                return json.dumps({"possible_causes": [], "recommended_actions": [], "unknowns": [], "notes": []})

        audit_adapter = ModuleDAuditAdapter(TestingSessionLocal)
        rca_svc = RCAService(c_settings, retriever, GroundedMockProvider(), audit_adapter)

        rca_outcome = rca_svc.handle_raw(rca_payload, actor="test_operator")
        assert rca_outcome.http_status == 200, f"RCA failed: {rca_outcome.response}"
        rca_resp = rca_outcome.response
        assert rca_resp["status"] == "success"
        print_pass(f"RCA completed with status '{rca_resp['status']}'")
        print_pass(f"Causes identified: {len(rca_resp['possible_causes'])} | Actions: {len(rca_resp['recommended_actions'])} | Citations: {len(rca_resp['citations'])}")
        print_pass(f"Cited document: {rca_resp['citations'][0]['document_id']} v{rca_resp['citations'][0]['version']} ({rca_resp['citations'][0]['title']})")

        # ---------------------------------------------------------------------
        # Step 6: Module D (Cryptographic SHA-256 Hash Chaining)
        # ---------------------------------------------------------------------
        print_step(6, "Module D: Cryptographic SHA-256 Hash-Chained Audit Ledger")
        with TestingSessionLocal() as db:
            events = db.execute(select(ModuleDAuditEvent).order_by(ModuleDAuditEvent.seq.asc())).scalars().all()
            print_pass(f"Total audit ledger events recorded: {len(events)}")
            for ev in events:
                target = f"{ev.target_type}:{ev.target_id[:8]}" if ev.target_id else f"{ev.target_type}:None"
                print(f"    - Seq #{ev.seq:02d} | Action: {ev.action:<20} | Target: {target:<20} | Hash: {ev.hash[:12]}...")

        # ---------------------------------------------------------------------
        # Step 7: Ledger Integrity Verification
        # ---------------------------------------------------------------------
        print_step(7, "Ledger Integrity: Full Cryptographic Chain Verification")
        with TestingSessionLocal() as db:
            v_res = verify_chain(db)
            assert v_res.valid is True, f"Chain invalid: {v_res.failure_reason}"
            print_pass(f"verify_chain(): VALID (Total: {v_res.total_events}, Verified: {v_res.verified_events}, Last Seq: {v_res.last_seq})")
            print_pass(f"Chain tip hash: {v_res.last_hash}")

        # ---------------------------------------------------------------------
        # Step 8: Negative Verification (Tamper Detection & Schema Rejection)
        # ---------------------------------------------------------------------
        print_step(8, "Negative Verification: Tamper Detection & Safety Gates")

        # 8a: Tamper detection
        with TestingSessionLocal() as db:
            db.execute(text("UPDATE audit_events SET metadata_json = '{\"forged\": 1}' WHERE action = 'rca.completed'"))
            db.commit()
            tamper_res = verify_chain(db)
            assert tamper_res.valid is False, "Tamper was not detected!"
            assert "Content altered" in tamper_res.failure_reason
            print_pass(f"Tamper detection: Successfully detected altered record at Seq #{tamper_res.first_failure_seq} ({tamper_res.failure_reason})")

        # 8b: Unsupported class rejection
        bd_app.dependency_overrides[get_detector] = lambda: FakeDetector([det("unknown_defect", 0.9, (10, 10, 40, 30))])
        p_res2 = client.post("/predict", files={"file": ("c2.png", img_payload, "image/png")}, headers=auth_header)
        iid2 = p_res2.json()["inspection_id"]
        neg_rca_resp = client.get(f"/inspections/{iid2}/rca-request", headers=auth_header)
        assert neg_rca_resp.status_code == 422
        print_pass(f"Unsupported defect class safety gate: Cleanly rejected with HTTP 422: {neg_rca_resp.json()['detail']['unsupported_classes']}")

        print("\n\033[1;32m=================================================================\033[0m")
        print("\033[1;32m  ALL 8 END-TO-END PLATFORM INTEGRATION CHECKS PASSED (A-B-C-D)  \033[0m")
        print("\033[1;32m=================================================================\033[0m\n")
        return 0

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
