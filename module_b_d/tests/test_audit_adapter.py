import concurrent.futures
import json
from pathlib import Path
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError

import sys
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from module_c.audit import AuditError, AuditEvent as ModuleCAuditEvent
from module_c.config import Settings as ModuleCSettings
from module_c.models import Defect, RCARequest
from module_c.rca import RCAService

from app.database import SessionLocal
from app.models.audit_event import AuditEvent as ModuleDAuditEvent
from app.services.audit_adapter import ModuleDAuditAdapter
from app.services.audit_service import (
    create_checkpoint,
    verify_chain,
)


KEY = "test-only-checkpoint-key-secret-32ch"


@pytest.fixture
def adapter():
    return ModuleDAuditAdapter(SessionLocal)


def test_adapter_emits_rca_completed_with_citations(fresh_db, adapter):
    ev = ModuleCAuditEvent(
        event_type="rca.completed",
        request_id="req-101",
        inspection_id="insp-202",
        image_id="img-303",
        model_version="yolo-v8-1.0",
        rca_status="success",
        http_status=200,
        defect_classes=["crack", "scratch"],
        cited_documents=[
            {"document_id": "SOP-12", "version": "2.1", "chunk_ids": ["c1", "c2"]}
        ],
        n_possible_causes=2,
        n_recommended_actions=2,
        llm_provider="mock",
        llm_model="stub-model",
        actor="user-operator-1",
    )

    recorded = adapter.record(ev)
    # Note: fresh_db fixture inserts the default policy (seq=1), so this event is seq=2
    assert recorded.seq >= 2
    assert recorded.action == "rca.completed"
    assert recorded.target_type == "inspection"
    assert recorded.target_id == "insp-202"
    assert recorded.actor_id == "user-operator-1"

    meta = json.loads(recorded.metadata_json)
    assert meta["request_id"] == "req-101"
    assert meta["rca_status"] == "success"
    assert meta["http_status"] == 200
    assert meta["defect_classes"] == ["crack", "scratch"]
    assert meta["cited_documents"] == [{"document_id": "SOP-12", "version": "2.1", "chunk_ids": ["c1", "c2"]}]
    assert meta["n_possible_causes"] == 2
    assert meta["llm_provider"] == "mock"

    with SessionLocal() as session:
        res = verify_chain(session)
        assert res.valid is True
        assert res.verified_events == res.total_events


def test_adapter_emits_insufficient_evidence(fresh_db, adapter):
    ev = ModuleCAuditEvent(
        event_type="rca.completed",
        request_id="req-102",
        inspection_id="insp-203",
        rca_status="insufficient_evidence",
        http_status=200,
        defect_classes=["porosity"],
        cited_documents=[],
        n_possible_causes=0,
        n_recommended_actions=0,
    )
    adapter.record(ev)

    with SessionLocal() as session:
        row = session.execute(select(ModuleDAuditEvent).where(ModuleDAuditEvent.target_id == "insp-203")).scalar_one()
        assert row.action == "rca.completed"
        meta = json.loads(row.metadata_json)
        assert meta["rca_status"] == "insufficient_evidence"
        assert meta["cited_documents"] == []
        assert verify_chain(session).valid is True


def test_adapter_emits_rca_failed_and_rejected(fresh_db, adapter):
    # 1. Failed RCA event
    ev_fail = ModuleCAuditEvent(
        event_type="rca.failed",
        request_id="req-fail-1",
        inspection_id="insp-fail-1",
        rca_status="error",
        http_status=504,
        error_code="llm_timeout",
    )
    adapter.record(ev_fail)

    # 2. Rejected RCA event (validation failure)
    ev_reject = ModuleCAuditEvent(
        event_type="rca.rejected",
        request_id="req-reject-1",
        http_status=422,
        error_code="invalid_request",
    )
    adapter.record(ev_reject)

    with SessionLocal() as session:
        events = session.execute(
            select(ModuleDAuditEvent)
            .where(ModuleDAuditEvent.action.in_(["rca.failed", "rca.rejected"]))
            .order_by(ModuleDAuditEvent.seq)
        ).scalars().all()
        assert len(events) == 2
        assert events[0].action == "rca.failed" and events[0].target_id == "insp-fail-1"
        assert events[1].action == "rca.rejected" and events[1].target_id == "req-reject-1"
        assert verify_chain(session).valid is True


def test_adapter_security_never_logs_passages_or_secrets(fresh_db, adapter):
    ev = ModuleCAuditEvent(
        event_type="rca.completed",
        request_id="req-sec",
        inspection_id="insp-sec",
        cited_documents=[{"document_id": "doc1", "version": "1.0", "chunk_ids": ["c1"]}],
    )
    adapter.record(ev)

    with SessionLocal() as session:
        row = session.execute(select(ModuleDAuditEvent).where(ModuleDAuditEvent.target_id == "insp-sec")).scalar_one()
        meta_str = row.metadata_json

        # Must not contain document text passages or secrets
        assert "passage" not in meta_str
        assert "password" not in meta_str
        assert "secret" not in meta_str
        assert "api_key" not in meta_str
        assert "image_bytes" not in meta_str


def test_adapter_events_subject_to_tamper_detection(fresh_db, adapter):
    ev1 = ModuleCAuditEvent(event_type="rca.completed", request_id="req-1", inspection_id="insp-1")
    ev2 = ModuleCAuditEvent(event_type="rca.completed", request_id="req-2", inspection_id="insp-2")
    adapter.record(ev1)
    adapter.record(ev2)

    with SessionLocal() as session:
        assert verify_chain(session).valid is True

        # Malicious alteration of event 1
        session.execute(text("UPDATE audit_events SET metadata_json = '{\"tampered\": true}' WHERE seq = 1"))
        session.commit()

        res = verify_chain(session)
        assert res.valid is False
        assert res.first_failure_seq == 1
        assert "Content altered" in res.failure_reason


def test_adapter_failure_raises_audit_error(fresh_db):
    def broken_session_factory():
        raise OperationalError("Simulated database disk failure", params={}, orig=Exception("Disk full"))

    broken_adapter = ModuleDAuditAdapter(broken_session_factory)
    ev = ModuleCAuditEvent(event_type="rca.completed", request_id="req-err")

    with pytest.raises(AuditError, match="Module D audit"):
        broken_adapter.record(ev)


def test_rca_service_fail_closed_with_module_d_adapter(fresh_db):
    class StubRetriever:
        def search(self, query):
            return [], 0

        def has_evidence(self, passages):
            return len(passages) > 0

    class StubProvider:
        name = "stub"
        model = "stub-v1"
        def generate(self, *a, **k):
            return {"possible_causes": [], "recommended_actions": [], "unknowns": []}

    def broken_session_factory():
        raise OperationalError("Simulated DB connection lost", params={}, orig=Exception("Lost connection"))

    broken_adapter = ModuleDAuditAdapter(broken_session_factory)
    settings = ModuleCSettings(audit_fail_mode="fail_closed")
    service = RCAService(settings, StubRetriever(), StubProvider(), broken_adapter)

    req = RCARequest(
        inspection_id="insp-fail-closed",
        image_id="img-1",
        model_version="v1",
        defects=[Defect("crack", 0.9, [10, 10, 20, 20])],
    )
    outcome = service.analyze(req)

    # In fail_closed mode, audit failure MUST result in HTTP 500 and result withholding
    assert outcome.http_status == 500
    assert outcome.response["status"] == "error"
    assert outcome.response["error"]["code"] == "audit_failure"
    assert outcome.response["possible_causes"] == []


def test_rca_service_fail_open_with_module_d_adapter(fresh_db):
    class StubRetriever:
        def search(self, query):
            return [], 0

        def has_evidence(self, passages):
            return len(passages) > 0

    class StubProvider:
        name = "stub"
        model = "stub-v1"
        def generate(self, *a, **k):
            return {"possible_causes": [], "recommended_actions": [], "unknowns": []}

    def broken_session_factory():
        raise OperationalError("Simulated DB connection lost", params={}, orig=Exception("Lost connection"))

    broken_adapter = ModuleDAuditAdapter(broken_session_factory)
    settings = ModuleCSettings(audit_fail_mode="fail_open")
    service = RCAService(settings, StubRetriever(), StubProvider(), broken_adapter)

    req = RCARequest(
        inspection_id="insp-fail-open",
        image_id="img-1",
        model_version="v1",
        defects=[],
    )
    outcome = service.analyze(req)

    # In fail_open mode, audit failure produces HTTP 200 with AUDIT WARNING in notes
    assert outcome.http_status == 200
    assert any("AUDIT WARNING" in note for note in outcome.response.get("notes", []))


def test_checkpoint_includes_adapter_events(fresh_db, adapter):
    ev = ModuleCAuditEvent(event_type="rca.completed", request_id="req-cp", inspection_id="insp-cp")
    adapter.record(ev)

    with SessionLocal() as session:
        cp = create_checkpoint(session, KEY)
        assert cp["seq"] >= 2
        assert "signature" in cp

        # Verify chain with checkpoint
        res = verify_chain(session, cp, KEY)
        assert res.valid is True
        assert res.checkpoint_checked is True


def test_concurrent_audit_adapter_writes(fresh_db, adapter):
    def emit(idx):
        ev = ModuleCAuditEvent(
            event_type="rca.completed",
            request_id=f"req-thread-{idx}",
            inspection_id=f"insp-{idx}",
            n_possible_causes=idx,
        )
        adapter.record(ev)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(emit, i) for i in range(20)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    with SessionLocal() as session:
        res = verify_chain(session)
        assert res.valid is True
        # 1 initial policy event from fresh_db + 20 concurrent adapter events = 21
        assert res.total_events == 21
        assert res.verified_events == 21

        # Verify no duplicate seqs or gaps
        seqs = [row.seq for row in session.execute(select(ModuleDAuditEvent).order_by(ModuleDAuditEvent.seq)).scalars()]
        assert seqs == list(range(1, 22))
