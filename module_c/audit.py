"""Audit adapter for Module D. This is an INTERFACE plus a minimal JSONL reference sink, not a security control.

Events never contain secrets, API keys or document passage text; only identifiers, statuses and cited
document/chunk ids. Module D can implement ``AuditSink`` (e.g. signed/append-only store) and inject it.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

from .config import Settings


class AuditError(RuntimeError):
    """Raised when an event cannot be durably recorded."""


@dataclass
class AuditEvent:
    event_type: str  # rca.completed | rca.rejected | rca.failed | ingest.document
    request_id: str
    inspection_id: Optional[str] = None
    image_id: Optional[str] = None
    model_version: Optional[str] = None
    rca_status: Optional[str] = None
    http_status: Optional[int] = None
    error_code: Optional[str] = None
    defect_classes: list = field(default_factory=list)
    cited_documents: list = field(default_factory=list)  # [{"document_id","version","chunk_ids":[...]}]
    n_possible_causes: int = 0
    n_recommended_actions: int = 0
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    actor: Optional[str] = None  # filled by Module D auth integration
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds"))

    def to_dict(self) -> dict:
        return asdict(self)


class AuditSink(Protocol):
    def record(self, event: AuditEvent) -> None: ...


class JsonlAuditSink:
    def __init__(self, path: Path):
        self.path = Path(path)

    def record(self, event: AuditEvent) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            raise AuditError(f"audit write failed: {type(e).__name__}") from e


class InMemoryAuditSink:
    def __init__(self):
        self.events: list = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


class NullAuditSink:
    """Explicit opt-out (RCA_AUDIT_SINK=none). Use only in development."""

    def record(self, event: AuditEvent) -> None:
        return None


def make_audit_sink(s: Settings) -> AuditSink:
    if s.audit_sink == "jsonl":
        return JsonlAuditSink(s.audit_path)
    if s.audit_sink == "none":
        return NullAuditSink()
    if s.audit_sink == "module_d":
        try:
            from app.database import SessionLocal
            from app.services.audit_adapter import ModuleDAuditAdapter
            return ModuleDAuditAdapter(SessionLocal)
        except ImportError as e:
            raise AuditError(f"Cannot initialize module_d audit sink: {e}") from e
    raise ValueError(f"unknown RCA_AUDIT_SINK {s.audit_sink!r}")
