"""Module D Audit Adapter for Module C RCA events.

Conforms to Module C's AuditSink protocol while delegating all writes to
Module D's tamper-evident SHA-256 hash-chained audit service.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.audit_event import AuditEvent as ModuleDAuditEvent
from app.services.audit_service import AuditWriteError, record_standalone

try:
    from module_c.audit import AuditError, AuditEvent as ModuleCAuditEvent, AuditSink
except ImportError:
    # Fallback protocol / error definition if module_c is not directly on sys.path
    class AuditError(RuntimeError):  # type: ignore[no-redef]
        pass

    class AuditSink:  # type: ignore[no-redef]
        pass

    ModuleCAuditEvent = Any  # type: ignore

log = logging.getLogger(__name__)


class _ActorWrapper:
    """Carries actor metadata to populate actor_id and actor_role in AuditEvent."""

    def __init__(self, actor_id: str | None, role: str | None = None):
        self.id = actor_id
        self.role = role


class ModuleDAuditAdapter:
    """Bridges Module C RCA events into Module D's hash-chained SQLite audit store.

    Guarantees:
    - Never bypasses Module D's audit service (uses record_standalone and audited_transaction).
    - Preserves SHA-256 hash chaining, prev_hash linking, and sequence monotonicity.
    - Synchronous and durable persistence: if DB transaction fails, raises AuditError so
      Module C can withhold results (fail-closed) or warn (fail-open).
    - Redacts sensitive keys and bounds metadata; never stores raw document text, prompts,
      or image bytes.
    """

    def __init__(self, session_factory: Callable[[], Session]):
        if not callable(session_factory):
            raise TypeError("session_factory must be a callable returning a SQLAlchemy Session.")
        self.session_factory = session_factory

    def record(self, event: ModuleCAuditEvent) -> ModuleDAuditEvent:
        """Record a Module C AuditEvent into Module D's tamper-evident audit chain."""
        if not hasattr(event, "event_type") or not hasattr(event, "request_id"):
            raise ValueError(f"Invalid audit event structure: {event!r}")

        action = str(event.event_type)
        target_type = "inspection" if getattr(event, "inspection_id", None) else (
            "image" if getattr(event, "image_id", None) else "rca_request"
        )
        target_id = (
            getattr(event, "inspection_id", None)
            or getattr(event, "image_id", None)
            or getattr(event, "request_id", None)
        )

        actor_val = getattr(event, "actor", None)
        actor_obj = None
        if actor_val is not None:
            if hasattr(actor_val, "id"):
                actor_obj = actor_val
            else:
                actor_obj = _ActorWrapper(str(actor_val), role=getattr(actor_val, "role", None))

        # Safe, structured metadata (bounded, no secrets, no document text)
        metadata: dict[str, Any] = {
            "request_id": getattr(event, "request_id", None),
            "module_c_event_id": getattr(event, "event_id", None),
            "image_id": getattr(event, "image_id", None),
            "model_version": getattr(event, "model_version", None),
            "rca_status": getattr(event, "rca_status", None),
            "http_status": getattr(event, "http_status", None),
            "error_code": getattr(event, "error_code", None),
            "defect_classes": list(getattr(event, "defect_classes", []) or []),
            "cited_documents": getattr(event, "cited_documents", []) or [],
            "n_possible_causes": getattr(event, "n_possible_causes", 0),
            "n_recommended_actions": getattr(event, "n_recommended_actions", 0),
            "llm_provider": getattr(event, "llm_provider", None),
            "llm_model": getattr(event, "llm_model", None),
        }

        # Filter out None values for clean canonical JSON serialization
        metadata = {k: v for k, v in metadata.items() if v is not None}

        try:
            with self.session_factory() as db:
                recorded_ev = record_standalone(
                    db,
                    action=action,
                    actor=actor_obj,
                    target_type=target_type,
                    target_id=target_id,
                    metadata=metadata,
                )
                return recorded_ev
        except AuditWriteError as exc:
            log.error("Module D audit write error: %s", exc)
            raise AuditError(f"Module D audit write failed: {exc}") from exc
        except SQLAlchemyError as exc:
            log.error("Module D audit database error: %s", exc)
            raise AuditError(f"Module D audit database failure: {exc}") from exc
        except Exception as exc:
            log.error("Unexpected error in Module D audit adapter: %s", exc)
            raise AuditError(f"Module D audit adapter failure: {exc}") from exc
