"""Hash-chained audit events.

Every column that is part of the hash is stored as TEXT/INTEGER exactly as it
was hashed (e.g. the timestamp is an ISO-8601 string), so re-computing the
hash later never depends on database type conversions.
"""
from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class AuditEvent(Base):
    __tablename__ = "audit_events"

    # Sequence number: 1, 2, 3 ... with no gaps. Part of the hash, so deleting
    # or reordering events is detectable.
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    event_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    timestamp: Mapped[str] = mapped_column(String(40), nullable=False, index=True)  # UTC ISO-8601
    actor_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    actor_role: Mapped[str | None] = mapped_column(String(32), nullable=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    target_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    target_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    prev_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
