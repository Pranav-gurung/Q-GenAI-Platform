"""Inspection records.

The original AI result (ai_* columns, defects_json) is written once and never
updated. Human decisions live in the separate `inspection_overrides` table;
`final_decision` is a convenience column that always equals the most recent
override's new_decision, or ai_decision when no override exists.
"""
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.user import new_uuid, utcnow

DECISIONS = ("PASS", "FAIL", "REVIEW")
_DECISION_SQL = "('PASS','FAIL','REVIEW')"


class Inspection(Base):
    __tablename__ = "inspections"
    __table_args__ = (
        CheckConstraint(f"ai_decision IN {_DECISION_SQL}", name="ck_ai_decision"),
        CheckConstraint(f"final_decision IN {_DECISION_SQL}", name="ck_final_decision"),
        CheckConstraint(
            "confidence_threshold > 0 AND confidence_threshold < 1", name="ck_conf_range"
        ),
        CheckConstraint("defect_count >= 0", name="ck_defect_count"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    site: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Image metadata only - the image itself is not stored by default.
    image_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    image_width: Mapped[int] = mapped_column(Integer, nullable=False)
    image_height: Mapped[int] = mapped_column(Integer, nullable=False)
    annotated_image_file: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Model provenance
    model_status: Mapped[str] = mapped_column(String(32), nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Settings in force
    confidence_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_overridden: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    policy_version: Mapped[int] = mapped_column(ForeignKey("quality_policies.version"), nullable=False)

    # ORIGINAL AI RESULT - immutable after creation
    ai_decision: Mapped[str] = mapped_column(String(8), nullable=False)
    ai_decision_reasons: Mapped[str] = mapped_column(Text, nullable=False)  # JSON list
    defects_json: Mapped[str] = mapped_column(Text, nullable=False)          # JSON list
    defect_count: Mapped[int] = mapped_column(Integer, nullable=False)
    processing_time_ms: Mapped[float] = mapped_column(Float, nullable=False)   # whole pipeline
    inference_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)  # model only

    # Current human-reviewed outcome (see module docstring)
    final_decision: Mapped[str] = mapped_column(String(8), nullable=False)

    overrides: Mapped[list["InspectionOverride"]] = relationship(
        back_populates="inspection", order_by="InspectionOverride.created_at"
    )


class InspectionOverride(Base):
    __tablename__ = "inspection_overrides"
    __table_args__ = (
        CheckConstraint(f"old_decision IN {_DECISION_SQL}", name="ck_old_decision"),
        CheckConstraint(f"new_decision IN {_DECISION_SQL}", name="ck_new_decision"),
        CheckConstraint("length(reason) >= 10", name="ck_reason_len"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspections.id"), nullable=False, index=True)
    actor_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    actor_role: Mapped[str] = mapped_column(String(32), nullable=False)
    old_decision: Mapped[str] = mapped_column(String(8), nullable=False)
    new_decision: Mapped[str] = mapped_column(String(8), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    inspection: Mapped[Inspection] = relationship(back_populates="overrides")
