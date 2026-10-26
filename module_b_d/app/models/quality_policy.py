"""Versioned inspection policy.

Every change creates a NEW row (new version). Old versions are never edited,
so each inspection can point at the exact policy that produced its decision.
"""
from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.user import utcnow


class QualityPolicy(Base):
    __tablename__ = "quality_policies"
    __table_args__ = (
        CheckConstraint("operator_min_threshold <= default_confidence_threshold", name="ck_min_le_default"),
        CheckConstraint("default_confidence_threshold <= operator_max_threshold", name="ck_default_le_max"),
        CheckConstraint("max_allowed_defects >= 0", name="ck_max_allowed"),
    )

    version: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    change_reason: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Detection threshold handed to the model (detections below it are dropped)
    default_confidence_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    # Range an OPERATOR may choose from. Managers/admins may go outside it,
    # which is recorded as a threshold override.
    operator_min_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    operator_max_threshold: Mapped[float] = mapped_column(Float, nullable=False)

    # A detection counts as "disqualifying" if its class is in fail_classes
    # (empty list = every non-ignored class) and confidence >= fail_confidence.
    fail_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    fail_classes: Mapped[str] = mapped_column(Text, nullable=False, default="[]")     # JSON list
    ignored_classes: Mapped[str] = mapped_column(Text, nullable=False, default="[]")  # JSON list
    # FAIL when the number of disqualifying detections exceeds this.
    max_allowed_defects: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # If true, results from a model that is not admin-approved go to REVIEW.
    require_approved_model: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
