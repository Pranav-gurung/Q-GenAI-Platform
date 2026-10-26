"""Operator feedback: submit -> review -> approve/reject -> export approved -> (explicit) apply to a NEW version.

Unreviewed (pending) or rejected feedback is never applied. Storage is a single JSONL file written
atomically; swap this class for a DB-backed one if several writers are expected.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schema import Annotation, ImageRecord, bbox_problem

STATUSES = ("pending", "approved", "rejected")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class FeedbackRecord:
    feedback_id: str
    image_id: str
    dataset_version: str
    original_class: Optional[str]
    corrected_class: Optional[str]
    corrected_bbox: Optional[list] = None
    corrected_mask_ref: Optional[str] = None
    annotation_index: Optional[int] = None  # which annotation of the image is corrected (None = new/unspecified)
    reviewer_feedback: str = ""
    submitted_by: str = ""
    submitted_at: str = ""
    review_status: str = "pending"
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    review_notes: str = ""


class FeedbackStore:
    def __init__(self, path: Path, classes: Optional[list] = None):
        self.path = Path(path)
        self.classes = set(classes) if classes else None

    def _load(self) -> list:
        if not self.path.exists():
            return []
        return [FeedbackRecord(**json.loads(ln)) for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def _save(self, recs: list) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(asdict(r), sort_keys=True) + "\n")
        os.replace(tmp, self.path)

    def submit(self, image_id: str, dataset_version: str, original_class: Optional[str], corrected_class: Optional[str],
               *, corrected_bbox: Optional[list] = None, corrected_mask_ref: Optional[str] = None,
               annotation_index: Optional[int] = None, reviewer_feedback: str = "", submitted_by: str,
               image_size: Optional[tuple] = None) -> FeedbackRecord:
        if not image_id or not dataset_version or not submitted_by:
            raise ValueError("image_id, dataset_version and submitted_by are required")
        if corrected_class is None and corrected_bbox is None and corrected_mask_ref is None:
            raise ValueError("feedback must correct at least one of: class, bbox, mask")
        if corrected_class is not None and self.classes is not None and corrected_class not in self.classes:
            raise ValueError(f"corrected_class {corrected_class!r} is not a known class")
        if corrected_bbox is not None:
            w, h = image_size if image_size else (None, None)
            problem = bbox_problem(corrected_bbox, w, h)
            if problem:
                raise ValueError(f"invalid corrected_bbox: {problem}")
        rec = FeedbackRecord(feedback_id=uuid.uuid4().hex[:12], image_id=image_id, dataset_version=dataset_version,
                             original_class=original_class, corrected_class=corrected_class,
                             corrected_bbox=list(corrected_bbox) if corrected_bbox else None,
                             corrected_mask_ref=corrected_mask_ref, annotation_index=annotation_index,
                             reviewer_feedback=reviewer_feedback, submitted_by=submitted_by, submitted_at=_now())
        recs = self._load()
        recs.append(rec)
        self._save(recs)
        return rec

    def list(self, status: Optional[str] = None) -> list:
        return [r for r in self._load() if status is None or r.review_status == status]

    def review(self, feedback_id: str, decision: str, reviewer: str, notes: str = "") -> FeedbackRecord:
        if decision not in ("approved", "rejected"):
            raise ValueError("decision must be 'approved' or 'rejected'")
        if not reviewer:
            raise ValueError("a reviewer identity is required")
        recs = self._load()
        for r in recs:
            if r.feedback_id == feedback_id:
                if r.review_status != "pending":
                    raise ValueError(f"feedback {feedback_id} already {r.review_status}")
                if reviewer == r.submitted_by:
                    raise ValueError("the submitter cannot approve their own correction (four-eyes rule)")
                r.review_status, r.reviewed_by, r.reviewed_at, r.review_notes = decision, reviewer, _now(), notes
                self._save(recs)
                return r
        raise KeyError(f"unknown feedback id {feedback_id}")

    def export_approved(self, out_path: Path, dataset_version: Optional[str] = None) -> int:
        approved = [r for r in self._load() if r.review_status == "approved"
                    and (dataset_version is None or r.dataset_version == dataset_version)]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps([asdict(r) for r in approved], indent=2))
        return len(approved)


def apply_approved(records: list, feedback: list) -> tuple:
    """Return (new_records, changelog). Only ``approved`` items are applied; inputs are not mutated.

    A correction targets annotation ``annotation_index``; if None and the image has exactly one
    annotation it targets that one; otherwise it is skipped and reported (never guessed).
    """
    feedback_recs = [FeedbackRecord(**fb) if isinstance(fb, dict) else fb for fb in feedback]
    records_list = [ImageRecord.from_dict(r) if isinstance(r, dict) else r for r in records]
    by_id = {r.image_id: ImageRecord.from_dict(r.to_dict()) for r in records_list}
    log = []
    for fb in feedback_recs:
        if fb.review_status != "approved":
            log.append({"feedback_id": fb.feedback_id, "applied": False, "reason": f"status={fb.review_status}"})
            continue
        rec = by_id.get(fb.image_id)
        if rec is None:
            log.append({"feedback_id": fb.feedback_id, "applied": False, "reason": "image not in dataset"})
            continue
        idx = fb.annotation_index
        if idx is None and len(rec.annotations) == 1:
            idx = 0
        if idx is None or not (0 <= idx < len(rec.annotations)):
            log.append({"feedback_id": fb.feedback_id, "applied": False, "reason": "ambiguous or invalid annotation_index"})
            continue
        a = rec.annotations[idx]
        before = a.to_dict()
        if fb.corrected_class is not None:
            a.class_name = fb.corrected_class
        if fb.corrected_bbox is not None:
            a.bbox = list(fb.corrected_bbox)
        if fb.corrected_mask_ref is not None:
            a.mask_ref = fb.corrected_mask_ref
        log.append({"feedback_id": fb.feedback_id, "applied": True, "image_id": fb.image_id, "before": before, "after": a.to_dict()})
    return list(by_id.values()), log
