"""Feedback error analysis and learning-candidate export (Part 6). Read-only; nothing is retrained."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from backend.api.dependencies import get_learning_service
from backend.api.errors import ERROR_RESPONSES
from backend.schemas.common import ID_PATTERN
from backend.schemas.learning import FeedbackAnalysis, LearningCandidate
from backend.services.learning_service import FeedbackFilters, LearningService

router = APIRouter(prefix="/feedback", tags=["feedback"], responses=ERROR_RESPONSES)
Learning = Annotated[LearningService, Depends(get_learning_service)]
StoreId = Annotated[str | None, Query(pattern=ID_PATTERN)]


@router.get("/analysis", response_model=FeedbackAnalysis, summary="Error analysis over operator feedback",
            description="Deterministic counts over the latest feedback per inspection: decision overrides, defect and "
                        "stage-attribution corrections (by AI value), RCA / action usefulness. For human review; "
                        "it changes no model, rule or production behaviour.")
def feedback_analysis(learning: Learning, store_id: StoreId = None, sku_id: StoreId = None) -> FeedbackAnalysis:
    return learning.analyze(FeedbackFilters(store_id=store_id, sku_id=sku_id))


@router.get("/learning-candidates", response_model=list[LearningCandidate],
            summary="Export reviewed inspections as dataset / evaluation CANDIDATES",
            description="AI value next to operator value per inspection. Candidates require curation by a person "
                        "before use in any dataset, evaluation or model version.")
def learning_candidates(learning: Learning, store_id: StoreId = None, sku_id: StoreId = None,
                        corrections_only: bool = False,
                        limit: Annotated[int, Query(ge=1, le=1000)] = 500) -> list[LearningCandidate]:
    return learning.candidates(FeedbackFilters(store_id=store_id, sku_id=sku_id), corrections_only=corrections_only,
                               limit=limit)
