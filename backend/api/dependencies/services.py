from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from backend.core.config import Settings
from backend.services.analytics_service import AnalyticsService
from backend.services.feedback_service import FeedbackService
from backend.services.inspection_service import InspectionService
from backend.services.learning_service import LearningService


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> Iterator[Session]:
    session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_inspection_service(request: Request, db: Session = Depends(get_db)) -> InspectionService:
    state = request.app.state
    return InspectionService(db, state.storage, state.orchestrator, state.settings)


def get_feedback_service(db: Session = Depends(get_db),
                         inspections: InspectionService = Depends(get_inspection_service)) -> FeedbackService:
    return FeedbackService(db, inspections)


def get_learning_service(db: Session = Depends(get_db)) -> LearningService:
    return LearningService(db)


def get_analytics_service(db: Session = Depends(get_db)) -> AnalyticsService:
    return AnalyticsService(db)
