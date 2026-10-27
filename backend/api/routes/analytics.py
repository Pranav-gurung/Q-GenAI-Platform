"""Operations intelligence (Part 7): deterministic quality analytics. Read-only."""
from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query

from backend.api.dependencies import get_analytics_service
from backend.api.errors import ERROR_RESPONSES
from backend.schemas.analytics import AnalyticsOverview
from backend.schemas.common import ID_PATTERN
from backend.services.analytics_service import AnalyticsFilters, AnalyticsService

router = APIRouter(prefix="/analytics", tags=["analytics"], responses=ERROR_RESPONSES)


@router.get("/overview", response_model=AnalyticsOverview, response_model_by_alias=True,
            summary="Operations intelligence: quality metrics, hotspots, trends, feedback quality",
            description="Deterministic aggregation of stored inspections and operator feedback for a period (UTC "
                        "dates, inclusive; default: the 30 days up to the latest inspection). Every rate states its "
                        "denominator in `definitions` and is null when the denominator is zero. Hotspots and trends "
                        "describe observed inspection patterns and should not be interpreted as causal proof.")
def analytics_overview(
    service: Annotated[AnalyticsService, Depends(get_analytics_service)],
    start_date: date | None = None,
    end_date: date | None = None,
    store_id: Annotated[str | None, Query(pattern=ID_PATTERN)] = None,
    sku_id: Annotated[str | None, Query(pattern=ID_PATTERN)] = None,
    granularity: Literal["day", "week"] | None = None,
) -> AnalyticsOverview:
    return service.overview(AnalyticsFilters(start_date=start_date, end_date=end_date, store_id=store_id,
                                             sku_id=sku_id, granularity=granularity))
