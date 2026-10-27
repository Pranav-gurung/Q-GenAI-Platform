from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api.dependencies import get_db
from backend.schemas.reference import ReferenceData
from backend.services.reference_service import get_reference_data

router = APIRouter(tags=["reference"])


@router.get("/reference-data", response_model=ReferenceData,
            summary="Stores, SKUs, fulfillment stages and pipeline states (for operator UIs)")
def reference_data(db: Session = Depends(get_db)) -> ReferenceData:
    return get_reference_data(db)
