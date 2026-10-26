"""Stores / SKUs reference data and sample-data seeding."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.db.models import Sku, Store
from backend.domain.fulfillment import FULFILLMENT_FLOW, STAGE_DESCRIPTIONS
from backend.pipeline.states import PipelineState
from backend.schemas.reference import ReferenceData, SkuOut, StageOut, StoreOut

logger = logging.getLogger(__name__)


def get_reference_data(session: Session) -> ReferenceData:
    return ReferenceData(
        stores=[StoreOut.model_validate(s) for s in session.scalars(select(Store).order_by(Store.store_id))],
        skus=[SkuOut.model_validate(s) for s in session.scalars(select(Sku).order_by(Sku.sku_id))],
        stages=[StageOut(stage=s.value, position=i, description=STAGE_DESCRIPTIONS[s])
                for i, s in enumerate(FULFILLMENT_FLOW)],
        pipeline_states=[s.value for s in PipelineState],
    )


def seed_sample_data(session: Session, path: Path) -> tuple[int, int]:
    """Insert simulated stores/SKUs that don't exist yet. Idempotent. Returns (stores_added, skus_added)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    existing_stores = set(session.scalars(select(Store.store_id)))
    existing_skus = set(session.scalars(select(Sku.sku_id)))
    stores = [Store(**s) for s in data["stores"] if s["store_id"] not in existing_stores]
    skus = [Sku(**s) for s in data["skus"] if s["sku_id"] not in existing_skus]
    session.add_all([*stores, *skus])
    session.commit()
    if stores or skus:
        logger.info("seeded sample data: %d stores, %d skus", len(stores), len(skus))
    return len(stores), len(skus)


def is_empty(session: Session) -> bool:
    return (session.scalar(select(func.count()).select_from(Store)) or 0) == 0
