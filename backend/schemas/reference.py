"""Reference data used to populate operator UIs."""
from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict


class StoreOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    store_id: str
    name: str
    city: str | None = None


class SkuOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    sku_id: str
    name: str
    category: str | None = None
    packaging_type: str | None = None


class StageOut(BaseModel):
    stage: str
    position: int
    description: str


class ReferenceData(BaseModel):
    stores: list[StoreOut]
    skus: list[SkuOut]
    stages: list[StageOut]
    pipeline_states: list[str]
    defect_types: list[str] = Field(default_factory=list, description="Platform defect types (for review forms).")
    decision_outcomes: list[str] = Field(default_factory=list, description="PASS / FAIL / REVIEW.")
