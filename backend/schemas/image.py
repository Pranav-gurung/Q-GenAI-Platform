"""Image metadata (what the pipeline knows about an inspection image)."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ImageMetadata(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    image_id: str
    inspection_id: str
    file_name: str = Field(description="Sanitised original file name (display only).")
    mime_type: str = Field(examples=["image/jpeg"])
    size_bytes: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str = Field(description="Content hash; used for de-duplication, not security.")
    storage_backend: str = Field(examples=["local"])
    storage_reference: str = Field(description="Backend-specific opaque reference (path / object key).")
    created_at: datetime
