"""Shared schema building blocks."""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# External business identifiers (store, SKU, order, operator). Letters, digits, '-', '_'.
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
ExternalId = Annotated[str, StringConstraints(strip_whitespace=True, pattern=ID_PATTERN)]

# Server-generated identifiers, e.g. INS-1A2B3C4D5E6F
INTERNAL_ID_RE = re.compile(r"^(INS|IMG|RES|FB)-[0-9A-F]{16}$")
InspectionId = Annotated[str, StringConstraints(pattern=r"^INS-[0-9A-F]{16}$")]

# Free-form metadata: flat key -> scalar map, bounded so it cannot be abused as a blob store.
MetadataValue = str | int | float | bool | None
Metadata = Annotated[dict[Annotated[str, StringConstraints(max_length=64)], MetadataValue],
                     Field(max_length=50)]


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16].upper()}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    """Base for request bodies: unknown fields are rejected."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ErrorBody(BaseModel):
    code: str = Field(examples=["INVALID_IMAGE"])
    message: str = Field(examples=["Unsupported image format"])
    details: Any = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorBody


T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int
