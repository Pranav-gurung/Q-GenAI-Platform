"""Reference entities: dark stores and SKUs (products)."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base, UTCDateTime
from backend.schemas.common import utcnow


class Store(Base):
    __tablename__ = "stores"

    store_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    city: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Sku(Base):
    __tablename__ = "skus"

    sku_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str | None] = mapped_column(String(100))
    packaging_type: Mapped[str | None] = mapped_column(String(50))  # pouch, tray, can, jar, bottle, box ...
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
