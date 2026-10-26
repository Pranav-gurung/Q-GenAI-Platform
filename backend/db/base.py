"""Declarative base and shared column types."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, MetaData
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator

# Deterministic constraint names -> clean Alembic migrations.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# JSONB on PostgreSQL, JSON elsewhere (SQLite in dev/tests).
JSONType = JSON().with_variant(JSONB(), "postgresql")


def str_enum(enum_cls) -> SAEnum:
    """Enum stored as VARCHAR (no native DB enum) so adding values needs no migration."""
    return SAEnum(enum_cls, native_enum=False, length=32, validate_strings=True,
                  values_callable=lambda e: [m.value for m in e], create_constraint=False)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetimes on every backend (SQLite drops tzinfo otherwise)."""
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect):
        if value is not None:
            if value.tzinfo is None:
                raise ValueError("naive datetime; use timezone-aware UTC")
            value = value.astimezone(timezone.utc)
        return value

    def process_result_value(self, value: datetime | None, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
