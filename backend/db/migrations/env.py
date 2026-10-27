"""Alembic environment. URL precedence: -x url=... > config sqlalchemy.url (tests) > DATABASE_URL."""
from __future__ import annotations

from logging.config import fileConfig

from alembic import context
import backend.db.models  # noqa: F401  (registers tables)
from backend.core.config import get_settings
from backend.db.base import Base
from backend.db.session import build_engine

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _url() -> str:
    return (context.get_x_argument(as_dictionary=True).get("url")
            or config.get_main_option("sqlalchemy.url")
            or get_settings().database_url)


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True,
                      render_as_batch=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = build_engine(_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
