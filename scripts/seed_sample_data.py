"""Load the SIMULATED stores/SKUs in data/samples/ into the database (idempotent).

    python -m scripts.seed_sample_data            # uses DATABASE_URL
"""
from __future__ import annotations

from backend.core.config import get_settings
from backend.core.logging import configure_logging
from backend.db.session import build_engine, build_session_factory
from backend.services.reference_service import seed_sample_data


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    engine = build_engine(settings.database_url)
    with build_session_factory(engine)() as session:
        stores, skus = seed_sample_data(session, settings.sample_data_path)
    engine.dispose()
    print(f"seeded {stores} stores and {skus} SKUs (existing rows left unchanged)")


if __name__ == "__main__":
    main()
