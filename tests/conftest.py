from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.core.config import Settings
from backend.db.base import Base
from backend.main import create_app
from backend.services.reference_service import seed_sample_data

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None, environment="test", database_url=f"sqlite:///{tmp_path / 'test.db'}",
        storage_path=tmp_path / "images", sample_data_path=ROOT / "data/samples/dark_store_sample.json",
        seed_sample_data=False, max_upload_mb=1, log_level="WARNING", api_key=None,
    )


@pytest.fixture
def app(settings: Settings):
    app = create_app(settings)
    Base.metadata.create_all(app.state.engine)
    with app.state.session_factory() as session:
        seed_sample_data(session, settings.sample_data_path)
    return app


@pytest.fixture
def client(app) -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_payload() -> dict:
    return {"store_id": "DS-BLR-001", "sku_id": "SKU-MILK-500ML", "stage": "PACKING",
            "order_id": "ORD-100245", "operator_id": "OP-17"}
