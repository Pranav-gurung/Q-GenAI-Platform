"""Test fixtures.

Every test runs against a TEMPORARY SQLite database and storage directory,
with test-only secrets. No real credentials are used anywhere.
"""
import io
import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="qgenai-test-"))
os.environ.update({
    "QGENAI_ENVIRONMENT": "test",
    "QGENAI_JWT_SECRET_KEY": "test-only-secret-key-not-for-production-0123456789",
    "QGENAI_DATABASE_URL": f"sqlite:///{_TMP / 'test.db'}",
    "QGENAI_STORAGE_DIR": str(_TMP / "storage"),
    "QGENAI_MODELS_DIR": str(_TMP / "models"),
    "QGENAI_MODEL_WEIGHTS": "does_not_exist.pt",
    "QGENAI_AUDIT_CHECKPOINT_KEY": "test-only-checkpoint-key",
    "QGENAI_LOGIN_MAX_ATTEMPTS": "5",
})

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.database import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models.user import Role, User  # noqa: E402
from app.routers.auth import get_login_limiter  # noqa: E402
from app.services.authentication import hash_password  # noqa: E402
from app.services.detector import Detection, ModelInfo, get_detector  # noqa: E402
from app.services.policy_service import ensure_default_policy  # noqa: E402

PASSWORD = "TestPassw0rd!"  # test-only credential


class FakeDetector:
    """Stub used ONLY in tests to feed known detections into the pipeline."""

    def __init__(self, detections=None, status="loaded", is_generic=False, approved=True):
        self.info = ModelInfo(status=status, message="test stub", name="stub.pt", version="test-1",
                              sha256="ab" * 32, class_names=["scratch", "crack", "rust"],
                              approved=approved, is_generic=is_generic, backend="stub", device="cpu")
        self.detections = detections or []

    def predict(self, rgb, conf):
        return [d for d in self.detections if d.confidence >= conf]


def det(cls="scratch", conf=0.9, box=(10, 10, 50, 50)):
    names = ["scratch", "crack", "rust"]
    return Detection(class_id=names.index(cls) if cls in names else 0, class_name=cls,
                     confidence=conf, x1=box[0], y1=box[1], x2=box[2], y2=box[3])


def image_bytes(fmt="PNG", size=(64, 48), color=(120, 120, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:  # runs startup (creates tables, loads "missing" model)
        yield c


@pytest.fixture(autouse=True)
def fresh_db(client):
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        ensure_default_policy(db)
    get_login_limiter().clear()
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def db():
    s = SessionLocal()
    yield s
    s.close()


def make_user(username, role: Role, site="plant-a", password=PASSWORD) -> User:
    with SessionLocal() as s:
        u = User(username=username, full_name=username.title(), role=role, site=site,
                 password_hash=hash_password(password))
        s.add(u)
        s.commit()
        s.refresh(u)
        return u


def login(client, username, password=PASSWORD) -> dict:
    r = client.post("/auth/login", data={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def users():
    return {
        "admin": make_user("admin1", Role.ADMIN),
        "op": make_user("operator1", Role.OPERATOR),
        "op2": make_user("operator2", Role.OPERATOR),
        "mgr": make_user("manager1", Role.QUALITY_MANAGER),
        "mgr_b": make_user("manager_b", Role.QUALITY_MANAGER, site="plant-b"),
        "op_b": make_user("operator_b", Role.OPERATOR, site="plant-b"),
    }


@pytest.fixture
def headers(client, users):
    return {k: login(client, u.username) for k, u in users.items()}


@pytest.fixture
def use_detector():
    def _set(detector):
        app.dependency_overrides[get_detector] = lambda: detector
        return detector
    return _set


def predict(client, hdrs, data=None, filename="part.png", **form):
    files = {"file": (filename, data if data is not None else image_bytes(), "image/png")}
    return client.post("/predict", headers=hdrs, files=files,
                       data={k: str(v) for k, v in form.items()})
