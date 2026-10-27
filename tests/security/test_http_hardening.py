"""HTTP layer: security headers, request-id hygiene, early body-size limit, rate limit on expensive endpoints, CORS."""
import io

import pytest
from fastapi.testclient import TestClient

from backend.api.security import RateLimiter, clean_request_id
from backend.db.base import Base
from backend.main import create_app
from backend.services.reference_service import seed_sample_data
from tests.utils import image_bytes


def client_with(settings, **update):
    app = create_app(settings.model_copy(update=update))
    Base.metadata.create_all(app.state.engine)
    with app.state.session_factory() as db:
        seed_sample_data(db, settings.sample_data_path)
    return TestClient(app)


def test_security_headers_on_api_errors_and_static(client):
    for r in (client.get("/api/health"), client.get("/api/inspections/INS-0000000000000000"), client.get("/")):
        assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
        assert r.headers["referrer-policy"] == "no-referrer"
        assert "default-src 'self'" in r.headers["content-security-policy"]
        assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert "content-security-policy" not in client.get("/api/docs").headers  # Swagger UI loads CDN assets


@pytest.mark.parametrize("raw,clean", [("abc-123_X.y", "abc-123_X.y"), ("a b\tc", "abc"),
                                       ("x\" onload=1 <script>", "xonload1script"), ("é" * 5, "")])
def test_request_id_is_sanitised(raw, clean):
    assert clean_request_id(raw) == clean


def test_request_id_echo_is_safe(client):
    r = client.get("/api/health", headers={"X-Request-ID": "id<script>alert(1)</script>"})
    assert r.headers["x-request-id"] == "idscriptalert1script"
    assert len(client.get("/api/health", headers={"X-Request-ID": "x" * 500}).headers["x-request-id"]) == 64
    assert len(client.get("/api/health", headers={"X-Request-ID": "!!!"}).headers["x-request-id"]) == 16  # generated


def test_oversized_upload_rejected_before_parsing(client, valid_payload):
    """settings fixture: MAX_UPLOAD_MB=1 -> body limit 2 MiB; the same 422 IMAGE_TOO_LARGE contract as before."""
    iid = client.post("/api/inspections", json=valid_payload).json()["inspection_id"]
    huge = io.BytesIO(b"\x89PNG" + b"0" * (3 * 1024 * 1024))
    r = client.post(f"/api/inspections/{iid}/image", files={"file": ("big.png", huge, "image/png")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "IMAGE_TOO_LARGE"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert client.get(f"/api/inspections/{iid}").json()["status"] == "CREATED"  # nothing stored, nothing run


def test_streamed_body_without_length_is_counted(client, valid_payload):
    iid = client.post("/api/inspections", json=valid_payload).json()["inspection_id"]

    def chunks():  # no Content-Length: chunked transfer encoding, a valid multipart file part that never ends
        yield (b'--zzz\r\nContent-Disposition: form-data; name="file"; filename="a.png"\r\n'
               b"Content-Type: image/png\r\n\r\n\x89PNG")
        for _ in range(40):
            yield b"x" * 100_000
    r = client.post(f"/api/inspections/{iid}/image", content=chunks(),
                    headers={"content-type": "multipart/form-data; boundary=zzz"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_oversized_json_body_is_413(client):
    r = client.post("/api/inspections", content=b"{" + b" " * (3 * 1024 * 1024) + b"}",
                    headers={"content-type": "application/json"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"
    r = client.post("/api/inspections", content=b"{}", headers={"content-type": "application/json",
                                                                 "content-length": "not-a-number"})
    assert r.status_code in (400, 413)


def test_rate_limit_on_expensive_endpoints(settings, valid_payload):
    with client_with(settings, rate_limit_per_minute=2) as c:
        iid = c.post("/api/inspections", json=valid_payload).json()["inspection_id"]
        codes = [c.get(f"/api/inspections/{iid}/rca").status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        r = c.get(f"/api/inspections/{iid}/rca")
        assert r.json()["error"]["code"] == "RATE_LIMITED" and 0 < r.json()["error"]["details"]["retry_after_seconds"] <= 60
        assert c.post(f"/api/inspections/{iid}/image",
                      files={"file": ("a.png", image_bytes(), "image/png")}).status_code == 429
        assert c.get(f"/api/inspections/{iid}").status_code == 200  # cheap endpoints are not limited
    with client_with(settings, rate_limit_per_minute=0) as c:  # 0 disables
        iid = c.post("/api/inspections", json=valid_payload).json()["inspection_id"]
        assert all(c.get(f"/api/inspections/{iid}/rca").status_code == 200 for _ in range(5))


def test_rate_limiter_window_resets():
    rl = RateLimiter()
    assert [rl.hit("a", 2, now=60.0) for _ in range(3)] == [None, None, 60]
    assert rl.hit("b", 2, now=61.0) is None  # per client
    assert rl.hit("a", 2, now=120.0) is None  # new minute


def test_cors_only_configured_origins(settings):
    with client_with(settings, cors_origins=["http://localhost:8000"]) as c:
        ok = c.options("/api/health", headers={"Origin": "http://localhost:8000", "Access-Control-Request-Method": "GET"})
        bad = c.options("/api/health", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
        assert ok.headers.get("access-control-allow-origin") == "http://localhost:8000"
        assert "access-control-allow-origin" not in bad.headers
        assert "access-control-allow-credentials" not in ok.headers
        assert c.options("/api/inspections", headers={"Origin": "http://localhost:8000",
                                                      "Access-Control-Request-Method": "DELETE"}).status_code == 400


def test_production_configuration_guards():
    from backend.core.config import Settings
    pg = "postgresql+psycopg://u:p@db/x"
    for bad in ({"cors_origins": "*"}, {"database_url": "sqlite:///x.db"}, {"llm_enabled": True, "llm_provider": "mock"}):
        with pytest.raises(ValueError):
            Settings(_env_file=None, environment="production", **{"database_url": pg, **bad})
    s = Settings(_env_file=None, environment="production", database_url=pg, api_key="unit-test-key-xyz",
                 llm_api_key="sk-ant-unit-test-xyz")
    assert "unit-test-key-xyz" not in repr(s) and "sk-ant-unit-test-xyz" not in repr(s)  # secrets masked in reprs
