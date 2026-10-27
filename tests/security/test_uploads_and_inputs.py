"""Uploads (filenames, content, size, resource limits) and API input validation across every endpoint."""
import io
import struct
import zlib
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import func, select

from backend.db.models import Feedback, Inspection, InspectionImage
from tests.utils import corrupt_png, image_bytes


def new_inspection(c, payload):
    return c.post("/api/inspections", json=payload).json()["inspection_id"]


def upload(c, iid, name, data, ctype="image/png"):
    return c.post(f"/api/inspections/{iid}/image", files={"file": (name, data, ctype)})


@pytest.mark.parametrize("name", ["../../../../etc/passwd.png", "/etc/cron.d/evil.png", "..\\..\\windows\\x.png",
                                  "a/b/../../c.png", "....//....//x.png", "😀 photo (1).png", "x" * 300 + ".png",
                                  "con.png", "%2e%2e%2fsecret.png"])
def test_hostile_filenames_cannot_escape_storage(client, valid_payload, settings, name):
    iid = new_inspection(client, valid_payload)
    r = upload(client, iid, name, image_bytes())
    assert r.status_code == 200, r.text
    root = Path(settings.storage_path).resolve()
    with client.app.state.session_factory() as db:
        img = db.scalars(select(InspectionImage).where(InspectionImage.inspection_id == iid)).one()
    stored = (root / img.storage_reference).resolve()
    assert stored.is_relative_to(root) and stored.is_file()
    assert img.storage_reference.startswith(f"{iid}/") and ".." not in img.storage_reference  # server-generated key
    assert "/" not in img.file_name and "\\" not in img.file_name and ".." not in img.file_name
    assert len(img.file_name) <= 100


def test_long_name_keeps_extension_and_nameless_file_is_rejected(client, valid_payload):
    from backend.services.image_validation import sanitize_filename
    assert sanitize_filename("y" * 300 + ".jpeg").endswith(".jpeg") and len(sanitize_filename("y" * 300 + ".jpeg")) == 100
    r = upload(client, new_inspection(client, valid_payload), " .png", image_bytes())  # sanitises to "png": no extension
    assert r.status_code == 422 and r.json()["error"]["code"] == "UNSUPPORTED_IMAGE_TYPE"


def test_storage_refuses_references_outside_root(tmp_path):
    from backend.core.exceptions import StorageError
    from backend.services.storage.local import LocalImageStorage
    st = LocalImageStorage(tmp_path / "root")
    for ref in ("../outside.png", "/etc/passwd", "a/../../x"):
        with pytest.raises(StorageError):
            st.load(ref)
        with pytest.raises(StorageError):
            st.save(ref, b"x", "image/png")
    assert not (tmp_path / "outside.png").exists()


def _gif():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "GIF")
    return buf.getvalue()


def _png_header_only(width, height):
    """A tiny file that CLAIMS huge dimensions (decompression-bomb shape)."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))  # noqa: E731
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00" * 64)) + chunk(b"IEND", b"")


@pytest.mark.parametrize("name,data,ctype,code", [
    ("a.gif", _gif(), "image/gif", "UNSUPPORTED_IMAGE_TYPE"),
    ("a.png", _gif(), "image/png", "UNSUPPORTED_IMAGE_TYPE"),           # GIF disguised as PNG
    ("a.svg", b"<svg xmlns='http://www.w3.org/2000/svg'/>", "image/svg+xml", "UNSUPPORTED_IMAGE_TYPE"),
    ("a.png", b"<svg onload=alert(1)>", "image/png", "CORRUPT_IMAGE"),
    ("a.png", b"MZ\x90\x00 not an image", "image/png", "CORRUPT_IMAGE"),
    ("a.png", corrupt_png(), "image/png", "CORRUPT_IMAGE"),
    ("a.png", b"", "image/png", "EMPTY_FILE"),
    ("a.jpg", image_bytes("PNG"), "image/jpeg", "UNSUPPORTED_IMAGE_TYPE"),  # extension / content mismatch
    ("a.png", image_bytes("PNG"), "text/html", "UNSUPPORTED_IMAGE_TYPE"),  # declared type mismatch
    ("noext", image_bytes("PNG"), "image/png", "UNSUPPORTED_IMAGE_TYPE"),
    ("a.png", _png_header_only(60_000, 60_000), "image/png", "IMAGE_TOO_LARGE"),  # pixel bomb
])
def test_invalid_files_rejected_before_cv(client, valid_payload, name, data, ctype, code):
    iid = new_inspection(client, valid_payload)
    r = upload(client, iid, name, data, ctype)
    assert r.status_code == 422 and r.json()["error"]["code"] == code, r.text
    assert "Traceback" not in r.text and "/home/" not in r.text and "PIL" not in r.text
    assert client.get(f"/api/inspections/{iid}").json()["status"] == "CREATED"  # pipeline never ran


def test_oversized_file_rejected(client, valid_payload, settings):
    iid = new_inspection(client, valid_payload)
    big = image_bytes("PNG", size=(64, 48)) + b"\x00" * (settings.max_upload_bytes + 10)
    r = upload(client, iid, "big.png", big)
    assert r.status_code == 422 and r.json()["error"]["code"] == "IMAGE_TOO_LARGE"


# ------------------------------------------------------------------------------------------------ API inputs
INJECTION = ["' OR '1'='1", "1; DROP TABLE inspections; --", "\" OR \"\"=\"", "%27%20OR%201%3D1", "../../etc",
             "INS-0000000000000000' OR 1=1--", "<script>alert(1)</script>", "${jndi:ldap://x}"]


@pytest.mark.parametrize("bad", INJECTION)
def test_injection_shaped_ids_and_filters(client, valid_payload, bad):
    iid = new_inspection(client, valid_payload)
    for path in (f"/api/inspections/{bad}", f"/api/inspections/{bad}/pipeline", f"/api/inspections/{bad}/rca",
                 f"/api/inspections/{bad}/feedback"):
        r = client.get(path)
        assert r.status_code in (404, 422), (path, r.status_code)
        assert "Traceback" not in r.text and "sqlalchemy" not in r.text.lower()
    for q in ("store_id", "sku_id", "item_id", "order_id"):
        r = client.get("/api/inspections", params={q: bad})
        assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"
    for q in ("store_id", "sku_id"):
        assert client.get("/api/analytics/overview", params={q: bad}).status_code == 422
        assert client.get("/api/feedback/analysis", params={q: bad}).status_code == 422
    r = client.post("/api/inspections", json={**valid_payload, "store_id": bad})
    assert r.status_code == 422
    with client.app.state.session_factory() as db:  # nothing dropped, nothing extra created
        assert db.scalar(select(func.count()).select_from(Inspection)) == 1
    assert client.get(f"/api/inspections/{iid}").status_code == 200


@pytest.mark.parametrize("params", [{"stage": "WAREHOUSE"}, {"status": "DELETED"}, {"limit": 0}, {"limit": 101},
                                    {"offset": -1}, {"limit": "ten"}])
def test_invalid_enums_and_pagination(client, params):
    r = client.get("/api/inspections", params=params)
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize("query,code", [
    ("start_date=2026-02-30", "VALIDATION_ERROR"), ("end_date=26-10-01", "VALIDATION_ERROR"),
    ("start_date=2026-10-22&end_date=2026-10-20", "INVALID_DATE_RANGE"),
    ("start_date=2025-01-01&end_date=2026-01-02", "INVALID_DATE_RANGE"),   # 367 days
    ("granularity=month", "VALIDATION_ERROR"),
])
def test_analytics_parameter_validation(client, query, code):
    r = client.get(f"/api/analytics/overview?{query}")
    assert r.status_code == 422 and r.json()["error"]["code"] == code


def test_analytics_366_days_is_the_maximum(client):
    assert client.get("/api/analytics/overview?start_date=2025-01-01&end_date=2026-01-01").status_code == 200


@pytest.mark.parametrize("body", [
    {"operator_id": "OP-1", "outcome": "SHIP_IT"},
    {"operator_id": "OP-1", "outcome": "FAIL", "defect_types": ["DROP TABLE"]},
    {"operator_id": "OP-1", "outcome": "FAIL", "suspected_stage": "SUPPLIER"},
    {"operator_id": "' OR 1=1 --", "outcome": "FAIL"},
    {"operator_id": "OP-1", "outcome": "FAIL", "learning_status": "USED"},
    {"operator_id": "OP-1", "outcome": "FAIL", "system_outcome": "PASS"},
    {"operator_id": "OP-1", "outcome": "FAIL", "comment": "x" * 1001},
])
def test_invalid_feedback_values(client, valid_payload, body):
    iid = new_inspection(client, valid_payload)
    upload(client, iid, "a.png", image_bytes())
    r = client.post(f"/api/inspections/{iid}/feedback", json=body)
    assert r.status_code == 422
    with client.app.state.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(Feedback)) == 0


def test_decoder_plugin_failure_is_still_a_clean_422(client, valid_payload, monkeypatch):
    """Defense in depth: even if a library globally patches Image.open with something that raises an unexpected
    exception (as Ultralytics does), an undecodable upload is CORRUPT_IMAGE, never a 500."""
    from PIL import Image

    def broken_open(*a, **k):
        raise ModuleNotFoundError("No module named 'pi_heif'")
    monkeypatch.setattr(Image, "open", broken_open)
    r = upload(client, new_inspection(client, valid_payload), "a.png", b"<svg onload=alert(1)>")
    assert r.status_code == 422 and r.json()["error"]["code"] == "CORRUPT_IMAGE"
