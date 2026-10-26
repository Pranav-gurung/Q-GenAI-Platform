import io

import pytest
from PIL import Image
from sqlalchemy import select

from app.models.audit_event import AuditEvent
from app.utils.image_utils import ImageValidationError, draw_detections, validate_and_decode
from tests.conftest import FakeDetector, image_bytes, predict

MB = 1024 * 1024


def v(data, name="x.png", max_bytes=10 * MB, max_pixels=40_000_000):
    return validate_and_decode(data, name, max_bytes, max_pixels)


def test_valid_png_and_jpeg():
    img = v(image_bytes("PNG", (30, 20)))
    assert (img.width, img.height, img.format) == (30, 20, "PNG")
    assert img.rgb.shape == (20, 30, 3) and len(img.sha256) == 64
    assert v(image_bytes("JPEG"), "a.JPEG").format == "JPEG"


def test_rgba_png_is_converted_to_rgb():
    buf = io.BytesIO()
    Image.new("RGBA", (10, 10), (1, 2, 3, 100)).save(buf, "PNG")
    assert v(buf.getvalue()).rgb.shape == (10, 10, 3)


@pytest.mark.parametrize("data,name,msg", [
    (b"", "x.png", "empty"),
    (b"hello world, not an image", "x.png", "could not be decoded"),
    (image_bytes("PNG"), "x.gif", "Only .jpg"),
    (image_bytes("PNG"), "noextension", "Only .jpg"),
    (image_bytes("PNG")[:40], "x.png", "could not be decoded"),  # truncated
])
def test_invalid_inputs(data, name, msg):
    with pytest.raises(ImageValidationError, match=msg):
        v(data, name)


def test_gif_renamed_to_jpg_is_rejected():
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, "GIF")
    with pytest.raises(ImageValidationError, match="not a JPEG or PNG"):
        v(buf.getvalue(), "fake.jpg")


def test_size_and_pixel_limits():
    with pytest.raises(ImageValidationError, match="too large"):
        v(image_bytes(), max_bytes=10)
    with pytest.raises(ImageValidationError, match="pixel count"):
        v(image_bytes("PNG", (100, 100)), max_pixels=5000)


def test_draw_detections_returns_png():
    img = v(image_bytes("PNG", (100, 100)))
    png = draw_detections(img.rgb, [{"class_id": 0, "class_name": "scratch", "confidence": 0.9,
                                     "bbox": {"x1": 5, "y1": 5, "x2": 50, "y2": 50}}])
    assert png[:4] == b"\x89PNG"


def test_api_rejects_invalid_upload_and_audits_it(client, headers, use_detector, db):
    use_detector(FakeDetector())
    r = predict(client, headers["op"], data=b"%PDF-1.4 not an image", filename="doc.png")
    assert r.status_code == 400
    assert "could not be decoded" in r.json()["detail"]
    assert "Traceback" not in r.text and "/home" not in r.text
    assert db.execute(select(AuditEvent).where(AuditEvent.action == "inspection.rejected")).first()


def test_api_rejects_oversized_upload(client, headers, use_detector, monkeypatch):
    from app.config import get_settings
    use_detector(FakeDetector())
    monkeypatch.setattr(get_settings(), "max_upload_mb", 0.0001)  # ~100 bytes
    r = predict(client, headers["op"], data=image_bytes("PNG", (200, 200), (1, 2, 3)))
    assert r.status_code == 400 and "too large" in r.json()["detail"]
