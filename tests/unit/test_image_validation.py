import pytest

from backend.core.exceptions import InvalidImageError
from backend.services.image_validation import sanitize_filename, validate_image
from tests.utils import corrupt_png, image_bytes

LIMITS = dict(allowed_types=["jpeg", "png", "webp"], max_bytes=1024 * 1024, max_pixels=40_000_000)


@pytest.mark.parametrize("fmt,name,mime", [("PNG", "a.png", "image/png"), ("JPEG", "a.jpg", "image/jpeg"),
                                           ("JPEG", "a.JPEG", "image/jpeg"), ("WEBP", "a.webp", "image/webp")])
def test_valid_images_accepted(fmt, name, mime):
    img = validate_image(image_bytes(fmt), name, mime, **LIMITS)
    assert (img.width, img.height, img.mime_type) == (64, 48, mime)
    assert len(img.sha256) == 64


def _code(data, name, ctype=None, **overrides):
    with pytest.raises(InvalidImageError) as exc:
        validate_image(data, name, ctype, **{**LIMITS, **overrides})
    return exc.value.code


def test_invalid_extension_rejected():
    assert _code(image_bytes("PNG"), "a.gif") == "UNSUPPORTED_IMAGE_TYPE"
    assert _code(image_bytes("PNG"), "noext") == "UNSUPPORTED_IMAGE_TYPE"
    assert _code(b"#!/bin/sh\necho hi", "evil.sh") == "UNSUPPORTED_IMAGE_TYPE"


def test_extension_must_match_content():
    assert _code(image_bytes("PNG"), "a.jpg") == "UNSUPPORTED_IMAGE_TYPE"


def test_declared_content_type_must_match():
    assert _code(image_bytes("PNG"), "a.png", "image/jpeg") == "UNSUPPORTED_IMAGE_TYPE"


def test_disallowed_format_rejected():
    assert _code(image_bytes("WEBP"), "a.webp", allowed_types=["jpeg", "png"]) == "UNSUPPORTED_IMAGE_TYPE"


def test_non_image_with_image_extension_rejected():
    assert _code(b"this is not an image", "a.png") == "CORRUPT_IMAGE"


def test_oversized_file_rejected():
    assert _code(image_bytes("PNG", (64, 64)), "a.png", max_bytes=100) == "IMAGE_TOO_LARGE"


def test_too_many_pixels_rejected():
    assert _code(image_bytes("PNG", (200, 200)), "a.png", max_pixels=10_000) == "IMAGE_TOO_LARGE"


def test_corrupt_image_rejected():
    assert _code(corrupt_png(), "a.png") == "CORRUPT_IMAGE"


def test_empty_file_rejected():
    assert _code(b"", "a.png") == "EMPTY_FILE"


def test_filename_sanitised():
    assert sanitize_filename("../../etc/passwd.png") == "passwd.png"
    assert sanitize_filename("my photo (1).jpg") == "my_photo_1_.jpg"
    assert sanitize_filename(None) == "upload"
