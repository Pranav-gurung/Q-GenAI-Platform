import io

import numpy as np
import pytest
from PIL import Image, ImageFilter

from backend.core.config import Settings
from backend.domain.enums import ModuleStatus
from ml.cv.preprocessing import PreprocessingError, decode_image, rgb_to_bgr, to_model_input
from ml.cv.quality import BasicImageQualityGate, find_issues, laplacian_variance, measure
from tests.cv.helpers import image_input, textured

SETTINGS = Settings(_env_file=None)


def encode(img: Image.Image, fmt="PNG", **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


class TestPreprocessing:
    def test_decode_rgb_and_bgr(self):
        data = encode(Image.new("RGB", (30, 20), (255, 0, 0)))
        rgb = decode_image(data)
        assert rgb.shape == (20, 30, 3) and rgb.dtype == np.uint8 and tuple(rgb[0, 0]) == (255, 0, 0)
        bgr, w, h = to_model_input(data)
        assert (w, h) == (30, 20) and tuple(bgr[0, 0]) == (0, 0, 255) and bgr.flags["C_CONTIGUOUS"]
        assert np.array_equal(rgb_to_bgr(rgb)[:, :, 0], rgb[:, :, 2])

    def test_grayscale_and_rgba_become_rgb(self):
        assert decode_image(encode(Image.new("L", (8, 8), 128))).shape == (8, 8, 3)
        assert decode_image(encode(Image.new("RGBA", (8, 8), (1, 2, 3, 4)))).shape == (8, 8, 3)

    def test_exif_orientation_applied(self):
        img = Image.new("RGB", (40, 20), (0, 255, 0))
        exif = img.getexif()
        exif[0x0112] = 6  # rotate 90 CW on display
        rgb = decode_image(encode(img, "JPEG", exif=exif.tobytes()))
        assert rgb.shape[:2] == (40, 20)  # height/width swapped: boxes refer to the upright image

    def test_deterministic(self):
        data = encode(textured((64, 64)))
        assert np.array_equal(decode_image(data), decode_image(data))

    def test_undecodable(self):
        with pytest.raises(PreprocessingError):
            decode_image(b"not an image")


class TestQualityGate:
    def test_normal_textured_image_accepted(self):
        r = BasicImageQualityGate(SETTINGS).assess(image_input(encode(textured((320, 240)))))
        assert r.status is ModuleStatus.COMPLETED and r.acceptable is True and r.issues == []
        assert set(r.metrics) == {"width", "height", "aspect_ratio", "blur_variance", "brightness"}

    @pytest.mark.parametrize("img,issue", [
        (textured((100, 300)), "LOW_RESOLUTION"),
        (textured((1600, 200)), "EXTREME_ASPECT"),
        (textured((320, 240)).filter(ImageFilter.GaussianBlur(8)), "BLURRY"),
        (Image.new("RGB", (320, 240), (5, 5, 5)), "TOO_DARK"),
        (Image.new("RGB", (320, 240), (252, 252, 252)), "TOO_BRIGHT"),
    ])
    def test_unusable_images_rejected(self, img, issue):
        r = BasicImageQualityGate(SETTINGS).assess(image_input(encode(img)))
        assert r.acceptable is False and issue in r.issues and r.is_real

    def test_thresholds_are_configurable(self):
        img = textured((320, 240)).filter(ImageFilter.GaussianBlur(8))
        metrics = measure(decode_image(encode(img)))
        assert "BLURRY" in find_issues(metrics, SETTINGS)
        assert "BLURRY" not in find_issues(metrics, Settings(_env_file=None, cv_blur_threshold=0))

    def test_laplacian(self):
        assert laplacian_variance(np.full((10, 10), 100, np.uint8)) == 0
        assert laplacian_variance(np.indices((10, 10)).sum(0) % 2 * 255) > 1000
