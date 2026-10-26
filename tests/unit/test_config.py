import pytest
from pydantic import ValidationError

from backend.core.config import Settings


def test_csv_env_values(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test")
    monkeypatch.setenv("ALLOWED_IMAGE_TYPES", "jpg,png")
    s = Settings(_env_file=None)
    assert s.cors_origins == ["http://a.test", "http://b.test"]
    assert s.allowed_image_types == ["jpeg", "png"]


def test_empty_api_key_disables_auth(monkeypatch):
    monkeypatch.setenv("API_KEY", "")
    assert Settings(_env_file=None).api_key is None


def test_unknown_image_type_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, allowed_image_types=["gif"])


def test_production_guards():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="production", cors_origins=["*"],
                 database_url="postgresql+psycopg://u:p@h/db")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="production", database_url="sqlite:///x.db")
    Settings(_env_file=None, environment="production", database_url="postgresql+psycopg://u:p@h/db")


def test_unknown_module_engine_rejected():
    from backend.pipeline.registry import build_modules
    with pytest.raises(ValueError, match="CV_ENGINE='yolo-not-yet'"):
        build_modules(Settings(_env_file=None, cv_engine="yolo-not-yet"))
