"""Centralised settings, loaded from environment variables (and an optional .env file).

Nothing secret has a real default. Every value can be overridden per environment.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Q-GenAI"
    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    log_format: str = Field("console", pattern="^(console|json)$")

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api"
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:8000", "http://127.0.0.1:8000"]
    serve_frontend: bool = True
    api_key: str | None = None  # optional shared key for write endpoints; None/empty disables auth

    # Database
    database_url: str = "sqlite:///./data/qgenai-dev.db"
    seed_sample_data: bool = False
    sample_data_path: Path = Path("data/samples/dark_store_sample.json")

    # Image storage / validation
    storage_backend: str = "local"
    storage_path: Path = Path("data/raw/images")
    max_upload_mb: float = Field(10, gt=0, le=100)
    allowed_image_types: Annotated[list[str], NoDecode] = ["jpeg", "png", "webp"]
    max_image_pixels: int = Field(40_000_000, gt=0)

    # Pipeline module selection (see backend/pipeline/registry.py)
    quality_gate_engine: str = "stub"     # stub | basic
    cv_engine: str = "stub"               # stub | yolov8
    decision_engine: str = "stub"         # stub | rules
    pipeline_analysis_engine: str = "stub"  # stub | deterministic
    rca_engine: str = "stub"
    action_engine: str = "stub"

    # Computer vision (used when CV_ENGINE=yolov8; see ml/cv/ and README "Computer Vision")
    cv_model_path: Path = Path("artifacts/cv/qgenai-yolov8n-synthetic-v1/model/best.pt")
    cv_model_version: str | None = None   # override; default: artifact metadata.json, else content hash
    cv_device: str = Field("auto", pattern=r"^(auto|cpu|cuda(:\d+)?)$")
    cv_confidence_threshold: float = Field(0.35, ge=0, le=1)
    cv_iou_threshold: float = Field(0.5, gt=0, le=1)
    cv_image_size: int | None = Field(None, ge=32, le=2048)  # None: the size the model was trained at
    cv_max_detections: int = Field(50, ge=1, le=1000)
    cv_taxonomy_path: Path = Path("ml/cv/taxonomy.yaml")
    # Severity: boxes covering >= this fraction of the image escalate severity by one level
    cv_severity_large_area_ratio: float = Field(0.20, gt=0, le=1)

    # Image quality gate (QUALITY_GATE_ENGINE=basic). Conservative: reject only unusable images.
    cv_min_image_width: int = Field(128, ge=1)
    cv_min_image_height: int = Field(128, ge=1)
    cv_max_aspect_ratio: float = Field(6.0, ge=1)
    cv_blur_threshold: float = Field(15.0, ge=0)        # variance of Laplacian (grayscale, long side 512)
    cv_min_brightness: float = Field(20.0, ge=0, le=255)  # mean luminance 0-255
    cv_max_brightness: float = Field(240.0, ge=0, le=255)

    # Decision rules (DECISION_ENGINE=rules) - operational decision, not ML
    decision_fail_min_severity: str = Field("HIGH", pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    decision_fail_min_confidence: float = Field(0.5, ge=0, le=1)
    decision_auto_pass: bool = False      # PASS when CV found nothing; off by default (limited model coverage)

    @field_validator("cors_origins", "allowed_image_types", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("allowed_image_types")
    @classmethod
    def _normalise_types(cls, v: list[str]) -> list[str]:
        supported = {"jpeg", "png", "webp"}
        out = [t.lower().replace("jpg", "jpeg") for t in v]
        unknown = set(out) - supported
        if unknown:
            raise ValueError(f"unsupported image types {sorted(unknown)}; supported: {sorted(supported)}")
        return out

    @field_validator("api_key", mode="before")
    @classmethod
    def _empty_key_is_none(cls, v: object) -> object:
        return v or None

    @model_validator(mode="after")
    def _brightness_order(self) -> "Settings":
        if self.cv_min_brightness >= self.cv_max_brightness:
            raise ValueError("CV_MIN_BRIGHTNESS must be lower than CV_MAX_BRIGHTNESS")
        return self

    @model_validator(mode="after")
    def _production_guards(self) -> "Settings":
        if self.environment is Environment.PRODUCTION:
            if "*" in self.cors_origins:
                raise ValueError("CORS_ORIGINS='*' is not allowed in production")
            if self.database_url.startswith("sqlite"):
                raise ValueError("SQLite is for development/tests only; set a PostgreSQL DATABASE_URL")
        return self

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION


@lru_cache
def get_settings() -> Settings:
    return Settings()
