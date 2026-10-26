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
    quality_gate_engine: str = "stub"
    cv_engine: str = "stub"
    decision_engine: str = "stub"
    pipeline_analysis_engine: str = "stub"
    rca_engine: str = "stub"
    action_engine: str = "stub"
    model_path: Path = Path("models/")

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
