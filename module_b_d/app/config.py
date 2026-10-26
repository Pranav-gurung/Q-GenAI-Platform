"""Application configuration.

All settings come from environment variables (or a local `.env` file).
Secrets have NO usable defaults: the app refuses to start in production
mode without a strong JWT secret.
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="QGENAI_", extra="ignore", case_sensitive=False,
        protected_namespaces=("settings_",),
    )

    # --- General -----------------------------------------------------------
    app_name: str = "Q-GenAI Inspection API"
    environment: str = Field("development", pattern="^(development|test|production)$")
    debug: bool = False

    # --- Database ----------------------------------------------------------
    # SQLite for the prototype. For PostgreSQL use e.g.
    # postgresql+psycopg://user:pass@host:5432/qgenai
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'storage' / 'qgenai.db'}"

    # --- Authentication ----------------------------------------------------
    jwt_secret_key: str = ""  # REQUIRED - set QGENAI_JWT_SECRET_KEY
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(30, ge=1, le=24 * 60)

    # Login rate limiting (in-memory, per client IP + username)
    login_max_attempts: int = Field(5, ge=1)
    login_window_seconds: int = Field(300, ge=10)

    # --- CORS --------------------------------------------------------------
    # Comma-separated list of allowed origins. Streamlit talks to the API
    # server-side, so browsers normally never need CORS access at all.
    cors_origins: str = "http://localhost:8501"

    # --- Uploads / storage -------------------------------------------------
    max_upload_mb: float = Field(10.0, gt=0, le=100)
    max_image_pixels: int = Field(40_000_000, gt=0)  # decompression-bomb guard
    storage_dir: Path = PROJECT_ROOT / "storage"
    store_original_images: bool = False  # originals are NOT kept by default

    # --- Model -------------------------------------------------------------
    models_dir: Path = PROJECT_ROOT / "models"
    model_weights: str = "defect_model.pt"  # file name inside models_dir
    model_version: str = "unversioned"
    inference_device: str = "cpu"  # "cpu", "0" (first CUDA GPU), "mps"
    inference_image_size: int = Field(640, ge=32, le=2048)

    # --- Audit -------------------------------------------------------------
    # Optional key for signing audit checkpoints (HMAC-SHA256). Keep it
    # somewhere other than the database (see docs/SECURITY.md).
    audit_checkpoint_key: str = ""

    @field_validator("models_dir", "storage_dir")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        return v if v.is_absolute() else (PROJECT_ROOT / v).resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_weights

    def validate_for_startup(self) -> None:
        """Fail fast on insecure configuration."""
        if len(self.jwt_secret_key) < 32:
            raise RuntimeError(
                "QGENAI_JWT_SECRET_KEY must be set to a random value of at least "
                "32 characters. Generate one with: "
                "python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )
        if self.environment == "production" and self.debug:
            raise RuntimeError("QGENAI_DEBUG must be false in production.")
        if self.environment == "production" and "*" in self.cors_origin_list:
            raise RuntimeError("Wildcard CORS origins are not allowed in production.")


@lru_cache
def get_settings() -> Settings:
    return Settings()
