from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.user import Role
from app.services.authentication import validate_password_strength

USERNAME_PATTERN = r"^[A-Za-z0-9_.-]{3,64}$"
SITE_PATTERN = r"^[A-Za-z0-9_.-]{1,64}$"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    username: str
    full_name: str
    role: Role
    site: str
    is_active: bool
    created_at: datetime


class UserCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(pattern=USERNAME_PATTERN)
    full_name: str = Field("", max_length=128)
    password: str = Field(min_length=1, max_length=128)
    role: Role
    site: str = Field("default", pattern=SITE_PATTERN)

    @field_validator("password")
    @classmethod
    def _strong(cls, v: str) -> str:
        validate_password_strength(v)
        return v


class UserUpdate(BaseModel):
    """Admin-only partial update. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    full_name: str | None = Field(None, max_length=128)
    role: Role | None = None
    site: str | None = Field(None, pattern=SITE_PATTERN)
    is_active: bool | None = None
    password: str | None = Field(None, max_length=128)

    @field_validator("password")
    @classmethod
    def _strong(cls, v: str | None) -> str | None:
        if v is not None:
            validate_password_strength(v)
        return v
