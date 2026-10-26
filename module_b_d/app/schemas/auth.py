from datetime import datetime

from pydantic import BaseModel


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    role: str
    username: str


class MeResponse(BaseModel):
    id: str
    username: str
    full_name: str
    role: str
    site: str
    permissions: list[str]
    token_expires_at: datetime


class MessageResponse(BaseModel):
    detail: str
