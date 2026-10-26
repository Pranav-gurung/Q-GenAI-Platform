from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AuditEventOut(BaseModel):
    seq: int
    event_id: str
    timestamp: str
    actor_id: str | None
    actor_role: str | None
    action: str
    target_type: str | None
    target_id: str | None
    metadata: dict[str, Any]
    prev_hash: str
    hash: str


class AuditEventList(BaseModel):
    total: int
    items: list[AuditEventOut]


class Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq: int = Field(ge=1)
    hash: str = Field(min_length=64, max_length=64)
    created_at: str
    signature: str = Field(min_length=64, max_length=64)


class VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: Checkpoint | None = None


class VerificationOut(BaseModel):
    valid: bool
    total_events: int
    verified_events: int
    last_seq: int | None
    last_hash: str | None
    first_failure_seq: int | None
    first_failure_event_id: str | None
    failure_reason: str | None
    checkpoint_checked: bool
    checked_at: str


class ModelRegister(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights_file: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,128}$")
    version_label: str = Field(pattern=r"^[A-Za-z0-9_.+-]{1,64}$")
    notes: str = Field("", max_length=2000)
    approved: bool = False


class ModelPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool


class ModelVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: str
    version_label: str
    weights_file: str
    sha256: str
    notes: str
    approved: bool
    is_active: bool
    registered_by: str
