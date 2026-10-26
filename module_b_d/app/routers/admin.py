"""Administration: users, model versions, audit log + integrity verification."""
import json

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.dependencies import require_permission
from app.models.audit_event import AuditEvent
from app.models.model_version import ModelVersion
from app.models.user import Role, User
from app.schemas.audit import (
    AuditEventList,
    ModelPatch,
    ModelRegister,
    ModelVersionOut,
    VerificationOut,
    VerifyRequest,
)
from app.schemas.users import UserCreate, UserOut, UserUpdate
from app.services.audit_service import (
    AuditWriteError,
    append_event,
    audited_transaction,
    create_checkpoint,
    verify_chain,
)
from app.services.authentication import hash_password
from app.services.authorization import Permission, audit_scope_filter
from app.services.detector import detector_manager, resolve_weights_path
from app.utils.hashing import sha256_file

router = APIRouter(prefix="/admin", tags=["Administration"])

admin_users = require_permission(Permission.USER_MANAGE)
admin_models = require_permission(Permission.MODEL_MANAGE)
admin_audit = require_permission(Permission.AUDIT_VERIFY)


# ----------------------------------------------------------------- users ---
@router.get("/users", response_model=list[UserOut], summary="List users")
def list_users(admin: User = Depends(admin_users), db: Session = Depends(get_db)):
    return list(db.execute(select(User).order_by(User.created_at)).scalars())


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED,
             summary="Create a user and assign a role")
def create_user(body: UserCreate, admin: User = Depends(admin_users), db: Session = Depends(get_db)):
    if db.execute(select(User.id).where(User.username == body.username)).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "Username already exists.")
    try:
        with audited_transaction(db):
            user = User(username=body.username, full_name=body.full_name, role=body.role,
                        site=body.site, password_hash=hash_password(body.password))
            db.add(user)
            db.flush()
            append_event(db, "user.created", actor=admin, target_type="user", target_id=user.id,
                         metadata={"username": user.username, "role": user.role.value, "site": user.site})
    except AuditWriteError as exc:
        if isinstance(exc.__cause__, IntegrityError):
            raise HTTPException(status.HTTP_409_CONFLICT, "Username already exists.") from None
        raise
    return user


@router.patch("/users/{user_id}", response_model=UserOut,
              summary="Change role, site, active flag, name or password")
def update_user(user_id: str, body: UserUpdate, admin: User = Depends(admin_users),
                db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")
    changes = body.model_dump(exclude_unset=True)
    if user.id == admin.id and (changes.get("is_active") is False or
                                changes.get("role", Role.ADMIN) != Role.ADMIN):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "You cannot disable or demote your own administrator account.")

    audit_changes: dict = {}
    with audited_transaction(db):
        for field in ("full_name", "site", "is_active", "role"):
            if field in changes and changes[field] is not None and getattr(user, field) != changes[field]:
                old = getattr(user, field)
                audit_changes[field] = {"old": getattr(old, "value", old),
                                        "new": getattr(changes[field], "value", changes[field])}
                setattr(user, field, changes[field])
        if changes.get("password"):
            user.password_hash = hash_password(changes["password"])
            audit_changes["password"] = "changed"  # value never logged
        if {"role", "is_active", "password", "site"} & audit_changes.keys():
            user.token_version += 1  # invalidates the user's existing tokens
        if audit_changes:
            action = "user.role_changed" if "role" in audit_changes else "user.updated"
            append_event(db, action, actor=admin, target_type="user", target_id=user.id,
                         metadata={"username": user.username,
                                   # key names avoid the redaction regex on purpose; no secret values here:
                                   "changed_fields": {k: v for k, v in audit_changes.items() if k != "password"},
                                   "reset_by_admin": "password" in audit_changes})
    return user


# ---------------------------------------------------------------- models ---
@router.get("/models", summary="Registered model versions and weight files on disk")
def list_models(admin: User = Depends(admin_models), db: Session = Depends(get_db),
                settings: Settings = Depends(get_settings)):
    registered = [ModelVersionOut.model_validate(m).model_dump()
                  for m in db.execute(select(ModelVersion).order_by(ModelVersion.created_at)).scalars()]
    files = sorted(p.name for p in settings.models_dir.glob("*") if p.suffix.lower() in {".pt", ".onnx"})
    current = detector_manager.current
    return {"registered": registered, "weight_files": files,
            "active": current.info.public_dict() if current else None}


@router.post("/models", response_model=ModelVersionOut, status_code=status.HTTP_201_CREATED,
             summary="Register a weights file from the models directory")
def register_model(body: ModelRegister, admin: User = Depends(admin_models),
                   db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    try:
        path = resolve_weights_path(settings.models_dir, body.weights_file)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Weights file not found in the models directory.")
    if db.execute(select(ModelVersion.id).where(ModelVersion.version_label == body.version_label)).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "Version label already registered.")
    with audited_transaction(db):
        mv = ModelVersion(version_label=body.version_label, weights_file=body.weights_file,
                          sha256=sha256_file(path), notes=body.notes, approved=body.approved,
                          registered_by=admin.id)
        db.add(mv)
        db.flush()
        append_event(db, "model.registered", actor=admin, target_type="model", target_id=mv.id,
                     metadata={"version_label": mv.version_label, "weights_file": mv.weights_file,
                               "model_sha256": mv.sha256, "approved": mv.approved})
    return mv


@router.patch("/models/{model_id}", response_model=ModelVersionOut, summary="Approve / unapprove a model")
def patch_model(model_id: str, body: ModelPatch, admin: User = Depends(admin_models),
                db: Session = Depends(get_db)):
    mv = db.get(ModelVersion, model_id)
    if mv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model version not found.")
    with audited_transaction(db):
        old = mv.approved
        mv.approved = body.approved
        append_event(db, "model.approval_changed", actor=admin, target_type="model", target_id=mv.id,
                     metadata={"version_label": mv.version_label, "old_value": old, "new_value": body.approved})
    if mv.is_active and detector_manager.current:
        detector_manager.current.approved = mv.approved
        detector_manager.current.info.approved = mv.approved
    return mv


@router.post("/models/{model_id}/activate", summary="Load this model version for all new inspections")
def activate_model(model_id: str, admin: User = Depends(admin_models), db: Session = Depends(get_db),
                   settings: Settings = Depends(get_settings)):
    mv = db.get(ModelVersion, model_id)
    if mv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model version not found.")
    path = resolve_weights_path(settings.models_dir, mv.weights_file)
    if not path.is_file() or sha256_file(path) != mv.sha256:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "Weights file is missing or its SHA-256 no longer matches the registered checksum.")
    info = detector_manager.load(settings.models_dir, mv.weights_file, mv.version_label,
                                 settings.inference_device, settings.inference_image_size,
                                 mv.approved, keep_previous_on_failure=True)
    if info.status != "loaded":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Model could not be loaded: {info.message}")
    with audited_transaction(db):
        db.execute(update(ModelVersion).values(is_active=False))
        mv.is_active = True
        mv.class_names = json.dumps(info.class_names)
        append_event(db, "model.activated", actor=admin, target_type="model", target_id=mv.id,
                     metadata={"version_label": mv.version_label, "model_sha256": mv.sha256,
                               "class_names": info.class_names, "is_generic": info.is_generic})
    return info.public_dict()


# ----------------------------------------------------------------- audit ---
def list_audit_events(db: Session, user: User, action: str | None, target_id: str | None,
                      limit: int, offset: int) -> dict:
    conds = [audit_scope_filter(user)]
    if action:
        conds.append(AuditEvent.action == action)
    if target_id:
        conds.append(AuditEvent.target_id == target_id)
    total = db.execute(select(func.count()).select_from(AuditEvent).where(*conds)).scalar_one()
    rows = db.execute(select(AuditEvent).where(*conds).order_by(AuditEvent.seq.desc())
                      .limit(limit).offset(offset)).scalars()
    return {"total": total, "items": [{
        "seq": e.seq, "event_id": e.event_id, "timestamp": e.timestamp, "actor_id": e.actor_id,
        "actor_role": e.actor_role, "action": e.action, "target_type": e.target_type,
        "target_id": e.target_id, "metadata": json.loads(e.metadata_json),
        "prev_hash": e.prev_hash, "hash": e.hash} for e in rows]}


@router.get("/audit-logs", response_model=AuditEventList, summary="Full audit log (newest first)")
def admin_audit_logs(
    action: str | None = Query(None, max_length=64),
    target_id: str | None = Query(None, max_length=64),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    admin: User = Depends(admin_audit),
    db: Session = Depends(get_db),
):
    return list_audit_events(db, admin, action, target_id, limit, offset)


@router.post("/audit-logs/verify", response_model=VerificationOut,
             summary="Verify the SHA-256 hash chain (optionally against a signed checkpoint)")
def verify_audit_logs(body: VerifyRequest | None = Body(None), admin: User = Depends(admin_audit),
                      db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    checkpoint = body.checkpoint.model_dump() if body and body.checkpoint else None
    result = verify_chain(db, checkpoint, settings.audit_checkpoint_key)
    # The verification itself is audited (appended AFTER the checked range).
    with audited_transaction(db):
        append_event(db, "audit.verified", actor=admin, target_type="audit_log",
                     metadata={"valid": result.valid, "verified_events": result.verified_events,
                               "total_events": result.total_events,
                               "first_failure_seq": result.first_failure_seq,
                               "failure_reason": result.failure_reason,
                               "checkpoint_checked": result.checkpoint_checked})
    return result.to_dict()


@router.post("/audit-logs/checkpoint", summary="Create an HMAC-signed checkpoint of the chain head")
def audit_checkpoint(admin: User = Depends(admin_audit), db: Session = Depends(get_db),
                     settings: Settings = Depends(get_settings)):
    try:
        cp = create_checkpoint(db, settings.audit_checkpoint_key)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    with audited_transaction(db):
        append_event(db, "audit.checkpoint_created", actor=admin, target_type="audit_log",
                     metadata={"checkpoint_seq": cp["seq"], "checkpoint_head": cp["hash"]})
    return {**cp, "instructions": "Store this JSON OUTSIDE the application database "
                                  "(e.g. a separate server, WORM storage or printed record)."}
