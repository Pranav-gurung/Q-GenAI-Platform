"""Section D: centralised role-based access control.

All permission decisions live here. Routers ask "does this user have
permission X?" or "may this user see inspection Y?" - they never compare role
strings themselves, and the frontend is never trusted.
"""
import enum

from sqlalchemy import or_, select
from sqlalchemy.sql import ColumnElement

from app.models.audit_event import AuditEvent
from app.models.inspection import Inspection
from app.models.user import Role, User


class Permission(str, enum.Enum):
    INSPECTION_RUN = "inspection:run"
    INSPECTION_VIEW_OWN = "inspection:view_own"
    INSPECTION_VIEW_SITE = "inspection:view_site"
    INSPECTION_VIEW_ALL = "inspection:view_all"
    INSPECTION_OVERRIDE = "inspection:override"      # record manual review decision
    THRESHOLD_OVERRIDE = "quality:threshold_override"  # go outside operator threshold range
    POLICY_VIEW = "policy:view"
    POLICY_EDIT = "policy:edit"
    USER_MANAGE = "user:manage"
    MODEL_MANAGE = "model:manage"
    AUDIT_VIEW_SITE = "audit:view_site"
    AUDIT_VIEW_ALL = "audit:view_all"
    AUDIT_VERIFY = "audit:verify"


P = Permission
ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.OPERATOR: frozenset({P.INSPECTION_RUN, P.INSPECTION_VIEW_OWN, P.POLICY_VIEW}),
    Role.QUALITY_MANAGER: frozenset({
        P.INSPECTION_RUN, P.INSPECTION_VIEW_OWN, P.INSPECTION_VIEW_SITE,
        P.INSPECTION_OVERRIDE, P.THRESHOLD_OVERRIDE, P.POLICY_VIEW, P.AUDIT_VIEW_SITE,
    }),
    # Separation of duties: administrators manage the system and audit it,
    # but do not inspect products or change quality decisions.
    Role.ADMIN: frozenset({
        P.INSPECTION_VIEW_ALL, P.POLICY_VIEW, P.POLICY_EDIT, P.USER_MANAGE,
        P.MODEL_MANAGE, P.AUDIT_VIEW_ALL, P.AUDIT_VERIFY,
    }),
}


def has_permission(user: User, permission: Permission) -> bool:
    return user.is_active and permission in ROLE_PERMISSIONS.get(user.role, frozenset())


def permissions_for(user: User) -> list[str]:
    return sorted(p.value for p in ROLE_PERMISSIONS.get(user.role, frozenset()))


# ---- Object-level checks (prevents insecure direct object references) -----
def inspection_scope_filter(user: User) -> ColumnElement[bool]:
    """SQL condition restricting inspections to what `user` may see."""
    if has_permission(user, P.INSPECTION_VIEW_ALL):
        return Inspection.id.isnot(None)  # everything
    if has_permission(user, P.INSPECTION_VIEW_SITE):
        return or_(Inspection.site == user.site, Inspection.user_id == user.id)
    if has_permission(user, P.INSPECTION_VIEW_OWN):
        return Inspection.user_id == user.id
    return Inspection.id.is_(None)  # nothing


def can_view_inspection(user: User, insp: Inspection) -> bool:
    if has_permission(user, P.INSPECTION_VIEW_ALL):
        return True
    if has_permission(user, P.INSPECTION_VIEW_SITE) and insp.site == user.site:
        return True
    return has_permission(user, P.INSPECTION_VIEW_OWN) and insp.user_id == user.id


def can_override_inspection(user: User, insp: Inspection) -> tuple[bool, str]:
    if not has_permission(user, P.INSPECTION_OVERRIDE):
        return False, "Your role cannot record review decisions."
    if insp.site != user.site:
        return False, "Inspection is outside your site."
    if insp.user_id == user.id:
        return False, "Separation of duties: you cannot review your own inspection."
    return True, ""


def audit_scope_filter(user: User) -> ColumnElement[bool]:
    if has_permission(user, P.AUDIT_VIEW_ALL):
        return AuditEvent.seq.isnot(None)
    if has_permission(user, P.AUDIT_VIEW_SITE):
        site_users = select(User.id).where(User.site == user.site).scalar_subquery()
        return AuditEvent.actor_id.in_(site_users)
    return AuditEvent.seq.is_(None)
